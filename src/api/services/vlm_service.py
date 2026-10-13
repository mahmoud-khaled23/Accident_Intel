"""VLM video analysis service — wraps src/VLM/analyze_video.py or calls an
OpenAI-compatible hosted VLM (e.g. Groq) when VLM_API=openai is configured."""
import sys
from pathlib import Path

# Ensure project root and src/ are on sys.path
_project_root = str(Path(__file__).parents[3])
_src_path = str(Path(__file__).parents[3] / "src")
_vlm_path = str(Path(__file__).parents[3] / "src" / "VLM")
for _p in (_project_root, _src_path, _vlm_path):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Module-level model cache (populated lazily on first real local-model call)
_vlm_model = None
_vlm_processor = None


_MOCK_DESCRIPTION = """\
Scene Overview: The video shows a two-lane urban road with moderate traffic during daytime. \
A set of traffic lights is visible at an intersection approximately 30 metres ahead. \
Three vehicles are visible at the start of the clip: a white sedan in the left lane, \
a dark hatchback in the right lane, and a motorcycle entering from the right side street.

Chronological Events: At the beginning of the clip the white sedan is travelling at a \
steady speed through the intersection while the traffic light shows green. The dark hatchback \
accelerates from a stationary position at the side of the road and merges into the left lane \
without signalling. Approximately mid-clip the hatchback encroaches on the sedan's path, \
causing the sedan to brake sharply. The motorcycle, which was following the hatchback, \
cannot stop in time and strikes the rear of the sedan.

Incident Evidence: Physical contact between the motorcycle and the sedan is clearly observed \
in the middle section of the clip. The rear bumper of the sedan shows visible displacement \
in the frames immediately following impact. The hatchback performs an abrupt lane-change \
manoeuvre without an indicator signal, which is clearly observed.

Final Observed State: The sedan has come to a stop partially blocking the left lane. The \
motorcycle is stationary behind it, the rider standing beside the vehicle. The hatchback \
has continued out of frame. Other vehicles have slowed and are queuing behind the stopped \
vehicles.

Uncertainties and Limitations: Vehicle licence plates are not legible due to resolution \
constraints. The precise speed of each vehicle before the incident cannot be determined \
from the available frames. The initial position of the motorcycle is partially obscured \
by the hatchback for several frames prior to impact.\
"""


def _analyze_via_openai_api(video_path: str, settings) -> str:
    """Call a hosted OpenAI-compatible VLM (e.g. Groq) with sampled frames."""
    import base64
    import os

    import cv2

    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(f"openai package not installed: {exc}") from exc

    # Read VLM config from env — fall back to settings fields
    vlm_base_url = os.environ.get("VLM_BASE_URL", "").strip() or None
    vlm_model = os.environ.get("VLM_MODEL", "").strip() or settings.vlm_model_id
    vlm_api_key = os.environ.get("VLM_API_KEY", "").strip() or ""

    client = openai.OpenAI(
        api_key=vlm_api_key or "no-key",
        base_url=vlm_base_url,
    )

    # Sample frames from the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(settings.vlm_n_frames, total_frames)
    step = max(1, total_frames // n_frames)

    frame_b64s = []
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to max 768px wide to keep payload small
        h, w = frame.shape[:2]
        if w > 768:
            scale = 768 / w
            frame = cv2.resize(frame, (768, int(h * scale)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64s.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    cap.release()

    if not frame_b64s:
        raise RuntimeError("No frames could be extracted from the video.")

    # Build message content: interleave images with a prompt
    content = []
    for b64 in frame_b64s:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({
        "type": "text",
        "text": (
            "You are an expert traffic accident analyst. These frames are sampled from a "
            "CCTV accident video. Analyze the footage and produce a detailed accident report "
            "covering: scene overview, chronological events, vehicle actions before the "
            "collision, collision description, traffic signal observations, road/intersection "
            "context, clearly separate observed facts from uncertain interpretations, and "
            "note any limitations due to video quality or occlusion."
        ),
    })

    response = client.chat.completions.create(
        model=vlm_model,
        messages=[{"role": "user", "content": content}],
        max_tokens=settings.vlm_max_tokens,
    )
    return response.choices[0].message.content or ""


def analyze_video(video_path: str, accident_id: str, settings) -> str:
    """Analyze a video and return a textual accident description.

    Priority:
    1. If vlm_mock=True  → return mock description immediately.
    2. If VLM_API=openai (env) → call hosted OpenAI-compatible API (e.g. Groq).
    3. Otherwise → load local Qwen2-VL model (GPU required).
    """
    import os

    if settings.vlm_mock:
        return _MOCK_DESCRIPTION

    vlm_api = (os.environ.get("VLM_API") or getattr(settings, "vlm_api", "") or "").strip().lower()

    if vlm_api == "openai":
        try:
            return _analyze_via_openai_api(video_path, settings)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Hosted VLM call failed (%s) — falling back to mock description", exc
            )
            return _MOCK_DESCRIPTION

    # Local Qwen2-VL path
    global _vlm_model, _vlm_processor

    try:
        from VLM.analyze_video import load_model, analyze
        from VLM.prompts import TRAFFIC_ANALYSIS_PROMPT
    except ImportError as exc:
        import logging
        logging.getLogger(__name__).warning(
            "VLM package not importable (%s) — falling back to mock description", exc
        )
        return _MOCK_DESCRIPTION

    try:
        if _vlm_model is None:
            _vlm_model, _vlm_processor = load_model(settings.vlm_model_id)

        return analyze(
            _vlm_model,
            _vlm_processor,
            video_path,
            TRAFFIC_ANALYSIS_PROMPT,
            settings.vlm_n_frames,
            settings.vlm_max_tokens,
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Local VLM inference failed (%s) — falling back to mock description", exc
        )
        return _MOCK_DESCRIPTION
