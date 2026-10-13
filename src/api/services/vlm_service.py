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


def _analysis_prompt(tracking_data: dict | None) -> str:
    import json
    from src.VLM.prompts import TRAFFIC_ANALYSIS_PROMPT
    from .tracking_service import summarize_tracking

    prompt = TRAFFIC_ANALYSIS_PROMPT
    if tracking_data:
        prompt += "\nTracking observations from this video:\n" + json.dumps(summarize_tracking(tracking_data))
        prompt += ("\nUse these track IDs only when spatial and temporal correspondence is clear. "
                   "Do not infer speed, contact, or fault from bounding-box overlap alone. "
                   "Give timestamps in seconds when supported by the sampled frames.")
    return prompt


def _analyze_via_openai_api(video_path: str, settings, tracking_data: dict | None = None) -> str:
    import base64
    import io
    import math
    from src.VLM.contact_sheet import sample_frames, make_sheet
    import openai
    from src.VLM.prompts import SYSTEM_PROMPT
    from .video_service import inspect_video

    model_id = settings.vlm_model
    if not model_id:
        raise RuntimeError("Set VLM_MODEL to a vision-capable model for VLM_API=openai.")
    if not settings.vlm_api_key:
        raise RuntimeError("Set VLM_API_KEY for the hosted visual model.")
    metadata = inspect_video(video_path)
    n_frames = min(settings.vlm_n_frames, metadata["frame_count"])
    frames = sample_frames(video_path, n_frames)
    if not frames:
        raise RuntimeError("No frames could be extracted from this video.")
    content = [{"type": "text", "text": _analysis_prompt(tracking_data) +
               "\nImages may be contact sheets: read panels left-to-right, top-to-bottom. "
               "Each panel is a different time, labelled in seconds; they are not simultaneous objects."}]
    group_size = math.ceil(len(frames) / settings.vlm_max_images)
    for offset in range(0, len(frames), group_size):
        group = frames[offset:offset + group_size]
        if len(group) == 1:
            ts, picture = group[0]
            picture.thumbnail((768, 768))
            caption = f"Frame at {ts:.3f} seconds"
        else:
            picture = make_sheet(group, cols=math.ceil(math.sqrt(len(group))), thumb_w=384, thumb_h=216)
            caption = "Video frames at " + ", ".join(f"{ts:.3f} seconds" for ts, _ in group)
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.append({"type": "text", "text": caption})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})

    with openai.OpenAI(api_key=settings.vlm_api_key, base_url=settings.vlm_base_url or None,
                       timeout=180.0, max_retries=1) as client:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
            max_tokens=settings.vlm_max_tokens,
            extra_body=settings.vlm_extra_params or None,
        )
    return response.choices[0].message.content or ""


def analyze_video(video_path: str, accident_id: str, settings, tracking_data: dict | None = None) -> str:
    """Use frames of the uploaded video, together with its actual tracking observations."""
    if settings.vlm_mock:
        return _MOCK_DESCRIPTION
    if settings.vlm_api == "openai":
        return _analyze_via_openai_api(video_path, settings, tracking_data)

    global _vlm_model, _vlm_processor
    from src.VLM.analyze_video import load_model, analyze
    if _vlm_model is None:
        _vlm_model, _vlm_processor = load_model(settings.vlm_model_id)
    return analyze(_vlm_model, _vlm_processor, video_path, _analysis_prompt(tracking_data),
                   settings.vlm_n_frames, settings.vlm_max_tokens)
