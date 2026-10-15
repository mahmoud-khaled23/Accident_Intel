"""VLM video analysis service — wraps src/VLM (local Qwen2-VL or hosted vision API)."""
import json
import logging

logger = logging.getLogger(__name__)
_vlm_model = None
_vlm_processor = None
_vlm_model_id = None


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


_HOSTED_ANALYSIS_PROMPT = """You are a cautious traffic-video evidence analyst. Inspect the supplied video frames in time order. Do not assume an accident occurred and do not infer fault, intent, exact speed, or traffic-law violations.

Write a concise report with exactly these headings:
Scene Overview:
Chronological Events:
Incident Evidence:
Final Observed State:
Uncertainties and Limitations:

Report only what is visually supported. For contact, a fall, a traffic signal, or any critical detail that is blurry, hidden, or not captured, say "unclear" rather than guessing. The tracking notes below are only approximate image-space observations; they do not prove contact or speed."""


def _compact_tracking_context(tracking_data: dict) -> dict:
    """Bound tracking text so it cannot consume a hosted model's vision budget."""
    vehicles = []
    for track in tracking_data.get("trajectories", [])[:12]:
        frames = track.get("frames", [])
        vehicles.append({
            "id": track.get("id"),
            "class": track.get("class", "vehicle"),
            "first_seen_s": frames[0].get("timestamp") if frames else None,
            "last_seen_s": frames[-1].get("timestamp") if frames else None,
        })
    candidates = []
    for candidate in (tracking_data.get("accident_candidates") or [])[:5]:
        candidates.append({
            "vehicle_ids": candidate.get("vehicle_ids"),
            "overlap_frame": candidate.get("overlap_frame"),
            "evidence": candidate.get("evidence", []),
        })
    return {
        "vehicle_count": tracking_data.get("vehicle_count", len(vehicles)),
        "duration_seconds": tracking_data.get("duration_seconds"),
        "vehicles": vehicles,
        "candidate_notes": candidates,
    }


def _analysis_prompt(tracking_data: dict | None, *, compact: bool = False) -> str:
    from src.VLM.prompts import TRAFFIC_ANALYSIS_PROMPT
    from .tracking_service import summarize_tracking

    prompt = _HOSTED_ANALYSIS_PROMPT if compact else TRAFFIC_ANALYSIS_PROMPT
    if tracking_data:
        context = _compact_tracking_context(tracking_data) if compact else summarize_tracking(tracking_data)
        prompt += "\nTracking observations from this video:\n" + json.dumps(context, separators=(",", ":"))
        prompt += ("\nUse these track IDs only when spatial and temporal correspondence is clear. "
                   "Do not infer speed, contact, or fault from bounding-box overlap alone. "
                   "Give timestamps in seconds when supported by the sampled frames.")
    return prompt


def _analyze_via_openai_api(video_path: str, settings, tracking_data: dict | None = None) -> str:
    from src.VLM.hosted import analyze_hosted
    from .video_service import inspect_video

    metadata = inspect_video(video_path)
    n_frames = min(settings.vlm_n_frames, metadata["frame_count"])
    return analyze_hosted(
        video_path,
        _analysis_prompt(tracking_data, compact=True),
        model=settings.vlm_model,
        api_key=settings.vlm_api_key,
        base_url=settings.vlm_base_url or None,
        n_frames=n_frames,
        max_images=settings.vlm_max_images,
        max_tokens=settings.vlm_max_tokens,
        extra_params=settings.vlm_extra_params or None,
    )


def backend_info(settings) -> dict:
    """Expose the selected video model without exposing provider credentials."""
    if settings.vlm_mock:
        return {"backend": "mock", "model": None, "module": None}
    if settings.video_vlm_api == "openai":
        return {"backend": "openai", "model": settings.vlm_model, "module": "src.VLM.hosted"}
    return {"backend": "local", "model": settings.vlm_model_id, "module": "src.VLM.analyze_video"}


def analyze_video(video_path: str, accident_id: str, settings, tracking_data: dict | None = None) -> str:
    """Use frames of the uploaded video, together with its actual tracking observations."""
    if settings.vlm_mock:
        return _MOCK_DESCRIPTION
    info = backend_info(settings)
    logger.info("Analysis %s: running %s with model %s", accident_id, info["module"], info["model"])
    if settings.video_vlm_api == "openai":
        return _analyze_via_openai_api(video_path, settings, tracking_data)

    global _vlm_model, _vlm_processor, _vlm_model_id
    from src.VLM.analyze_video import analyze, load_model
    if _vlm_model is None or _vlm_model_id != settings.vlm_model_id:
        # Drop the previous weights before loading a different model on the GPU.
        _vlm_model = _vlm_processor = _vlm_model_id = None
        _vlm_model, _vlm_processor = load_model(settings.vlm_model_id)
        _vlm_model_id = settings.vlm_model_id
    return analyze(
        _vlm_model,
        _vlm_processor,
        video_path,
        _analysis_prompt(tracking_data),
        settings.vlm_n_frames,
        settings.vlm_max_tokens,
    )
