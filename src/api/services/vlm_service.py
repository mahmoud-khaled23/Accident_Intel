"""VLM video analysis service — wraps src/VLM/analyze_video.py."""
import sys
from pathlib import Path

# Ensure project root and src/ are on sys.path
_project_root = str(Path(__file__).parents[3])
_src_path = str(Path(__file__).parents[3] / "src")
for _p in (_project_root, _src_path):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Module-level model cache (populated lazily on first real call)
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


def analyze_video(video_path: str, accident_id: str, settings) -> str:
    """Analyze a video and return a textual accident description.

    Uses a mock response when settings.vlm_mock is True, otherwise loads the
    Qwen2-VL model (lazily cached) and calls analyze().
    """
    if settings.vlm_mock:
        return _MOCK_DESCRIPTION

    global _vlm_model, _vlm_processor

    try:
        from VLM.analyze_video import load_model, analyze
        from VLM.prompts import TRAFFIC_ANALYSIS_PROMPT
    except ImportError as exc:
        raise RuntimeError(
            f"VLM package not importable — are transformers/qwen-vl-utils installed? Original error: {exc}"
        ) from exc

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
