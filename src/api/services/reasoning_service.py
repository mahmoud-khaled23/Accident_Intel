"""Accident reasoning service — wraps src/reasoning and validates the structured report."""
import sys
from pathlib import Path

_project_root = str(Path(__file__).parents[3])
_src_path = str(Path(__file__).parents[3] / "src")
for _p in (_project_root, _src_path):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_MOCK_REPORT = {
    "overall_assessment": (
        "Based on the video analysis and tracking data, Vehicle 1 appears to have failed to "
        "yield at the intersection, resulting in a collision with Vehicle 2. The evidence "
        "suggests primary fault lies with Vehicle 1."
    ),
    "vehicles": [
        {
            "id": 1,
            "role": "primary_at_fault",
            "violations": ["failure_to_yield", "speeding"],
            "fault_percentage": 75,
        },
        {
            "id": 2,
            "role": "secondary",
            "violations": [],
            "fault_percentage": 25,
        },
    ],
    "violations": [
        {
            "law_section": "1110",
            "description": "Failure to obey traffic control device",
            "severity": "major",
        },
        {
            "law_section": "1111",
            "description": "Unsafe speed for conditions",
            "severity": "moderate",
        },
    ],
    "confidence": 0.72,
    "limitations": [
        "Video resolution limits precise speed estimation",
        "Partial occlusion during critical moment",
        "Mock analysis — real OpenAI reasoning not enabled",
    ],
}


def analyze_accident(
    description: str,
    tracking_data: dict,
    laws: list[dict],
    settings,
    user_context: dict | None = None,
) -> dict:
    """Return a structured liability assessment dict."""
    if settings.reasoning_mock:
        return _MOCK_REPORT

    from reasoning.reasoning_accident import analyze_accident as run_reasoning

    from ..schemas.accident import ReasoningReport
    from .tracking_service import summarize_tracking

    try:
        raw = run_reasoning(
            description,
            summarize_tracking(tracking_data),
            laws,
            user_context=user_context,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.reasoning_model,
        )
        report = ReasoningReport.model_validate(raw)
        if not report.overall_assessment.strip():
            raise ValueError("Reasoning returned an empty assessment.")
        return report.model_dump()
    except RuntimeError:
        raise
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"Invalid reasoning model response: {exc}") from exc
