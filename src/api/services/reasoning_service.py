"""Accident reasoning service — wraps src/reasoning and validates the structured report.

Provides a single public function :func:`analyze_accident` that produces a
structured liability assessment from the VLM description, tracking data, and
retrieved laws.

In mock mode a hard-coded :data:`_MOCK_REPORT` is returned immediately.  In
real mode the function calls
:func:`reasoning.reasoning_accident.analyze_accident` and validates the
response against :class:`~schemas.accident.ReasoningReport`.
"""
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
"""Canned liability report returned when ``settings.reasoning_mock = True``."""


def analyze_accident(
    description: str,
    tracking_data: dict,
    laws: list[dict],
    settings,
    user_context: dict | None = None,
) -> dict:
    """Return a structured liability assessment dict.

    In mock mode returns :data:`_MOCK_REPORT` immediately.  In real mode:
    1. Calls :func:`reasoning.reasoning_accident.analyze_accident` with the
       description, compressed tracking summary, and retrieved laws.
    2. Validates the raw response against
       :class:`~schemas.accident.ReasoningReport` to ensure it is well-formed.
    3. Raises :class:`RuntimeError` on validation errors so the pipeline
       marks the accident as failed rather than storing malformed data.

    Args:
        description: VLM scene description string from the VLM service.
        tracking_data: Full tracking result dict from the tracking service.
        laws: List of retrieved traffic-law entry dicts.
        settings: Application settings with reasoning API credentials and
            model selection.
        user_context: Optional dict with user-supplied context fields such as
            ``location``, ``date_time``, ``description``, and ``notes``.

    Returns:
        A dict matching the :class:`~schemas.accident.ReasoningReport` shape.

    Raises:
        RuntimeError: If the reasoning module raises, or if the response fails
            schema validation.
    """
    if settings.reasoning_mock:
        return _MOCK_REPORT

    from src.reasoning.reasoning_accident import analyze_accident as run_reasoning

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
        def section_id(value):
            return str(value).strip().upper().removeprefix("VTL-").removeprefix("§").strip()
        known_sections = {section_id(law.get("section_id", law.get("id", ""))) for law in laws}
        for violation in report.violations:
            if not violation.law_section or section_id(violation.law_section) not in known_sections:
                raise ValueError(f"Reasoning cited an unretrieved law: {violation.law_section!r}.")
        known_vehicles = {str(track["id"]) for track in tracking_data.get("trajectories", [])}
        if any(str(vehicle.id) not in known_vehicles for vehicle in report.vehicles):
            raise ValueError("Reasoning attributed responsibility to an unknown track ID.")
        return report.model_dump()
    except RuntimeError:
        raise
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"Invalid reasoning model response: {exc}") from exc
