"""Accident reasoning service — uses OpenAI to produce a structured liability assessment."""
import json
import sys
from pathlib import Path

# Ensure project root and src/ are on sys.path
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

_SYSTEM_PROMPT = (
    "You are an expert traffic accident analyst and legal advisor. "
    "Analyze the provided accident description, vehicle tracking data, and relevant "
    "traffic laws to produce a structured liability assessment."
)


def analyze_accident(
    description: str,
    tracking_data: dict,
    laws: list[dict],
    settings,
) -> dict:
    """Return a structured liability assessment dict.

    Uses mock data when settings.reasoning_mock is True, otherwise calls OpenAI.
    """
    if settings.reasoning_mock:
        return _MOCK_REPORT

    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            f"openai package not installed. Original error: {exc}"
        ) from exc

    # Build the user message
    tracking_summary = json.dumps(tracking_data, indent=2)
    laws_text = "\n\n".join(
        f"Section {law.get('section_id', '')}: {law.get('title', '')}\n{law.get('content', '')}"
        for law in laws
    )
    user_message = (
        f"## Accident Description\n{description}\n\n"
        f"## Vehicle Tracking Summary\n{tracking_summary}\n\n"
        f"## Relevant Traffic Laws\n{laws_text}\n\n"
        "Provide a structured JSON analysis with: overall_assessment, vehicles "
        "(list with id, role, violations, fault_percentage), violations "
        "(list with law_section, description, severity), confidence (0.0-1.0), "
        "limitations (list of strings)"
    )

    import logging
    import os

    logger = logging.getLogger(__name__)

    api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    if not base_url and api_key.startswith("sk-or-v1-"):
        base_url = "https://openrouter.ai/api/v1"

    model = settings.reasoning_model
    if base_url and "openrouter.ai" in base_url and not ("/" in model):
        model = f"openai/{model}"

    try:
        client = openai.OpenAI(api_key=api_key or "no-key", base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )
        response_text = response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("OpenAI reasoning failed (%s) — falling back to mock report", exc)
        return _MOCK_REPORT

    # Try to extract JSON from the response
    try:
        # Strip markdown code fences if present
        text = response_text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.rsplit("```", 1)[0].strip()
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {
            "overall_assessment": response_text,
            "vehicles": [],
            "violations": [],
            "confidence": 0.5,
            "limitations": ["Raw text response — JSON parsing failed"],
        }
