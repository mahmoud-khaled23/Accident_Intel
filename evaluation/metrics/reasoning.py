"""Accident reasoning and legal assessment metrics."""
from __future__ import annotations

import re
from typing import Any, Sequence


def _clean_section(s: str) -> str:
    """Normalize a law section ID (e.g., 'VTL-1111#d.1' -> '1111', 'Section 1141' -> '1141')."""
    s = s.strip()
    m = re.search(r"(\d{3,4}(?:-[a-z])?)", s, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return s.lower()


def validate_reasoning_output_schema(data: Any) -> tuple[bool, list[str]]:
    """Validate that the model response conforms to the reasoning JSON contract."""
    if not isinstance(data, dict):
        return False, ["Output is not a valid JSON object"]

    errors: list[str] = []
    required_keys = ["overall_assessment", "overall_level", "vehicles", "violations", "confidence", "limitations"]
    for k in required_keys:
        if k not in data:
            errors.append(f"Missing required key: '{k}'")

    valid_levels = {"potential_responsibility", "shared_responsibility", "insufficient_evidence"}
    level = data.get("overall_level")
    if level not in valid_levels:
        errors.append(f"Invalid overall_level '{level}'. Must be one of {valid_levels}")

    conf = data.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        errors.append(f"Confidence '{conf}' is not a float between 0.0 and 1.0")

    vehicles = data.get("vehicles")
    if not isinstance(vehicles, list):
        errors.append("'vehicles' must be a list")

    violations = data.get("violations")
    if not isinstance(violations, list):
        errors.append("'violations' must be a list")

    limitations = data.get("limitations")
    if not isinstance(limitations, list):
        errors.append("'limitations' must be a list")

    return len(errors) == 0, errors


def calculate_citation_metrics(cited_sections: Sequence[str], gold_sections: Sequence[str]) -> dict[str, float]:
    """Calculate Precision, Recall, and F1 for cited traffic law sections."""
    if not gold_sections and not cited_sections:
        return {"citation_precision": 1.0, "citation_recall": 1.0, "citation_f1": 1.0}
    if not gold_sections:
        return {"citation_precision": 0.0, "citation_recall": 1.0, "citation_f1": 0.0}
    if not cited_sections:
        return {"citation_precision": 1.0, "citation_recall": 0.0, "citation_f1": 0.0}

    cited_set = {_clean_section(s) for s in cited_sections if s}
    gold_set = {_clean_section(s) for s in gold_sections if s}

    hits = len(cited_set & gold_set)
    precision = hits / len(cited_set) if cited_set else 0.0
    recall = hits / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1,
        "hits": hits,
        "total_cited": len(cited_set),
        "total_gold": len(gold_set),
    }


def calculate_responsibility_metrics(pred_level: str, gold_level: str) -> dict[str, Any]:
    """Calculate responsibility classification agreement."""
    match = 1.0 if pred_level == gold_level else 0.0
    return {
        "exact_match": match,
        "pred_level": pred_level,
        "gold_level": gold_level,
    }


def evaluate_insufficient_evidence_compliance(pred_level: str, is_insufficient_evidence: bool) -> dict[str, float]:
    """Evaluate whether the model correctly refrains from assigning fault when evidence is ambiguous.

    If is_insufficient_evidence=True, pred_level must be 'insufficient_evidence'.
    """
    if is_insufficient_evidence:
        success = 1.0 if pred_level == "insufficient_evidence" else 0.0
    else:
        success = 1.0 if pred_level != "insufficient_evidence" else 0.0
    return {
        "insufficient_evidence_accuracy": success,
    }

