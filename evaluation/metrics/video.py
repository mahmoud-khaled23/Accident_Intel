"""Video understanding and accident description evaluation metrics."""
from __future__ import annotations

import re
from typing import Any, Sequence


def calculate_event_f1(
    pred_events: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    time_tolerance_sec: float = 2.0,
) -> dict[str, float]:
    """Calculate Precision, Recall, and F1 for detected video events.

    Each event dict must have:
    - 'event_type': str
    - 'timestamp': float (seconds)
    """
    if not pred_events and not gt_events:
        return {"event_precision": 1.0, "event_recall": 1.0, "event_f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if not pred_events:
        return {"event_precision": 0.0, "event_recall": 0.0, "event_f1": 0.0, "tp": 0, "fp": 0, "fn": len(gt_events)}
    if not gt_events:
        return {"event_precision": 0.0, "event_recall": 0.0, "event_f1": 0.0, "tp": 0, "fp": len(pred_events), "fn": 0}

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    for p_idx, p in enumerate(pred_events):
        p_type = p.get("event_type", "").strip().lower()
        p_time = float(p.get("timestamp", 0.0))
        for g_idx, g in enumerate(gt_events):
            if g_idx in matched_gt:
                continue
            g_type = g.get("event_type", "").strip().lower()
            g_time = float(g.get("timestamp", 0.0))

            if p_type == g_type and abs(p_time - g_time) <= time_tolerance_sec:
                matched_pred.add(p_idx)
                matched_gt.add(g_idx)
                break

    tp = len(matched_pred)
    fp = len(pred_events) - tp
    fn = len(gt_events) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def calculate_temporal_ordering_accuracy(
    pred_events: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
) -> float:
    """Calculate the pairwise relative temporal ordering accuracy for shared events."""
    # Find matching pairs by event_type
    pred_by_type = {e.get("event_type", "").lower(): float(e.get("timestamp", 0.0)) for e in pred_events}
    gt_by_type = {e.get("event_type", "").lower(): float(e.get("timestamp", 0.0)) for e in gt_events}

    common_events = list(set(pred_by_type.keys()) & set(gt_by_type.keys()))
    if len(common_events) < 2:
        return 1.0  # vacuously concordant

    pairs_total = 0
    pairs_concordant = 0

    for i in range(len(common_events)):
        for j in range(i + 1, len(common_events)):
            e1 = common_events[i]
            e2 = common_events[j]
            gt_order = gt_by_type[e1] < gt_by_type[e2]
            pred_order = pred_by_type[e1] < pred_by_type[e2]

            pairs_total += 1
            if gt_order == pred_order:
                pairs_concordant += 1

    return pairs_concordant / pairs_total if pairs_total > 0 else 1.0


def calculate_collision_metrics(
    pred_collision: bool,
    pred_time: float | None,
    gt_collision: bool,
    gt_time: float | None,
    time_tolerance_sec: float = 2.0,
) -> dict[str, Any]:
    """Evaluate collision detection and timing."""
    detection_match = (pred_collision == gt_collision)
    timing_error: float | None = None
    timing_accurate = False

    if pred_collision and gt_collision:
        if pred_time is not None and gt_time is not None:
            timing_error = abs(pred_time - gt_time)
            timing_accurate = timing_error <= time_tolerance_sec
        else:
            timing_accurate = True  # both detected collision, no timestamp comparison

    return {
        "collision_detection_accuracy": 1.0 if detection_match else 0.0,
        "timing_error_sec": timing_error,
        "timing_accurate": 1.0 if timing_accurate else 0.0,
    }


def evaluate_claims_grounding(
    description_text: str,
    unsupported_claims_to_check: Sequence[str],
    required_visible_events: Sequence[str],
) -> dict[str, float]:
    """Evaluate evidence-grounding: unsupported claims (hallucinations) vs omission of visible facts."""
    desc_lower = description_text.lower()

    # Check for presence of prohibited/unsupported claims (e.g. driver on phone, illegal u-turn if unverified)
    hallucinations_found = 0
    for claim in unsupported_claims_to_check:
        # Simple keyword / phrase search
        pattern = re.escape(claim.lower())
        if re.search(pattern, desc_lower):
            hallucinations_found += 1

    hallucination_rate = hallucinations_found / len(unsupported_claims_to_check) if unsupported_claims_to_check else 0.0

    # Check for presence of required visible events
    omissions = 0
    for req in required_visible_events:
        words = req.lower().split()
        # If at least 60% of keywords appear
        found_words = sum(1 for w in words if w in desc_lower)
        if found_words / len(words) < 0.6:
            omissions += 1

    omission_rate = omissions / len(required_visible_events) if required_visible_events else 0.0
    evidence_compliance = max(0.0, 1.0 - (hallucination_rate * 0.5 + omission_rate * 0.5))

    return {
        "hallucination_rate": hallucination_rate,
        "omission_rate": omission_rate,
        "evidence_compliance_score": evidence_compliance,
        "hallucinations_detected": hallucinations_found,
        "omissions_detected": omissions,
    }

