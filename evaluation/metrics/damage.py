"""Damage segmentation, mapping, and repair cost estimation metrics."""
from __future__ import annotations

import math
from typing import Any, Sequence


def calculate_mask_iou(mask_a: Any, mask_b: Any) -> float:
    """Calculate IoU between two boolean or binary 2D numpy arrays."""
    import numpy as np

    arr_a = np.asarray(mask_a, dtype=bool)
    arr_b = np.asarray(mask_b, dtype=bool)

    intersection = np.logical_and(arr_a, arr_b).sum()
    union = np.logical_or(arr_a, arr_b).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def calculate_mask_dice(mask_a: Any, mask_b: Any) -> float:
    """Calculate Dice similarity coefficient between two boolean or binary 2D numpy arrays."""
    import numpy as np

    arr_a = np.asarray(mask_a, dtype=bool)
    arr_b = np.asarray(mask_b, dtype=bool)

    intersection = np.logical_and(arr_a, arr_b).sum()
    total_area = arr_a.sum() + arr_b.sum()

    if total_area == 0:
        return 1.0 if intersection == 0 else 0.0
    return float((2.0 * intersection) / total_area)


def calculate_cost_errors(pred_min: float, pred_max: float, gold_cost: float) -> dict[str, float]:
    """Calculate error metrics for repair cost estimation (EGP)."""
    central_estimate = (pred_min + pred_max) / 2.0
    abs_err = abs(central_estimate - gold_cost)
    pct_err = (abs_err / gold_cost * 100.0) if gold_cost > 0 else 0.0
    within_spread = 1.0 if (pred_min <= gold_cost <= pred_max) else 0.0

    return {
        "pred_min": pred_min,
        "pred_max": pred_max,
        "pred_central": central_estimate,
        "gold_cost": gold_cost,
        "absolute_error": abs_err,
        "percentage_error": pct_err,
        "within_spread": within_spread,
    }


def calculate_action_accuracy(pred_action: str, gold_action: str) -> float:
    """Evaluate repair action classification (repair_only, paint_only, repair_and_paint, replace)."""
    return 1.0 if pred_action.strip().lower() == gold_action.strip().lower() else 0.0

