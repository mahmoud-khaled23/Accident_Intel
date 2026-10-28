"""Standardized, reproducible metric calculations for Accident Intel."""
from .damage import (
    calculate_action_accuracy,
    calculate_cost_errors,
    calculate_mask_dice,
    calculate_mask_iou,
)
from .reasoning import (
    calculate_citation_metrics,
    calculate_responsibility_metrics,
    validate_reasoning_output_schema,
)
from .retrieval import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .tracking import (
    calculate_box_iou,
    evaluate_detection_frame,
    evaluate_tracking_sequence,
)
from .video import (
    calculate_collision_metrics,
    calculate_event_f1,
    calculate_temporal_ordering_accuracy,
    evaluate_claims_grounding,
)

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "hit_rate_at_k",
    "ndcg_at_k",
    "calculate_box_iou",
    "evaluate_detection_frame",
    "evaluate_tracking_sequence",
    "validate_reasoning_output_schema",
    "calculate_citation_metrics",
    "calculate_responsibility_metrics",
    "calculate_mask_iou",
    "calculate_mask_dice",
    "calculate_cost_errors",
    "calculate_action_accuracy",
    "calculate_event_f1",
    "calculate_temporal_ordering_accuracy",
    "calculate_collision_metrics",
    "evaluate_claims_grounding",
]

