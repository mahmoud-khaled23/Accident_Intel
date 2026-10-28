"""Vehicle detection and tracking evaluation metrics."""
from __future__ import annotations

from typing import Any, Sequence


def calculate_box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_width = max(0.0, x_b - x_a)
    inter_height = max(0.0, y_b - y_a)
    inter_area = inter_width * inter_height

    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def evaluate_detection_frame(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    iou_thresh: float = 0.5,
) -> dict[str, float]:
    """Evaluate detection metrics for a single frame.

    Each dict in predictions/ground_truth should have 'bbox': [x1, y1, x2, y2].
    Optional 'class_id' for class-aware matching.
    """
    if not predictions and not ground_truth:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "mean_iou": 1.0}
    if not predictions:
        return {"tp": 0, "fp": 0, "fn": len(ground_truth), "precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_iou": 0.0}
    if not ground_truth:
        return {"tp": 0, "fp": len(predictions), "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_iou": 0.0}

    # Calculate all pairwise IoUs
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[float, int, int]] = []  # (iou, pred_idx, gt_idx)

    for p_idx, p in enumerate(predictions):
        p_box = p["bbox"]
        p_cls = p.get("class_id")
        for g_idx, g in enumerate(ground_truth):
            g_box = g["bbox"]
            g_cls = g.get("class_id")
            # If classes are specified, they must match
            if p_cls is not None and g_cls is not None and p_cls != g_cls:
                continue
            iou = calculate_box_iou(p_box, g_box)
            if iou >= iou_thresh:
                matches.append((iou, p_idx, g_idx))

    # Sort matches by highest IoU first (greedy matching)
    matches.sort(key=lambda x: x[0], reverse=True)
    matched_ious: list[float] = []

    for iou, p_idx, g_idx in matches:
        if p_idx not in matched_pred and g_idx not in matched_gt:
            matched_pred.add(p_idx)
            matched_gt.add(g_idx)
            matched_ious.append(iou)

    tp = len(matched_pred)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
    }


def evaluate_tracking_sequence(
    frames_pred: list[list[dict[str, Any]]],
    frames_gt: list[list[dict[str, Any]]],
    iou_thresh: float = 0.5,
) -> dict[str, Any]:
    """Evaluate track consistency across a sequence of frames.

    Counts TP, FP, FN, ID switches, and track fragmentations.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    ious: list[float] = []

    # Map: gt_track_id -> last_assigned_pred_track_id
    gt_to_pred_map: dict[Any, Any] = {}
    gt_last_seen_frame: dict[Any, int] = {}

    id_switches = 0
    fragmentations = 0

    num_frames = min(len(frames_pred), len(frames_gt))

    for frame_idx in range(num_frames):
        preds = frames_pred[frame_idx]
        gts = frames_gt[frame_idx]

        # Greedy match on this frame
        matches: list[tuple[float, int, int]] = []
        for p_idx, p in enumerate(preds):
            for g_idx, g in enumerate(gts):
                iou = calculate_box_iou(p["bbox"], g["bbox"])
                if iou >= iou_thresh:
                    matches.append((iou, p_idx, g_idx))

        matches.sort(key=lambda x: x[0], reverse=True)
        matched_pred = set()
        matched_gt = set()

        for iou, p_idx, g_idx in matches:
            if p_idx not in matched_pred and g_idx not in matched_gt:
                matched_pred.add(p_idx)
                matched_gt.add(g_idx)
                ious.append(iou)

                p_track = preds[p_idx].get("track_id")
                g_track = gts[g_idx].get("track_id")

                if g_track is not None and p_track is not None:
                    # Check ID switch
                    if g_track in gt_to_pred_map:
                        if gt_to_pred_map[g_track] != p_track:
                            id_switches += 1
                            gt_to_pred_map[g_track] = p_track
                    else:
                        gt_to_pred_map[g_track] = p_track

                    # Check track fragmentation (gap in tracking)
                    if g_track in gt_last_seen_frame:
                        if frame_idx - gt_last_seen_frame[g_track] > 1:
                            fragmentations += 1
                    gt_last_seen_frame[g_track] = frame_idx

        tp = len(matched_pred)
        fp = len(preds) - tp
        fn = len(gts) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "id_switches": id_switches,
        "fragmentations": fragmentations,
        "frames_evaluated": num_frames,
    }

