"""Unit and regression tests for the evaluation framework, metrics, validators, and endpoints."""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.config import EvaluationConfig
from evaluation.manifest import RunManifest
from evaluation.validators import (
    validate_rag_dataset,
    validate_video_understanding_dataset,
    validate_tracking_dataset,
    validate_reasoning_dataset,
    validate_damage_cost_dataset,
)
from evaluation.metrics.retrieval import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    hit_rate_at_k,
    ndcg_at_k,
)
from evaluation.metrics.tracking import (
    calculate_box_iou,
    evaluate_detection_frame,
    evaluate_tracking_sequence,
)
from evaluation.metrics.damage import (
    calculate_mask_iou,
    calculate_mask_dice,
    calculate_cost_errors,
    calculate_action_accuracy,
)
from evaluation.metrics.reasoning import (
    validate_reasoning_output_schema,
    calculate_citation_metrics,
    calculate_responsibility_metrics,
    evaluate_insufficient_evidence_compliance,
)
from evaluation.metrics.video import (
    calculate_event_f1,
    calculate_temporal_ordering_accuracy,
    calculate_collision_metrics,
    evaluate_claims_grounding,
)


# =============================================================================
# 1. Dataset Validator Tests
# =============================================================================

def test_validate_rag_dataset_valid(tmp_path):
    valid_file = tmp_path / "valid_rag.jsonl"
    data = [
        {"user_input": "What is the speed limit?", "reference_context_ids": ["c1", "c2"], "reference_contexts": ["Text 1"], "query_type": "single_hop"},
        {"user_input": "Can I turn right on red in NYC?", "reference_context_ids": [], "reference_contexts": [], "query_type": "unanswerable"},
    ]
    valid_file.write_text("\n".join(json.dumps(d) for d in data), encoding="utf-8")
    is_valid, errors, count = validate_rag_dataset(str(valid_file))
    assert is_valid is True
    assert count == 2
    assert len(errors) == 0


def test_validate_rag_dataset_empty_file(tmp_path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    is_valid, errors, count = validate_rag_dataset(str(empty_file))
    assert is_valid is False
    assert count == 0


def test_validate_rag_dataset_missing_fields(tmp_path):
    corrupt_file = tmp_path / "corrupt.jsonl"
    corrupt_file.write_text(json.dumps({"some_key": "val"}), encoding="utf-8")
    is_valid, errors, count = validate_rag_dataset(str(corrupt_file))
    assert is_valid is False
    assert len(errors) > 0


def test_validate_benchmarks_on_repo_datasets():
    """Verify that all bundled gold datasets pass strict pre-flight validation."""
    is_valid, errs, cnt = validate_rag_dataset("evaluation/datasets/rag/ragas_testset.jsonl")
    assert is_valid is True and cnt == 54

    is_valid, errs, cnt = validate_video_understanding_dataset("evaluation/datasets/video/video_benchmark.json")
    assert is_valid is True and cnt == 8

    is_valid, errs, cnt = validate_tracking_dataset("evaluation/datasets/tracking/tracking_benchmark.json")
    assert is_valid is True and cnt == 30

    is_valid, errs, cnt = validate_reasoning_dataset("evaluation/datasets/reasoning/reasoning_benchmark.json")
    assert is_valid is True and cnt == 12

    is_valid, errs, cnt = validate_damage_cost_dataset("evaluation/datasets/damage/damage_benchmark.json")
    assert is_valid is True and cnt == 10


# =============================================================================
# 2. Retrieval Metric Tests
# =============================================================================

def test_retrieval_metrics():
    gold = ["c1", "c3"]
    retrieved = ["c2", "c1", "c4", "c3", "c5"]

    # Recall
    assert recall_at_k(retrieved, gold, k=1) == 0.0
    assert recall_at_k(retrieved, gold, k=2) == 0.5
    assert recall_at_k(retrieved, gold, k=4) == 1.0

    # Precision
    assert precision_at_k(retrieved, gold, k=2) == 0.5
    assert precision_at_k(retrieved, gold, k=4) == 0.5

    # Hit Rate
    assert hit_rate_at_k(retrieved, gold, k=1) == 0.0
    assert hit_rate_at_k(retrieved, gold, k=2) == 1.0

    # MRR (first relevant at rank 2 -> 1/2 = 0.5)
    assert mean_reciprocal_rank(retrieved, gold) == 0.5

    # nDCG@k
    ndcg = ndcg_at_k(retrieved, gold, k=4)
    assert 0.0 < ndcg <= 1.0


def test_retrieval_empty_cases():
    assert recall_at_k([], ["c1"], k=5) == 0.0
    assert recall_at_k(["c1"], [], k=5) == 0.0
    assert mean_reciprocal_rank([], ["c1"]) == 0.0
    assert ndcg_at_k([], ["c1"], k=5) == 0.0


# =============================================================================
# 3. Tracking Metric Tests
# =============================================================================

def test_box_iou():
    box_a = [0, 0, 10, 10]
    box_b = [0, 0, 10, 10]
    assert calculate_box_iou(box_a, box_b) == 1.0

    box_c = [5, 0, 15, 10]
    assert abs(calculate_box_iou(box_a, box_c) - (1.0 / 3.0)) < 1e-4

    box_d = [20, 20, 30, 30]
    assert calculate_box_iou(box_a, box_d) == 0.0


def test_tracking_sequence_evaluation():
    ground_truth = [
        [{"track_id": 1, "bbox": [10, 10, 50, 50]}],
        [{"track_id": 1, "bbox": [12, 10, 52, 50]}],
    ]
    predictions = [
        [{"track_id": 100, "bbox": [10, 10, 50, 50]}],
        [{"track_id": 100, "bbox": [12, 10, 52, 50]}],
    ]
    res = evaluate_tracking_sequence(predictions, ground_truth, iou_thresh=0.5)
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert res["mean_iou"] == 1.0
    assert res["id_switches"] == 0


# =============================================================================
# 4. Damage & Cost Metric Tests
# =============================================================================

def test_mask_metrics():
    mask1 = [[0, 1], [1, 0]]
    mask2 = [[0, 1], [1, 0]]
    assert calculate_mask_iou(mask1, mask2) == 1.0
    assert calculate_mask_dice(mask1, mask2) == 1.0

    mask3 = [[1, 0], [0, 1]]
    assert calculate_mask_iou(mask1, mask3) == 0.0
    assert calculate_mask_dice(mask1, mask3) == 0.0


def test_cost_errors():
    res = calculate_cost_errors(pred_min=900.0, pred_max=1200.0, gold_cost=1050.0)
    assert res["pred_central"] == 1050.0
    assert res["absolute_error"] == 0.0
    assert res["within_spread"] == 1.0


def test_action_accuracy():
    assert calculate_action_accuracy("replace", "replace") == 1.0
    assert calculate_action_accuracy("repair", "REPAIR") == 1.0
    assert calculate_action_accuracy("paint", "replace") == 0.0


# =============================================================================
# 5. Reasoning Metric Tests
# =============================================================================

def test_reasoning_metrics():
    pred_citations = ["VTL 1111", "VTL 1128"]
    gold_citations = ["VTL 1111", "VTL 1141"]
    res = calculate_citation_metrics(pred_citations, gold_citations)
    assert res["citation_precision"] == 0.5
    assert res["citation_recall"] == 0.5
    assert res["citation_f1"] == 0.5

    assert calculate_responsibility_metrics("shared_responsibility", "shared_responsibility")["exact_match"] == 1.0
    assert calculate_responsibility_metrics("potential_responsibility", "insufficient_evidence")["exact_match"] == 0.0

    # Compliance on insufficient evidence
    assert evaluate_insufficient_evidence_compliance("insufficient_evidence", True)["insufficient_evidence_accuracy"] == 1.0
    assert evaluate_insufficient_evidence_compliance("potential_responsibility", True)["insufficient_evidence_accuracy"] == 0.0


# =============================================================================
# 6. Video Metric Tests
# =============================================================================

def test_video_metrics():
    # Event F1
    pred_events = [
        {"event_type": "intersection_entry", "timestamp": 1.0},
        {"event_type": "red_light", "timestamp": 2.0},
    ]
    gold_events = [
        {"event_type": "red_light", "timestamp": 2.1},
        {"event_type": "collision", "timestamp": 3.0},
    ]
    res = calculate_event_f1(pred_events, gold_events, time_tolerance_sec=0.5)
    assert res["event_precision"] == 0.5
    assert res["event_recall"] == 0.5
    assert res["event_f1"] == 0.5

    # Temporal ordering
    pred_times = [{"event_type": "A", "timestamp": 1.0}, {"event_type": "B", "timestamp": 2.0}]
    gold_times = [{"event_type": "A", "timestamp": 1.0}, {"event_type": "B", "timestamp": 2.0}]
    assert calculate_temporal_ordering_accuracy(pred_times, gold_times) == 1.0

    # Collision metrics
    c_res = calculate_collision_metrics(
        pred_collision=True,
        pred_time=2.5,
        gt_collision=True,
        gt_time=2.4,
        time_tolerance_sec=0.5,
    )
    assert c_res["collision_detection_accuracy"] == 1.0
    assert c_res["timing_accurate"] == 1.0


# =============================================================================
# 7. Manifest and Config Tests
# =============================================================================

def test_manifest_creation():
    manifest = RunManifest(run_id="test_run_01")
    manifest.add_dataset("rag_testset", "evaluation/datasets/rag/ragas_testset.jsonl", 54)
    manifest.record_stage("rag_retrieval", "SUCCESS", sample_count=54, duration_seconds=1.23)

    data = manifest.to_dict()
    assert data["run_id"] == "test_run_01"
    assert "git" in data
    assert "environment" in data
    assert len(data["datasets"]) == 1
    assert data["datasets"][0]["sample_count"] == 54
    assert data["stages"]["rag_retrieval"]["status"] == "SUCCESS"


def test_evaluation_config():
    cfg = EvaluationConfig(stages=["rag_retrieval"], seed=123)
    assert cfg.stages == ["rag_retrieval"]
    assert cfg.seed == 123
    assert cfg.rag_mode == "hybrid"


# =============================================================================
# 8. Evaluation API Endpoints Tests
# =============================================================================

def test_evaluation_router_endpoints():
    from src.api.routers.evaluation import get_latest_evaluation, list_evaluation_runs
    latest = get_latest_evaluation()
    assert "status" in latest

    history = list_evaluation_runs()
    assert isinstance(history, list)
