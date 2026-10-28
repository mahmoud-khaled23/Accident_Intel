"""Dataset and annotation validation routines."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DatasetValidationError(ValueError):
    """Raised when a dataset or annotation file fails schema validation."""
    pass


def validate_rag_dataset(file_path: Path | str) -> tuple[bool, list[str], int]:
    """Validate a RAG evaluation dataset file (JSONL format).

    Returns:
        (is_valid, error_messages, sample_count)
    """
    path = Path(file_path)
    if not path.is_file():
        return False, [f"File not found: {path}"], 0

    errors: list[str] = []
    count = 0

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {idx}: Invalid JSON ({e})")
                continue

            count += 1
            # Required fields
            if "user_input" not in row or not isinstance(row["user_input"], str):
                errors.append(f"Line {idx}: Missing or invalid 'user_input'")
            if "reference_context_ids" not in row or not isinstance(row["reference_context_ids"], list):
                errors.append(f"Line {idx}: Missing or invalid 'reference_context_ids' list")
            elif len(row["reference_context_ids"]) == 0 and row.get("query_type") != "unanswerable":
                errors.append(f"Line {idx}: 'reference_context_ids' is empty for answerable query")
            if "reference_contexts" not in row or not isinstance(row["reference_contexts"], list):
                errors.append(f"Line {idx}: Missing or invalid 'reference_contexts' list")

    if count == 0:
        errors.append("Dataset contains 0 valid samples")

    return len(errors) == 0, errors, count


def validate_video_understanding_dataset(file_path: Path | str) -> tuple[bool, list[str], int]:
    """Validate a video understanding evaluation dataset (JSON format).

    Returns:
        (is_valid, error_messages, sample_count)
    """
    path = Path(file_path)
    if not path.is_file():
        return False, [f"File not found: {path}"], 0

    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"Could not parse JSON in {path}: {e}"], 0

    if not isinstance(data, list):
        return False, ["Root element must be a list of video samples"], 0

    count = len(data)
    if count == 0:
        return False, ["Dataset is empty"], 0

    for idx, sample in enumerate(data, 1):
        if not isinstance(sample, dict):
            errors.append(f"Sample #{idx}: Must be a dictionary")
            continue
        if "video_id" not in sample:
            errors.append(f"Sample #{idx}: Missing 'video_id'")
        gt = sample.get("ground_truth")
        if not gt or not isinstance(gt, dict):
            errors.append(f"Sample #{idx}: Missing or invalid 'ground_truth' dict")
            continue
        if "vehicles" not in gt or not isinstance(gt["vehicles"], list):
            errors.append(f"Sample #{idx}: 'ground_truth.vehicles' must be a list")
        if "events" not in gt or not isinstance(gt["events"], list):
            errors.append(f"Sample #{idx}: 'ground_truth.events' must be a list")
        if "collision" not in gt:
            errors.append(f"Sample #{idx}: Missing 'ground_truth.collision'")

    return len(errors) == 0, errors, count


def validate_tracking_dataset(file_path: Path | str) -> tuple[bool, list[str], int]:
    """Validate a vehicle tracking evaluation dataset (JSON format).

    Returns:
        (is_valid, error_messages, frame_count)
    """
    path = Path(file_path)
    if not path.is_file():
        return False, [f"File not found: {path}"], 0

    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"Could not parse tracking JSON: {e}"], 0

    if not isinstance(data, dict):
        return False, ["Root element must be a dictionary with metadata and frames"], 0

    frames = data.get("frames", [])
    if not isinstance(frames, list) or len(frames) == 0:
        return False, ["'frames' must be a non-empty list"], 0

    for idx, f in enumerate(frames):
        if "frame_id" not in f:
            errors.append(f"Frame #{idx}: Missing 'frame_id'")
        boxes = f.get("boxes", [])
        if not isinstance(boxes, list):
            errors.append(f"Frame #{idx}: 'boxes' must be a list")
            continue
        for b_idx, box in enumerate(boxes):
            if "bbox" not in box or len(box["bbox"]) != 4:
                errors.append(f"Frame #{idx} box #{b_idx}: 'bbox' must be [x1, y1, x2, y2]")
            else:
                x1, y1, x2, y2 = box["bbox"]
                if x2 <= x1 or y2 <= y1:
                    errors.append(f"Frame #{idx} box #{b_idx}: Invalid coordinates: x2 <= x1 or y2 <= y1")
            if "track_id" not in box:
                errors.append(f"Frame #{idx} box #{b_idx}: Missing 'track_id'")

    return len(errors) == 0, errors, len(frames)


def validate_reasoning_dataset(file_path: Path | str) -> tuple[bool, list[str], int]:
    """Validate an accident reasoning evaluation dataset (JSON format).

    Returns:
        (is_valid, error_messages, sample_count)
    """
    path = Path(file_path)
    if not path.is_file():
        return False, [f"File not found: {path}"], 0

    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"Could not parse reasoning JSON: {e}"], 0

    if not isinstance(data, list) or len(data) == 0:
        return False, ["Root must be a non-empty list of reasoning cases"], 0

    valid_levels = {"potential_responsibility", "shared_responsibility", "insufficient_evidence"}
    for idx, sample in enumerate(data, 1):
        if "case_id" not in sample:
            errors.append(f"Sample #{idx}: Missing 'case_id'")
        if "description" not in sample:
            errors.append(f"Sample #{idx}: Missing 'description'")
        if "gold_level" not in sample or sample["gold_level"] not in valid_levels:
            errors.append(f"Sample #{idx}: 'gold_level' must be one of {valid_levels}")
        if "gold_sections" not in sample or not isinstance(sample["gold_sections"], list):
            errors.append(f"Sample #{idx}: 'gold_sections' must be a list")

    return len(errors) == 0, errors, len(data)


def validate_damage_cost_dataset(file_path: Path | str) -> tuple[bool, list[str], int]:
    """Validate a damage and cost estimation dataset (JSON format).

    Returns:
        (is_valid, error_messages, sample_count)
    """
    path = Path(file_path)
    if not path.is_file():
        return False, [f"File not found: {path}"], 0

    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"Could not parse damage JSON: {e}"], 0

    if not isinstance(data, list) or len(data) == 0:
        return False, ["Root must be a non-empty list of damage cases"], 0

    for idx, sample in enumerate(data, 1):
        if "case_id" not in sample:
            errors.append(f"Sample #{idx}: Missing 'case_id'")
        if "vehicle_info" not in sample or not isinstance(sample["vehicle_info"], dict):
            errors.append(f"Sample #{idx}: Missing 'vehicle_info' dict")
        if "gold_damages" not in sample or not isinstance(sample["gold_damages"], list):
            errors.append(f"Sample #{idx}: Missing 'gold_damages' list")
        if "gold_cost_range" not in sample or len(sample["gold_cost_range"]) != 2:
            errors.append(f"Sample #{idx}: 'gold_cost_range' must be [min_cost, max_cost]")

    return len(errors) == 0, errors, len(data)
