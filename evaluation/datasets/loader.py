"""Dataset loaders with schema validation and sample limiting."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..validators import (
    DatasetValidationError,
    validate_damage_cost_dataset,
    validate_rag_dataset,
    validate_reasoning_dataset,
    validate_tracking_dataset,
    validate_video_understanding_dataset,
)

BASE_DATA_DIR = Path(__file__).resolve().parent


def load_rag_testset(limit: int | None = None) -> list[dict[str, Any]]:
    """Load the 54-question traffic-law RAG evaluation dataset."""
    path = BASE_DATA_DIR / "rag" / "ragas_testset.jsonl"
    if not path.is_file():
        # Fallback to evaluation/rag_evaluation/ragas_testset.jsonl
        path = BASE_DATA_DIR.parent / "rag_evaluation" / "ragas_testset.jsonl"

    is_valid, errors, count = validate_rag_dataset(path)
    if not is_valid:
        raise DatasetValidationError(f"RAG testset invalid:\n" + "\n".join(errors[:5]))

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
            if limit and len(items) >= limit:
                break
    return items


def load_video_benchmark(limit: int | None = None) -> list[dict[str, Any]]:
    """Load the video understanding evaluation benchmark dataset."""
    path = BASE_DATA_DIR / "video" / "video_benchmark.json"
    is_valid, errors, _ = validate_video_understanding_dataset(path)
    if not is_valid:
        raise DatasetValidationError(f"Video benchmark invalid:\n" + "\n".join(errors[:5]))

    data = json.loads(path.read_text(encoding="utf-8"))
    return data[:limit] if limit else data


def load_tracking_benchmark() -> dict[str, Any]:
    """Load the vehicle tracking evaluation benchmark dataset."""
    path = BASE_DATA_DIR / "tracking" / "tracking_benchmark.json"
    is_valid, errors, _ = validate_tracking_dataset(path)
    if not is_valid:
        raise DatasetValidationError(f"Tracking benchmark invalid:\n" + "\n".join(errors[:5]))

    return json.loads(path.read_text(encoding="utf-8"))


def load_reasoning_benchmark(limit: int | None = None) -> list[dict[str, Any]]:
    """Load the accident reasoning evaluation benchmark dataset."""
    path = BASE_DATA_DIR / "reasoning" / "reasoning_benchmark.json"
    is_valid, errors, _ = validate_reasoning_dataset(path)
    if not is_valid:
        raise DatasetValidationError(f"Reasoning benchmark invalid:\n" + "\n".join(errors[:5]))

    data = json.loads(path.read_text(encoding="utf-8"))
    return data[:limit] if limit else data


def load_damage_benchmark(limit: int | None = None) -> list[dict[str, Any]]:
    """Load the vehicle damage and repair cost benchmark dataset."""
    path = BASE_DATA_DIR / "damage" / "damage_benchmark.json"
    is_valid, errors, _ = validate_damage_cost_dataset(path)
    if not is_valid:
        raise DatasetValidationError(f"Damage benchmark invalid:\n" + "\n".join(errors[:5]))

    data = json.loads(path.read_text(encoding="utf-8"))
    return data[:limit] if limit else data

