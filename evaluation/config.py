"""Configuration schemas and defaults for the evaluation pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

StageName = Literal[
    "video_understanding",
    "vehicle_tracking",
    "rag_retrieval",
    "rag_generation",
    "accident_reasoning",
    "damage_cost",
    "end_to_end",
    "comparisons",
]

ALL_STAGES: list[StageName] = [
    "video_understanding",
    "vehicle_tracking",
    "rag_retrieval",
    "rag_generation",
    "accident_reasoning",
    "damage_cost",
    "end_to_end",
    "comparisons",
]


@dataclass
class EvaluationConfig:
    """Configures an evaluation run."""

    # Stages to execute
    stages: list[StageName] = field(default_factory=lambda: list(ALL_STAGES))

    # Run metadata & paths
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S"))
    output_dir: str = "evaluation/results"
    latest_dir_name: str = "latest"

    # Reproducibility
    seed: int = 42
    sample_limit: int | None = None  # None = evaluate all available samples

    # Retrieval evaluation parameters
    retrieval_k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    rag_mode: str = "hybrid"  # "bm25", "dense", "hybrid"
    rag_rerank: bool = False  # Set to True to enable Cross-Encoder reranking

    # Tracking evaluation parameters
    tracking_iou_threshold: float = 0.5
    tracking_conf_threshold: float = 0.25

    # Reasoning evaluation parameters
    reasoning_min_confidence: float = 0.0

    # Damage cost evaluation parameters
    damage_currency: str = "EGP"
    price_table_version: str = "2026.1"

    # Hardware & provider execution mode
    device: str = "auto"  # "auto", "cpu", "cuda"
    allow_paid_api_calls: bool = False  # Strict guard against unintended paid API requests

    # Custom model overrides
    model_overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        """Return the directory path for this specific run."""
        return Path(self.output_dir) / "runs" / self.run_id

    @property
    def latest_dir(self) -> Path:
        """Return the directory path for the latest results alias."""
        return Path(self.output_dir) / self.latest_dir_name

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        return {
            "stages": self.stages,
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "latest_dir_name": self.latest_dir_name,
            "seed": self.seed,
            "sample_limit": self.sample_limit,
            "retrieval_k_values": self.retrieval_k_values,
            "rag_mode": self.rag_mode,
            "rag_rerank": self.rag_rerank,
            "tracking_iou_threshold": self.tracking_iou_threshold,
            "tracking_conf_threshold": self.tracking_conf_threshold,
            "reasoning_min_confidence": self.reasoning_min_confidence,
            "damage_currency": self.damage_currency,
            "price_table_version": self.price_table_version,
            "device": self.device,
            "allow_paid_api_calls": self.allow_paid_api_calls,
            "model_overrides": self.model_overrides,
        }

