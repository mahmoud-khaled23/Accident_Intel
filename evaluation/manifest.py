"""Manifest and provenance metadata generator for evaluation runs."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_file_hash(path: Path | str) -> str:
    """Compute the SHA-256 hash of a file."""
    path = Path(path)
    if not path.is_file():
        return "file_not_found"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_info() -> dict[str, Any]:
    """Retrieve current Git commit, branch, and status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        branch = "unknown"

    try:
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        is_dirty = bool(status_output)
    except Exception:
        is_dirty = False

    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": is_dirty,
    }


def get_environment_info() -> dict[str, Any]:
    """Capture environment and dependency runtime metadata."""
    cuda_available = False
    cuda_device = None
    torch_version = None

    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
    }


@dataclass
class DatasetManifestItem:
    name: str
    file_path: str
    sample_count: int
    sha256: str


@dataclass
class StageStatusItem:
    stage: str
    status: str  # "SUCCESS", "FAILED", "SKIPPED", "BLOCKED"
    reason: str | None = None
    sample_count: int = 0
    duration_seconds: float = 0.0


@dataclass
class RunManifest:
    run_id: str
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git: dict[str, Any] = field(default_factory=get_git_info)
    environment: dict[str, Any] = field(default_factory=get_environment_info)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)

    def add_dataset(self, name: str, file_path: str, sample_count: int) -> None:
        p = Path(file_path)
        sha = compute_file_hash(p) if p.exists() else "missing"
        self.datasets.append({
            "name": name,
            "file_path": str(file_path),
            "sample_count": sample_count,
            "sha256": sha,
        })

    def record_stage(
        self,
        stage: str,
        status: str,
        sample_count: int = 0,
        duration_seconds: float = 0.0,
        reason: str | None = None,
    ) -> None:
        self.stages[stage] = {
            "status": status,
            "sample_count": sample_count,
            "duration_seconds": round(duration_seconds, 3),
            "reason": reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "git": self.git,
            "environment": self.environment,
            "datasets": self.datasets,
            "stages": self.stages,
            "models": self.models,
        }

