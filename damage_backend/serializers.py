"""JSON-safe conversion helpers for numpy, pathlib, enums, and Pydantic models."""
from __future__ import annotations

import enum
from pathlib import Path

import numpy as np


def make_json_safe(obj):
    """Recursively convert non-JSON-serialisable types to Python primitives."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [make_json_safe(i) for i in obj.tolist()]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    # Pydantic v2 BaseModel
    try:
        from pydantic import BaseModel
        if isinstance(obj, BaseModel):
            return make_json_safe(obj.model_dump())
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(i) for i in obj]
    return obj
