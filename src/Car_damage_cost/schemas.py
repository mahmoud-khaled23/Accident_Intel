"""Data models.

- `Detection` is internal (holds a numpy mask) -> dataclass.
- Everything that goes into the report is Pydantic (JSON-serialisable).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field


class Severity(str, Enum):
    minor = "minor"
    moderate = "moderate"
    severe = "severe"


SEVERITY_RANK = {Severity.minor: 0, Severity.moderate: 1, Severity.severe: 2}

UNKNOWN_PART = "unknown_part"


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (original image px)
    mask: np.ndarray                          # bool (H, W), original image size
    area_px: int


# ---------------- report models ----------------

class DamageOnPart(BaseModel):
    damage_type: str
    part: str                                  # part name or "unknown_part"
    confidence: float
    coverage: float = Field(description="fraction of damage mask inside the part mask")
    area_ratio: float = Field(description="damage area / part area (or / image area if part unknown)")
    severity: Severity
    severity_source: str = "heuristic"         # heuristic | vlm
    part_source: str = "overlap"               # overlap | vlm | unknown
    bbox: list[float]


class PhotoResult(BaseModel):
    photo_id: str
    usable: bool = True
    notes: list[str] = []
    parts_found: list[str] = []
    items: list[DamageOnPart] = []


class FusedDamage(BaseModel):
    part: str
    damage_type: str
    severity: Severity
    count: int = Field(description="max number of instances seen in any single photo")
    max_confidence: float
    photos: list[str]
    needs_review: bool = False
    review_reasons: list[str] = []


class CaseReport(BaseModel):
    case_id: str
    photos: list[PhotoResult]
    fused: list[FusedDamage]
    warnings: list[str] = []


# ---------------- cost models ----------------

class Tier(str, Enum):
    economy = "economy"
    mid = "mid"
    luxury = "luxury"


class CarInfo(BaseModel):
    make: str
    model: str
    year: int
    tier: Tier


class CostLine(BaseModel):
    part: str
    action: str                      # repair_only | paint_only | repair_and_paint | replace
    damages: list[str]
    parts_min: float = 0
    parts_max: float = 0
    labor_min: float = 0
    labor_max: float = 0
    paint_min: float = 0
    paint_max: float = 0
    total_min: float = 0
    total_max: float = 0
    price_source: str = "-"          # placeholder table | gemini | -
    needs_review: bool = False
    notes: list[str] = []


class CostReport(BaseModel):
    car: CarInfo
    currency: str = "EGP"
    lines: list[CostLine]
    total_min: float
    total_max: float
    unconfirmed_min: float = 0       # part of the total that comes from needs_review lines
    unconfirmed_max: float = 0
    unpriced: list[str] = []         # damages we could not price (unknown part)
    notes: list[str] = []