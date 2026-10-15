"""Data models shared across the entire Car Damage Cost pipeline.

Defines two categories of types:

* **Internal** — :class:`Detection` is a plain dataclass that holds a NumPy
  boolean mask; it is never serialised to JSON.
* **Report models** — all other classes are Pydantic ``BaseModel`` subclasses
  and are fully JSON-serialisable. They are used for inter-step communication
  and for the final report that is written to disk.

Key exports:
    Severity, SEVERITY_RANK, UNKNOWN_PART, Detection,
    DamageOnPart, PhotoResult, FusedDamage, CaseReport,
    Tier, CarInfo, CostLine, CostReport
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Enumeration of damage severity levels.

    Values are lowercase strings so they round-trip cleanly through JSON.

    Attributes:
        minor: Cosmetic damage repairable by polish or spot-paint.
        moderate: Damage that requires body work.
        severe: Damage that most likely requires part replacement.
    """

    minor = "minor"
    moderate = "moderate"
    severe = "severe"


SEVERITY_RANK = {Severity.minor: 0, Severity.moderate: 1, Severity.severe: 2}
"""Mapping from :class:`Severity` to an integer rank, used for ``max()`` comparisons."""

UNKNOWN_PART = "unknown_part"
"""Sentinel string assigned to a damage when no car part could be identified."""


@dataclass
class Detection:
    """Raw output from a single YOLO segmentation prediction.

    This is an internal type — it is never serialised to JSON because
    ``mask`` is a NumPy array.

    Attributes:
        cls_id: Integer class index as returned by the YOLO model.
        cls_name: Human-readable class name (e.g. ``"dent"``, ``"hood"``).
        confidence: Detection confidence in the range ``[0, 1]``.
        bbox: Bounding box in original-image pixel coordinates
            ``(x1, y1, x2, y2)``.
        mask: Boolean array of shape ``(H, W)`` in the original image
            resolution. ``True`` where the instance mask is active.
        area_px: Number of ``True`` pixels in :attr:`mask`.
    """

    cls_id: int
    cls_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (original image px)
    mask: np.ndarray                          # bool (H, W), original image size
    area_px: int


# ---------------- report models ----------------

class DamageOnPart(BaseModel):
    """A single damage instance associated with a car part, from one photo.

    Attributes:
        damage_type: Damage class name (e.g. ``"dent"``, ``"scratch"``).
        part: Part name (e.g. ``"hood"``) or :data:`UNKNOWN_PART` if no part
            could be identified.
        confidence: YOLO detection confidence of the damage mask.
        coverage: Fraction of the damage mask area that lies inside the part
            mask. ``1.0`` means the damage is entirely within the part.
        area_ratio: Damage area divided by part area. When the part is unknown
            this is relative to the full image area instead.
        severity: Estimated severity of this damage instance.
        severity_source: How severity was determined —
            ``"heuristic"`` (area-based rule) or ``"vlm"`` (VLM assessment).
        part_source: How the part was identified — ``"overlap"`` (mask
            intersection), ``"vlm"`` (VLM guess), or ``"unknown"``.
        bbox: Bounding box of the damage in original-image pixel coordinates
            ``[x1, y1, x2, y2]``.
    """

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
    """Analysis result for a single input photograph.

    Attributes:
        photo_id: Filename (basename) of the source image.
        usable: ``False`` when the photo could not be read or the VLM
            determined it does not show a car.
        notes: Non-fatal observations about the photo (blur warning,
            VLM quality note, etc.).
        parts_found: Sorted list of distinct part names detected by the parts
            segmentation model.
        items: All damage-on-part instances detected in this photo.
    """

    photo_id: str
    usable: bool = True
    notes: list[str] = []
    parts_found: list[str] = []
    items: list[DamageOnPart] = []


class FusedDamage(BaseModel):
    """A damage group that has been merged across multiple photos.

    Grouping key is ``(part, damage_type)``. See :mod:`fusion` for the
    aggregation rules.

    Attributes:
        part: Car part name.
        damage_type: Damage class name.
        severity: Maximum severity observed across all photos.
        count: Maximum number of simultaneous instances seen in any single
            photo. A dent viewed from three angles still counts as 1; two
            distinct dents in one photo count as 2.
        max_confidence: Highest YOLO confidence across all matching detections.
        photos: Sorted list of photo IDs where this damage was seen.
        needs_review: ``True`` when at least one review reason was raised
            (low confidence, unknown part, VLM-only part identification).
        review_reasons: Human-readable explanations for why this group needs
            manual review.
    """

    part: str
    damage_type: str
    severity: Severity
    count: int = Field(description="max number of instances seen in any single photo")
    max_confidence: float
    photos: list[str]
    needs_review: bool = False
    review_reasons: list[str] = []


class CaseReport(BaseModel):
    """Top-level analysis report for a complete damage assessment case.

    Attributes:
        case_id: Short unique identifier for this case (8-hex-char UUID fragment).
        photos: One :class:`PhotoResult` per input image.
        fused: Deduplicated, cross-photo damage groups produced by
            :func:`~fusion.fuse_case`.
        warnings: Case-level warnings (e.g. "no usable photos").
    """

    case_id: str
    photos: list[PhotoResult]
    fused: list[FusedDamage]
    warnings: list[str] = []


# ---------------- cost models ----------------

class Tier(str, Enum):
    """Vehicle market segment, used to select the appropriate labour-rate row.

    Attributes:
        economy: Budget / entry-level vehicles.
        mid: Mid-range vehicles.
        luxury: Premium / luxury vehicles.
    """

    economy = "economy"
    mid = "mid"
    luxury = "luxury"


class CarInfo(BaseModel):
    """Identifying information about the vehicle being assessed.

    Attributes:
        make: Manufacturer name (e.g. ``"Toyota"``).
        model: Model name (e.g. ``"Corolla"``).
        year: Model year (e.g. ``2018``).
        tier: Market segment used to look up labour rates and part prices.
    """

    make: str
    model: str
    year: int
    tier: Tier


class CostLine(BaseModel):
    """Cost breakdown for repairing or replacing a single car part.

    All monetary values are in Egyptian Pounds (EGP) and are rounded to
    the nearest 10 EGP.

    Attributes:
        part: Full part name (e.g. ``"left_front-door"``).
        action: The repair action chosen — one of ``repair_only``,
            ``paint_only``, ``repair_and_paint``, or ``replace``.
        damages: Human-readable list of the damage descriptions that drove
            this cost line (e.g. ``["dent x1 (moderate)"]``).
        parts_min: Lower bound of the replacement-parts cost.
        parts_max: Upper bound of the replacement-parts cost.
        labor_min: Lower bound of the labour cost.
        labor_max: Upper bound of the labour cost.
        paint_min: Lower bound of the paint cost.
        paint_max: Upper bound of the paint cost.
        total_min: Sum of all min costs (parts + labour + paint).
        total_max: Sum of all max costs (parts + labour + paint).
        price_source: Where the parts price came from — ``"placeholder table"``,
            a URL from the price agent, or ``"-"`` when no parts are needed.
        needs_review: ``True`` when this line includes uncertain data that a
            human should verify before trusting the total.
        notes: Additional notes explaining non-obvious cost decisions.
    """

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
    """Complete cost estimate for repairing all detected damage on a vehicle.

    Attributes:
        car: The vehicle this estimate applies to.
        currency: Currency of all monetary values (always ``"EGP"``).
        lines: One :class:`CostLine` per affected part, sorted by
            descending ``total_max``.
        total_min: Sum of all ``CostLine.total_min`` values.
        total_max: Sum of all ``CostLine.total_max`` values.
        unconfirmed_min: Portion of ``total_min`` that comes from lines
            flagged ``needs_review``.
        unconfirmed_max: Portion of ``total_max`` that comes from lines
            flagged ``needs_review``.
        unpriced: Damage descriptions that could not be priced because the
            affected part is unknown. These are NOT included in the totals.
        notes: Report-level notes (placeholder prices, unpriced items, etc.).
    """

    car: CarInfo
    currency: str = "EGP"
    lines: list[CostLine]
    total_min: float
    total_max: float
    unconfirmed_min: float = 0       # part of the total that comes from needs_review lines
    unconfirmed_max: float = 0
    unpriced: list[str] = []         # damages we could not price (unknown part)
    notes: list[str] = []
