"""Step 3: map each damage mask to the car part it lies on.

For each damage detection, this module finds the car-part detection whose mask
has the greatest pixel overlap with the damage mask. A compatibility check
ensures that semantically improbable pairings (e.g. a broken-lamp detection on
a door) are deprioritised in favour of a compatible part that clears the
overlap threshold.

Key exports:
    COMPAT, ALWAYS_SEVERE, MappedDamage,
    map_damage_to_parts, heuristic_severity
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from schemas import Detection, Severity

# Some damage types only make sense on certain parts. If a compatible part
# overlaps enough, prefer it over a (slightly) larger incompatible overlap
# (e.g. a broken lamp mask that also touches the bumper).
COMPAT: dict[str, tuple[str, ...]] = {
    "glass shatter": ("glass", "window", "windshield", "windscreen"),
    "lamp broken": ("light", "lamp", "headlight", "taillight"),
    "tire flat": ("tire", "tyre", "wheel"),
}
"""Damage-type → tuple of substrings that must appear in the part name for the
part to be considered *compatible* with that damage type.  Damages not listed
here are considered compatible with any part."""

# These are treated as "replace" regardless of size.
ALWAYS_SEVERE = {"glass shatter", "lamp broken", "tire flat"}
"""Damage types whose heuristic severity is always :attr:`~schemas.Severity.severe`
regardless of the area ratio, because they require replacement by definition."""


@dataclass
class MappedDamage:
    """Association between one damage detection and its best-matching part.

    Attributes:
        det: The original damage :class:`~schemas.Detection`.
        part: Name of the matched car part, or ``None`` if no part met the
            overlap threshold.
        part_det: The part :class:`~schemas.Detection` that was chosen, or
            ``None`` if no match was found.
        coverage: Fraction of the damage mask area that overlaps the chosen
            part mask (``intersection / damage_area``). ``0.0`` when unknown.
        area_ratio: Ratio used for heuristic severity estimation. When a part
            was matched this is ``damage_area / part_area``; when the part is
            unknown it is ``damage_area / image_area``.
        part_source: How the part was found — ``"overlap"`` (mask
            intersection) or ``"unknown"`` (no match).
        candidates: Up to 3 ``(part_name, coverage)`` tuples for the
            highest-overlap parts, for debugging.
    """

    det: Detection
    part: str | None                 # None => unknown
    part_det: Detection | None
    coverage: float                  # inter / damage_area for the chosen part
    area_ratio: float
    part_source: str                 # overlap | unknown
    candidates: list[tuple[str, float]] = field(default_factory=list)


def _is_compatible(damage_type: str, part_name: str) -> bool:
    """Check whether a damage type is semantically compatible with a part name.

    Compatibility is determined by substring matching against the tuples in
    :data:`COMPAT`.  Damage types not listed in :data:`COMPAT` are always
    considered compatible.

    Args:
        damage_type: Damage class name (e.g. ``"lamp broken"``).
        part_name: Part class name as returned by the parts model
            (e.g. ``"left_headlight"``).

    Returns:
        ``True`` if the damage type is compatible with the part name.
    """
    keys = COMPAT.get(damage_type)
    if not keys:
        return True
    p = part_name.lower()
    return any(k in p for k in keys)


def map_damage_to_parts(
    damages: list[Detection],
    parts: list[Detection],
    image_area: int,
    threshold: float = 0.30,
) -> list[MappedDamage]:
    """Associate each damage detection with its most likely car part.

    For each damage, the function:

    1. Computes the pixel overlap between the damage mask and every part mask.
    2. Sorts candidate parts by overlap fraction (descending).
    3. Selects the first compatible part that clears ``threshold``.
    4. Falls back to the highest-overlap part (regardless of compatibility)
       if no compatible part clears the threshold.
    5. Produces a ``MappedDamage`` with ``part=None`` when no part clears the
       threshold at all.

    Args:
        damages: Damage detections from the damage YOLO model.
        parts: Part detections from the parts YOLO model.
        image_area: Total pixel area of the image (``height * width``), used
            as the denominator for ``area_ratio`` when no part is matched.
        threshold: Minimum overlap fraction (``intersection / damage_area``)
            required for a part to be accepted as a match. Default ``0.30``.

    Returns:
        A list of :class:`MappedDamage` objects, one per damage detection, in
        the same order as ``damages``.
    """
    out: list[MappedDamage] = []

    for d in damages:
        scored: list[tuple[float, Detection]] = []
        for p in parts:
            inter = int(np.logical_and(d.mask, p.mask).sum())
            if inter == 0:
                continue
            scored.append((inter / d.area_px, p))
        scored.sort(key=lambda t: t[0], reverse=True)

        candidates = [(p.cls_name, round(c, 3)) for c, p in scored[:3]]

        # prefer compatible parts that clear the threshold
        chosen: tuple[float, Detection] | None = None
        for c, p in scored:
            if c >= threshold and _is_compatible(d.cls_name, p.cls_name):
                chosen = (c, p)
                break
        if chosen is None and scored and scored[0][0] >= threshold:
            chosen = scored[0]

        if chosen is None:
            out.append(
                MappedDamage(
                    det=d,
                    part=None,
                    part_det=None,
                    coverage=scored[0][0] if scored else 0.0,
                    area_ratio=d.area_px / image_area,
                    part_source="unknown",
                    candidates=candidates,
                )
            )
        else:
            cov, p = chosen
            out.append(
                MappedDamage(
                    det=d,
                    part=p.cls_name,
                    part_det=p,
                    coverage=cov,
                    area_ratio=d.area_px / max(p.area_px, 1),
                    part_source="overlap",
                    candidates=candidates,
                )
            )
    return out


def heuristic_severity(damage_type: str, area_ratio: float) -> Severity:
    """Estimate damage severity from the damage type and relative area.

    This is a fast, threshold-based first guess used before any VLM call.
    Tune the thresholds (0.05 and 0.20) on your own labelled dataset.

    Args:
        damage_type: Damage class name. Types in :data:`ALWAYS_SEVERE`
            immediately return :attr:`~schemas.Severity.severe` regardless of
            ``area_ratio``.
        area_ratio: Damage area divided by part area (or image area when the
            part is unknown).  Values below 0.05 are ``minor``; below 0.20
            are ``moderate``; 0.20 and above are ``severe``.

    Returns:
        The estimated :class:`~schemas.Severity`.

    Example:
        >>> heuristic_severity("dent", 0.03)
        <Severity.minor: 'minor'>
        >>> heuristic_severity("glass shatter", 0.01)
        <Severity.severe: 'severe'>
    """
    if damage_type in ALWAYS_SEVERE:
        return Severity.severe
    if area_ratio < 0.05:
        return Severity.minor
    if area_ratio < 0.20:
        return Severity.moderate
    return Severity.severe
