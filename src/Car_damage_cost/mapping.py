"""Step 3: map each damage mask to the car part it lies on."""
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

# These are treated as "replace" regardless of size.
ALWAYS_SEVERE = {"glass shatter", "lamp broken", "tire flat"}


@dataclass
class MappedDamage:
    det: Detection
    part: str | None                 # None => unknown
    part_det: Detection | None
    coverage: float                  # inter / damage_area for the chosen part
    area_ratio: float
    part_source: str                 # overlap | unknown
    candidates: list[tuple[str, float]] = field(default_factory=list)


def _is_compatible(damage_type: str, part_name: str) -> bool:
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
    """Cheap first guess. Tune the thresholds on your own data."""
    if damage_type in ALWAYS_SEVERE:
        return Severity.severe
    if area_ratio < 0.05:
        return Severity.minor
    if area_ratio < 0.20:
        return Severity.moderate
    return Severity.severe