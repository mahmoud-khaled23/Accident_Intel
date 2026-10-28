"""Step 5: merge per-photo results so the same damage isn't counted twice.

Grouping key: (part, damage_type).
- severity   -> max across photos
- confidence -> max across photos
- count      -> max number of instances in any SINGLE photo
               (the same dent seen from 3 angles stays 1; two dents in one photo = 2)
"""
from __future__ import annotations

from collections import defaultdict




import numpy as np 

from .schemas import (
    SEVERITY_RANK,
    UNKNOWN_PART,
    FusedDamage,
    PhotoResult,
)

LOW_CONF = 0.40
"""Confidence threshold below which a detection is flagged for human review."""


def fuse_case(photos: list[PhotoResult]) -> list[FusedDamage]:
    """Merge per-photo damage detections into a deduplicated cross-photo list.

    Groups detections by ``(part, damage_type)``. For each group:

    * **severity** — the highest severity seen in any photo.
    * **confidence** — the highest detection confidence seen in any photo.
    * **count** — the maximum number of simultaneous instances observed in a
      single photo (viewing the same dent from three angles still counts as 1;
      two distinct dents in one photo count as 2).
    * **needs_review** — ``True`` when any detection has low confidence, an
      unknown part, or a VLM-only part identification.

    Unusable photos (``usable=False``) are silently skipped.

    The returned list is sorted by descending severity, then descending
    confidence, so the most critical damages appear first.

    Args:
        photos: List of :class:`~schemas.PhotoResult` objects from
            :func:`~pipeline.analyze_photo`, one per input image.

    Returns:
        A deduplicated list of :class:`~schemas.FusedDamage` objects.
    """
    sev = {}
    conf = defaultdict(float)
    count = defaultdict(int)
    seen_in = defaultdict(set)
    reasons: dict[tuple[str, str], set[str]] = defaultdict(set)

    for ph in photos:
        if not ph.usable:
            continue
        per_photo = defaultdict(int)
        for it in ph.items:
            key = (it.part, it.damage_type)
            per_photo[key] += 1
            seen_in[key].add(ph.photo_id)
            conf[key] = max(conf[key], it.confidence)

            if key not in sev or SEVERITY_RANK[it.severity] > SEVERITY_RANK[sev[key]]:
                sev[key] = it.severity

            if it.part == UNKNOWN_PART:
                reasons[key].add("part could not be identified")
            elif it.part_source == "vlm":
                reasons[key].add("part guessed by VLM only")
            if it.confidence < LOW_CONF:
                reasons[key].add("low detection confidence")

        for key, n in per_photo.items():
            count[key] = max(count[key], n)

    fused = []
    for key, s in sev.items():
        part, dmg = key
        r = sorted(reasons.get(key, set()))
        fused.append(
            FusedDamage(
                part=part,
                damage_type=dmg,
                severity=s,
                count=count[key],
                max_confidence=round(conf[key], 3),
                photos=sorted(seen_in[key]),
                needs_review=bool(r),
                review_reasons=r,
            )
        )

    fused.sort(key=lambda f: (-SEVERITY_RANK[f.severity], -f.max_confidence))
    return fused
