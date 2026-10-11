"""Step 5: merge per-photo results so the same damage isn't counted twice.

Grouping key: (part, damage_type).
- severity   -> max across photos
- confidence -> max across photos
- count      -> max number of instances in any SINGLE photo
               (the same dent seen from 3 angles stays 1; two dents in one photo = 2)
"""
from __future__ import annotations

from collections import defaultdict

from schemas import (
    SEVERITY_RANK,
    UNKNOWN_PART,
    FusedDamage,
    PhotoResult,
)

LOW_CONF = 0.40


def fuse_case(photos: list[PhotoResult]) -> list[FusedDamage]:
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