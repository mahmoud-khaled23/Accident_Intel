"""Pure helpers for the Streamlit review screen (no Streamlit imports, easy to test).

The review table has one row per fused damage:
    part | damage_type | severity | count | confidence | needs_review | reviewed
The user can edit part / type / severity / count, tick `reviewed`, delete rows,
and add rows for damage the models missed (those have no confidence).
"""
from __future__ import annotations

import pandas as pd

from schemas import UNKNOWN_PART, FusedDamage, Severity

PART_NAMES = [
    "back-bumper", "back-windshield", "front-bumper", "grille", "hood",
    "left_back-door", "left_back-wheel", "left_back-window", "left_fender",
    "left_front-door", "left_front-wheel", "left_front-window", "left_headlight",
    "left_mirror", "left_quarter-panel", "left_rocker-panel", "left_tail-light",
    "license-plate", "right_back-door", "right_back-wheel", "right_back-window",
    "right_fender", "right_front-door", "right_front-wheel", "right_front-window",
    "right_headlight", "right_mirror", "right_quarter-panel", "right_rocker-panel",
    "right_tail-light", "roof", "trunk", "windshield",
]
PART_CHOICES = PART_NAMES + [UNKNOWN_PART]
DAMAGE_TYPES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
SEVERITIES = [s.value for s in Severity]

COLUMNS = ["part", "damage_type", "severity", "count", "confidence", "needs_review", "reviewed"]


def fused_to_rows(fused: list[FusedDamage]) -> list[dict]:
    return [
        {
            "part": f.part,
            "damage_type": f.damage_type,
            "severity": f.severity.value,
            "count": int(f.count),
            "confidence": float(f.max_confidence),
            "needs_review": bool(f.needs_review),
            "reviewed": False,
        }
        for f in fused
    ]


def make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.astype(
        {
            "part": "object",
            "damage_type": "object",
            "severity": "object",
            "count": "int64",
            "confidence": "float64",
            "needs_review": "bool",
            "reviewed": "bool",
        }
    )


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA or v == ""


def rows_to_fused(rows: list[dict]) -> tuple[list[FusedDamage], list[str]]:
    """Edited table rows -> FusedDamage list for the cost engine (+ warnings for skipped rows)."""
    out: list[FusedDamage] = []
    warnings: list[str] = []

    for i, r in enumerate(rows, 1):
        part, dmg = r.get("part"), r.get("damage_type")
        if _blank(part) or _blank(dmg):
            warnings.append(f"Row {i} skipped: part and damage type are required.")
            continue

        sev = r.get("severity")
        sev = Severity(sev) if not _blank(sev) and sev in SEVERITIES else Severity.moderate

        count = r.get("count")
        count = 1 if _blank(count) or int(count) < 1 else int(count)

        conf = r.get("confidence")
        manual = _blank(conf)                         # row added by the user
        reviewed = True if manual else bool(r.get("reviewed"))
        flagged = False if manual else bool(r.get("needs_review"))
        needs_review = flagged and not reviewed

        out.append(
            FusedDamage(
                part=str(part),
                damage_type=str(dmg),
                severity=sev,
                count=count,
                max_confidence=1.0 if manual else round(float(conf), 3),
                photos=["added by user"] if manual else ["detected"],
                needs_review=needs_review,
                review_reasons=["not reviewed yet"] if needs_review else [],
            )
        )
    return out, warnings
