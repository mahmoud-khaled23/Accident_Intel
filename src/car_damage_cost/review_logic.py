"""Pure helpers for the Streamlit review screen (no Streamlit imports, easy to test).

The review table has one row per fused damage:
    part | damage_type | severity | count | confidence | needs_review | reviewed
The user can edit part / type / severity / count, tick `reviewed`, delete rows,
and add rows for damage the models missed (those have no confidence).
"""
from __future__ import annotations

import pandas as pd

from .schemas import UNKNOWN_PART, FusedDamage, Severity

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
"""Canonical list of recognised car part names shown in the UI dropdown."""

PART_CHOICES = PART_NAMES + [UNKNOWN_PART]
"""All valid part name choices, including the unknown-part sentinel."""

DAMAGE_TYPES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
"""Valid damage type strings shown in the damage-type dropdown."""

SEVERITIES = [s.value for s in Severity]
"""Valid severity string values (``"minor"``, ``"moderate"``, ``"severe"``)."""

COLUMNS = ["part", "damage_type", "severity", "count", "confidence", "needs_review", "reviewed"]
"""Ordered column names for the review DataFrame."""


def fused_to_rows(fused: list[FusedDamage]) -> list[dict]:
    """Convert a list of :class:`~schemas.FusedDamage` objects to plain row dicts.

    Each dict contains the fields expected by the review table:
    ``part``, ``damage_type``, ``severity``, ``count``, ``confidence``,
    ``needs_review``, and ``reviewed`` (always ``False`` initially).

    Args:
        fused: Fused damage list from :func:`~fusion.fuse_case`.

    Returns:
        A list of dicts ready to be passed to :func:`make_df`.
    """
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
    """Build a typed pandas DataFrame from a list of review row dicts.

    Enforces the correct dtype for each column so Streamlit's
    ``st.data_editor`` uses the appropriate widget type (checkbox for bools,
    number input for ints, etc.).

    Args:
        rows: List of row dicts, e.g. from :func:`fused_to_rows`.

    Returns:
        A :class:`~pandas.DataFrame` with the columns defined in
        :data:`COLUMNS` and appropriate dtypes.
    """
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
    """Return ``True`` when a cell value is effectively empty.

    Handles ``None``, floating-point ``NaN``, ``pd.NA``, and the empty
    string, which all indicate a missing value in the review table.

    Args:
        v: Cell value from the edited DataFrame.

    Returns:
        ``True`` if ``v`` represents a missing/blank value.
    """
    return v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA or v == ""


def rows_to_fused(rows: list[dict]) -> tuple[list[FusedDamage], list[str]]:
    """Edited table rows -> FusedDamage list for the cost engine (+ warnings for skipped rows).

    Converts the user-edited rows from the Streamlit review table back into
    :class:`~schemas.FusedDamage` objects that can be passed to the cost
    engine.  Rows added manually by the user (no ``confidence`` value) are
    treated as already reviewed.  Rows that are missing ``part`` or
    ``damage_type`` are skipped with a warning.

    Args:
        rows: List of dicts representing the current state of the review
            table, typically from ``st.data_editor``.

    Returns:
        A tuple ``(fused_list, warnings)`` where:
        * ``fused_list`` is a list of :class:`~schemas.FusedDamage` objects.
        * ``warnings`` is a list of human-readable strings for any rows that
          were skipped due to missing required fields.
    """
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
