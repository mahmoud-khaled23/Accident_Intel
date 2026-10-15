"""Step 6: deterministic cost engine (no LLM here).

Converts a list of :class:`~schemas.FusedDamage` objects into a
:class:`~schemas.CostReport` by:

1. Grouping damages by physical part.
2. Deciding the required repair action via :mod:`cost_rules`.
3. Fetching the part replacement price from the price agent (or the
   placeholder table as a fallback).
4. Applying labour and paint rates from the tier-specific rate table.
5. Producing min/max cost ranges rounded to the nearest 10 EGP.

Key functions:
    load_tables: Load the JSON price/rate tables from disk.
    estimate_cost: Main entry point — fused damages → :class:`~schemas.CostReport`.
"""
from __future__ import annotations

import json
from pathlib import Path

from cost_rules import (
    NO_REPLACE,
    PAINT_AFTER_REPLACE,
    REPAIR_ACTIONS,
    SIZE,
    Action,
    base_name,
    decide_action,
    is_mismatch,
    merge_actions,
    part_group,
)
from schemas import (
    SEVERITY_RANK,
    CarInfo,
    CostLine,
    CostReport,
    FusedDamage,
    Severity,
)

SEV_MULT = {Severity.minor: 0.6, Severity.moderate: 1.0, Severity.severe: 1.6}
"""Per-severity multipliers applied to the base repair labour rate."""

LOW, HIGH = 0.85, 1.20
"""Spread factors applied around a single price estimate to produce a min/max
range (±15 % / +20 % of the central value)."""

TIER_IDX = {"economy": 0, "mid": 1, "luxury": 2}
"""Index into the price-table list for each vehicle tier."""

PLACEHOLDER = "placeholder table"
"""``price_source`` label used when the price comes from the static JSON table
rather than a live web search."""


def load_tables(path: str | Path) -> dict:
    """Load the JSON price and labour-rate tables from disk.

    Args:
        path: Path to the ``prices.json`` file.

    Returns:
        Parsed dictionary with at minimum the keys ``"part_prices"`` and
        ``"tiers"``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _price_key(base: str) -> str:
    """Normalise wheel-related part names to a single lookup key.

    ``front-wheel`` and ``back-wheel`` map to ``"tire"`` because the only
    wheel component in the price table is the tyre itself.

    Args:
        base: Base part name (no side prefix).

    Returns:
        The key to use when looking up the part in the price table.
    """
    return "tire" if base.endswith("wheel") else base


def _round10(x: float) -> float:
    """Round a monetary value to the nearest 10 EGP.

    Args:
        x: Raw cost value in EGP.

    Returns:
        Value rounded to the nearest 10.
    """
    return float(round(x / 10.0) * 10)


def _part_price(part, base, car, tables, price_agent):
    """Look up the replacement-part price for one part.

    Tries the price agent first (live web search result); falls back to the
    placeholder table when the agent is unavailable or returns ``None``.

    Args:
        part: Full part name (used by the price agent for the query).
        base: Base part name (used for the price-table key).
        car: :class:`~schemas.CarInfo` with make/model/year/tier.
        tables: Parsed price tables dict from :func:`load_tables`.
        price_agent: A :class:`~price_agent.PriceAgent` instance or ``None``.

    Returns:
        A tuple ``((min_egp, max_egp), source_string, note_or_None)`` where
        ``note_or_None`` is a human-readable warning string when no price is
        available, or ``None`` when the price was found.
    """
    key = _price_key(base)
    row = tables["part_prices"].get(key)
    expected = row[TIER_IDX[car.tier.value]] if row else None

    if price_agent is not None:
        res = price_agent.get_price(car, part if key != "tire" else "tire", expected)
        if res:
            lo, hi, src = res
            return (lo, hi), src, None

    if expected is None:
        return (0.0, 0.0), "-", f"no price available for {key}"
    return (expected * LOW, expected * HIGH), PLACEHOLDER, None


def _build_line(part, w, car, tables, price_agent) -> CostLine:
    """Build one :class:`~schemas.CostLine` from the work dict for a part.

    Applies the merged repair action to compute parts, labour, and paint
    costs, then assembles a :class:`~schemas.CostLine`.

    Args:
        part: Full part name (e.g. ``"left_front-door"``).
        w: Work dictionary for this part with keys ``"group"``, ``"base"``,
            ``"items"`` (list of ``(FusedDamage, Action)`` tuples),
            ``"review"`` (bool), and ``"notes"`` (list of str).
        car: Vehicle information for rate lookup.
        tables: Parsed price tables dict.
        price_agent: Optional price agent for live pricing.

    Returns:
        A populated :class:`~schemas.CostLine` with min/max cost ranges.
    """
    t = tables["tiers"][car.tier.value]
    group, base, items = w["group"], w["base"], w["items"]
    final = merge_actions({a for _, a in items})

    size = SIZE.get(base, 1.0)
    repair_unit, paint_unit = t["repair"] * size, t["paint"] * size

    parts = labor = paint = (0.0, 0.0)
    source, notes = "-", list(w["notes"])

    if final == Action.replace:
        parts, source, note = _part_price(part, base, car, tables, price_agent)
        if note:
            notes.append(note)
            w["review"] = True
        labor = (t["replace_labor"] * LOW, t["replace_labor"] * HIGH)
        if group in PAINT_AFTER_REPLACE:
            paint = (paint_unit * LOW, paint_unit * HIGH)
    else:
        if final in REPAIR_ACTIONS:
            rep = [f for f, a in items if a in REPAIR_ACTIONS]
            worst = max((f.severity for f in rep), key=lambda s: SEVERITY_RANK[s])
            n = sum(f.count for f in rep)
            cost = repair_unit * SEV_MULT[worst] * n
            labor = (cost * LOW, cost * HIGH)
        if final in (Action.paint_only, Action.repair_and_paint):
            paint = (paint_unit * LOW, paint_unit * HIGH)

    if any(a == Action.repair_and_paint and base in NO_REPLACE and f.severity == Severity.severe
           for f, a in items):
        notes.append("severe damage on a welded panel: priced as repair+paint, may need structural work")

    r = _round10
    return CostLine(
        part=part,
        action=final.value,
        damages=[f"{f.damage_type} x{f.count} ({f.severity.value})" for f, _ in items],
        parts_min=r(parts[0]), parts_max=r(parts[1]),
        labor_min=r(labor[0]), labor_max=r(labor[1]),
        paint_min=r(paint[0]), paint_max=r(paint[1]),
        total_min=r(parts[0] + labor[0] + paint[0]),
        total_max=r(parts[1] + labor[1] + paint[1]),
        price_source=source,
        needs_review=w["review"],
        notes=notes,
    )


def estimate_cost(
    fused: list[FusedDamage],
    car: CarInfo,
    tables: dict,
    price_agent=None,
) -> CostReport:
    """Convert fused damage detections into a complete cost estimate.

    For each :class:`~schemas.FusedDamage`:

    1. Determine the part group with :func:`~cost_rules.part_group`.
    2. Decide the required action with :func:`~cost_rules.decide_action`.
    3. Merge multiple actions on the same part with
       :func:`~cost_rules.merge_actions`.
    4. Build a :class:`~schemas.CostLine` with :func:`_build_line`.

    Damages on unrecognised parts (``part_group`` returns ``None``) are
    collected in ``unpriced`` and excluded from the totals.

    Args:
        fused: List of :class:`~schemas.FusedDamage` objects from
            :func:`~fusion.fuse_case`.
        car: Vehicle make/model/year/tier used for rate and price lookup.
        tables: Parsed price/rate tables from :func:`load_tables`.
        price_agent: Optional :class:`~price_agent.PriceAgent` instance.
            When ``None``, only the placeholder table is used for part prices.

    Returns:
        A :class:`~schemas.CostReport` with per-part cost lines sorted by
        descending ``total_max``, plus grand totals, unpriced items, and
        report-level notes.
    """
    work: dict[str, dict] = {}
    unpriced: list[str] = []

    for f in fused:
        group = part_group(f.part)
        if group is None:
            unpriced.append(f"{f.damage_type} ({f.severity.value}) on {f.part}")
            continue
        base = base_name(f.part)
        act = decide_action(group, base, f.damage_type, f.severity)

        w = work.setdefault(
            f.part,
            {"group": group, "base": base, "items": [], "review": False, "notes": []},
        )
        w["items"].append((f, act))
        if f.needs_review:
            w["review"] = True
        if is_mismatch(f.damage_type, group):
            w["review"] = True
            w["notes"].append(f"'{f.damage_type}' does not normally occur on {f.part}")

    lines = [_build_line(p, w, car, tables, price_agent) for p, w in work.items()]
    lines.sort(key=lambda l: -l.total_max)

    notes = []
    if any(l.price_source == PLACEHOLDER for l in lines):
        notes.append("Some part prices come from the PLACEHOLDER table, not real quotes.")
    if any(l.needs_review for l in lines):
        notes.append("Lines marked needs_review are included in the total; check them before trusting it.")
    if unpriced:
        notes.append("Damages on unknown parts are NOT included in the total.")

    unconf = [l for l in lines if l.needs_review]
    return CostReport(
        car=car,
        lines=lines,
        total_min=sum(l.total_min for l in lines),
        total_max=sum(l.total_max for l in lines),
        unconfirmed_min=sum(l.total_min for l in unconf),
        unconfirmed_max=sum(l.total_max for l in unconf),
        unpriced=unpriced,
        notes=notes,
    )
