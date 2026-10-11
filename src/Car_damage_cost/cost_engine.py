"""Step 6: deterministic cost engine (no LLM here).

fused damages -> one job per physical part -> min/max cost in EGP.
The only external input is the price of a replacement part; it comes from the
price agent when available, otherwise from the placeholder table.
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
LOW, HIGH = 0.85, 1.20          # spread around a single estimate
TIER_IDX = {"economy": 0, "mid": 1, "luxury": 2}
PLACEHOLDER = "placeholder table"


def load_tables(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _price_key(base: str) -> str:
    """front-wheel / back-wheel -> tire (the only wheel item we replace)."""
    return "tire" if base.endswith("wheel") else base


def _round10(x: float) -> float:
    return float(round(x / 10.0) * 10)


def _part_price(part, base, car, tables, price_agent):
    """-> ((min, max), source, note|None)"""
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
