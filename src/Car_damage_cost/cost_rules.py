"""Pure rules, no prices. Decides WHAT work a damage needs.

Four actions only:
    repair_only | paint_only | repair_and_paint | replace
"""
from __future__ import annotations

import re
from enum import Enum

from schemas import Severity


class Action(str, Enum):
    repair_only = "repair_only"
    paint_only = "paint_only"
    repair_and_paint = "repair_and_paint"
    replace = "replace"


REPAIR_ACTIONS = {Action.repair_only, Action.repair_and_paint}
PAINT_ACTIONS = {Action.paint_only, Action.repair_and_paint}

PANEL_BASES = {
    "hood", "trunk", "roof", "front-door", "back-door",
    "fender", "quarter-panel", "rocker-panel",
}
# welded to the body: "replace" falls back to repair_and_paint
NO_REPLACE = {"quarter-panel", "roof", "rocker-panel"}
# a new part of these groups arrives unpainted
PAINT_AFTER_REPLACE = {"bumper", "panel"}
# relative size for repair / paint cost (default 1.0)
SIZE = {"fender": 0.7, "rocker-panel": 0.7, "grille": 0.4, "mirror": 0.4}

# damage types that mean "replace" when found on the matching group
EXPECTED_GROUP = {"glass shatter": "glass", "lamp broken": "lamp", "tire flat": "wheel"}


def base_name(part: str) -> str:
    """left_front-door -> front-door"""
    return re.sub(r"^(left|right)_", "", part)


def part_group(part: str) -> str | None:
    b = base_name(part)
    if b in ("front-bumper", "back-bumper"):
        return "bumper"
    if b in PANEL_BASES:
        return "panel"
    if b == "grille":
        return "grille"
    if b in ("headlight", "tail-light"):
        return "lamp"
    if b in ("windshield", "back-windshield") or b.endswith("window"):
        return "glass"
    if b.endswith("wheel"):
        return "wheel"
    if b == "mirror":
        return "mirror"
    if b == "license-plate":
        return "plate"
    return None


def is_mismatch(damage: str, group: str) -> bool:
    """e.g. 'lamp broken' found on a door -> suspicious."""
    exp = EXPECTED_GROUP.get(damage)
    return exp is not None and exp != group


def decide_action(group: str, base: str, damage: str, sev: Severity) -> Action:
    if EXPECTED_GROUP.get(damage) == group:
        return Action.replace

    if group in ("glass", "lamp", "plate"):
        return Action.replace
    if group == "wheel":
        return Action.paint_only if damage == "scratch" else Action.repair_and_paint
    if group in ("mirror", "grille"):
        return Action.paint_only if damage == "scratch" else Action.replace

    # bumper / panel
    if damage == "scratch":
        act = Action.repair_and_paint if sev == Severity.severe else Action.paint_only
    elif damage == "dent":
        act = {
            Severity.minor: Action.repair_only,
            Severity.moderate: Action.repair_and_paint,
            Severity.severe: Action.replace,
        }[sev]
    else:  # crack, or a mismatched type
        act = Action.replace if sev == Severity.severe else Action.repair_and_paint

    if act == Action.replace and base in NO_REPLACE:
        act = Action.repair_and_paint
    return act


def merge_actions(actions: set[Action]) -> Action:
    """One part can have several damages -> one final job (paint is charged once)."""
    if Action.replace in actions:
        return Action.replace
    needs_repair = bool(actions & REPAIR_ACTIONS)
    needs_paint = bool(actions & PAINT_ACTIONS)
    if needs_repair and needs_paint:
        return Action.repair_and_paint
    return Action.repair_only if needs_repair else Action.paint_only
