"""Pure repair-action rules — no prices, no I/O, no external dependencies.

Given a part group, the base part name, the damage type, and the severity, this
module decides *what work* is required using a deterministic rule table.  The
four possible actions are defined by :class:`Action`.

Key exports:
    Action, REPAIR_ACTIONS, PAINT_ACTIONS, PANEL_BASES, NO_REPLACE,
    PAINT_AFTER_REPLACE, SIZE, EXPECTED_GROUP,
    base_name, part_group, is_mismatch, decide_action, merge_actions
"""
from __future__ import annotations

import re
from enum import Enum

from schemas import Severity


class Action(str, Enum):
    """The set of repair actions that can be applied to a damaged part.

    Attributes:
        repair_only: Body work (e.g. PDR / dent pulling) with no paint.
        paint_only: Surface preparation and repaint only.
        repair_and_paint: Body work followed by repainting.
        replace: Full part replacement (and repainting where applicable).
    """

    repair_only = "repair_only"
    paint_only = "paint_only"
    repair_and_paint = "repair_and_paint"
    replace = "replace"


REPAIR_ACTIONS = {Action.repair_only, Action.repair_and_paint}
"""Actions that involve physical body-repair work."""

PAINT_ACTIONS = {Action.paint_only, Action.repair_and_paint}
"""Actions that involve repainting."""

PANEL_BASES = {
    "hood", "trunk", "roof", "front-door", "back-door",
    "fender", "quarter-panel", "rocker-panel",
}
"""Base names of sheet-metal panels that belong to the ``"panel"`` group."""

# welded to the body: "replace" falls back to repair_and_paint
NO_REPLACE = {"quarter-panel", "roof", "rocker-panel"}
"""Parts that are welded to the body structure and cannot be bolt-on replaced;
a ``replace`` action is downgraded to ``repair_and_paint`` for these."""

# a new part of these groups arrives unpainted
PAINT_AFTER_REPLACE = {"bumper", "panel"}
"""Part groups whose replacement parts arrive unpainted and therefore always
incur an additional paint cost when replaced."""

# relative size for repair / paint cost (default 1.0)
SIZE = {"fender": 0.7, "rocker-panel": 0.7, "grille": 0.4, "mirror": 0.4}
"""Size multipliers applied to the per-unit repair and paint labour rates.
Parts not listed here default to ``1.0``."""

# damage types that mean "replace" when found on the matching group
EXPECTED_GROUP = {"glass shatter": "glass", "lamp broken": "lamp", "tire flat": "wheel"}
"""Mapping from damage type to the part group where it is the canonical damage.
When the damage type matches the part group, the action is always ``replace``."""


def base_name(part: str) -> str:
    """Strip the left/right side prefix from a part name.

    Args:
        part: Full part name, possibly prefixed with ``"left_"`` or
            ``"right_"`` (e.g. ``"left_front-door"``).

    Returns:
        The base part name without any side prefix
        (e.g. ``"front-door"``).

    Example:
        >>> base_name("left_front-door")
        'front-door'
        >>> base_name("hood")
        'hood'
    """
    return re.sub(r"^(left|right)_", "", part)


def part_group(part: str) -> str | None:
    """Map a part name to its functional group.

    Groups are used to look up labour-rate categories and replacement rules.

    Args:
        part: Full or base part name (side prefix is stripped internally).

    Returns:
        One of ``"bumper"``, ``"panel"``, ``"grille"``, ``"lamp"``,
        ``"glass"``, ``"wheel"``, ``"mirror"``, or ``"plate"``, or
        ``None`` if the part is not recognised.

    Example:
        >>> part_group("left_front-door")
        'panel'
        >>> part_group("right_headlight")
        'lamp'
        >>> part_group("front-bumper")
        'bumper'
    """
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
    """Return ``True`` when a damage type is unexpected on the given part group.

    Used to flag potentially misclassified detections for human review
    (e.g. ``"lamp broken"`` detected on a door panel).

    Args:
        damage: Damage class name (e.g. ``"lamp broken"``).
        group: Part group string as returned by :func:`part_group`.

    Returns:
        ``True`` if ``damage`` is a type with an expected group and that
        expected group does not match ``group``; ``False`` otherwise.

    Example:
        >>> is_mismatch("lamp broken", "panel")
        True
        >>> is_mismatch("dent", "panel")
        False
    """
    exp = EXPECTED_GROUP.get(damage)
    return exp is not None and exp != group


def decide_action(group: str, base: str, damage: str, sev: Severity) -> Action:
    """Determine the repair action for a single damage on a single part.

    Rules are applied in priority order:
    1. If the damage type canonically belongs to this group (e.g. ``"glass
       shatter"`` on ``"glass"``), always replace.
    2. Glass, lamp, and plate groups always replace.
    3. Wheel damage: ``"scratch"`` → paint only; anything else → repair and
       paint.
    4. Mirror and grille: ``"scratch"`` → paint only; anything else →
       replace.
    5. Bumper / panel: action depends on damage type and severity.
    6. Parts in :data:`NO_REPLACE` have ``replace`` downgraded to
       ``repair_and_paint``.

    Args:
        group: Part group as returned by :func:`part_group`.
        base: Base part name (no side prefix) as returned by :func:`base_name`.
        damage: Damage class name (e.g. ``"dent"``, ``"scratch"``).
        sev: Damage severity.

    Returns:
        The :class:`Action` that should be applied.

    Example:
        >>> decide_action("panel", "hood", "dent", Severity.moderate)
        <Action.repair_and_paint: 'repair_and_paint'>
        >>> decide_action("glass", "windshield", "glass shatter", Severity.severe)
        <Action.replace: 'replace'>
    """
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
    """Collapse multiple per-damage actions on the same part into one final job.

    A single part may have several damage detections requiring different
    actions. The merge rules ensure paint is charged only once and that the
    most invasive action dominates:
    * Any ``replace`` → ``replace``.
    * Any mix of repair + paint actions → ``repair_and_paint``.
    * Repair only (no paint) → ``repair_only``.
    * Paint only (no repair) → ``paint_only``.

    Args:
        actions: Set of :class:`Action` values from individual damages on the
            same part.

    Returns:
        A single :class:`Action` representing the combined job.

    Example:
        >>> merge_actions({Action.repair_only, Action.paint_only})
        <Action.repair_and_paint: 'repair_and_paint'>
        >>> merge_actions({Action.replace, Action.paint_only})
        <Action.replace: 'replace'>
    """
    if Action.replace in actions:
        return Action.replace
    needs_repair = bool(actions & REPAIR_ACTIONS)
    needs_paint = bool(actions & PAINT_ACTIONS)
    if needs_repair and needs_paint:
        return Action.repair_and_paint
    return Action.repair_only if needs_repair else Action.paint_only
