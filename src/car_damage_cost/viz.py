"""Debug visualisation: draws segmentation masks so you can SEE what the models found.

Writes 3 images per photo into the debug folder:
  <stem>_parts.jpg     all car-part masks (pastel colours + names)
  <stem>_damage.jpg    all damage masks (bright colours + class + confidence)
  <stem>_combined.jpg  parts faint underneath, damage on top, labelled "damage conf -> part"
"""
from __future__ import annotations

import colorsys
from pathlib import Path

import cv2
import numpy as np

from .schemas import UNKNOWN_PART, Detection

# BGR, bright and distinct from the pastel part colours
DAMAGE_COLORS = {
    "crack": (0, 0, 255),            # red
    "dent": (0, 140, 255),           # orange
    "glass shatter": (255, 255, 0),  # cyan
    "lamp broken": (0, 255, 255),    # yellow
    "scratch": (255, 0, 255),        # magenta
    "tire flat": (0, 255, 0),        # green
}
"""BGR colour map for known damage classes used in the debug overlay."""


def _part_color(idx: int) -> tuple[int, int, int]:
    """Generate a pastel BGR colour deterministically from a class index.

    Uses a golden-ratio hue step so adjacent indices produce visually
    distinct, easily distinguishable pastel colours.

    Args:
        idx: Integer class index from the parts YOLO model.

    Returns:
        A ``(blue, green, red)`` tuple with values in ``[0, 255]``.
    """
    h = (idx * 0.137) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.45, 0.95)  # pastel
    return int(b * 255), int(g * 255), int(r * 255)


def _damage_color(name: str, idx: int) -> tuple[int, int, int]:
    """Look up the BGR colour for a damage class name.

    Falls back to bright red for unrecognised class names.

    Args:
        name: Damage class name (e.g. ``"dent"``, ``"scratch"``).
        idx: Detection index (unused; reserved for future per-instance colouring).

    Returns:
        A ``(blue, green, red)`` tuple with values in ``[0, 255]``.
    """
    return DAMAGE_COLORS.get(name, (0, 0, 255))


def _put_label(img: np.ndarray, text: str, org: tuple[int, int], color, scale: float) -> None:
    """Draw a filled-background text label onto an image in-place.

    The label is clamped to the image boundaries so it is never partially
    outside the frame.

    Args:
        img: BGR image array to draw on (modified in place).
        text: Label string to render.
        org: Desired ``(x, y)`` anchor for the top-left corner of the text.
        color: BGR colour tuple for the text foreground.
        scale: Font scale factor passed to :func:`cv2.putText`.
    """
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    x = max(0, min(x, img.shape[1] - tw - 2))
    y = max(th + 2, min(y, img.shape[0] - 2))
    cv2.rectangle(img, (x, y - th - 3), (x + tw + 4, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 1), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw(
    base: np.ndarray,
    dets: list[Detection],
    colors: list[tuple[int, int, int]],
    labels: list[str],
    alpha: float,
    contour_px: int,
    font_scale: float,
) -> np.ndarray:
    """Blend detection masks and contours onto a base image.

    Each detection's mask is filled with its colour and blended with the
    base image using ``alpha``-blending, then contours and text labels are
    drawn on top.

    Args:
        base: Original BGR image (not modified).
        dets: List of :class:`~schemas.Detection` objects to render.
        colors: Per-detection BGR colour tuples (same length as ``dets``).
        labels: Per-detection label strings. Use ``""`` to suppress a label.
        alpha: Opacity of the mask fill layer (``0`` = transparent,
            ``1`` = opaque).
        contour_px: Contour line thickness in pixels.
        font_scale: Font size scale factor for labels.

    Returns:
        A new BGR image with masks, contours, and labels composited on top
        of ``base``.
    """
    layer = base.copy()
    for d, c in zip(dets, colors):
        layer[d.mask] = c
    out = cv2.addWeighted(layer, alpha, base, 1 - alpha, 0)

    for d, c, text in zip(dets, colors, labels):
        cnts, _ = cv2.findContours(d.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, c, contour_px, cv2.LINE_AA)
        if text:
            x1, y1, _, _ = (int(v) for v in d.bbox)
            _put_label(out, text, (x1, y1), c, font_scale)
    return out


def save_debug(
    img: np.ndarray,
    parts: list[Detection],
    damages: list[Detection],
    damage_part_names: list[str],
    out_dir: str | Path,
    stem: str,
) -> list[Path]:
    """Save three debug visualisation images for a single photo.

    Produces:
    * ``<stem>_parts.jpg``    — all car-part masks with pastel colours and names.
    * ``<stem>_damage.jpg``   — all damage masks with bright colours, class names,
      and confidence scores.
    * ``<stem>_combined.jpg`` — parts rendered faintly underneath with full-opacity
      damage masks on top; each damage label includes the assigned part name.

    ``damage_part_names[i]`` is the final part name chosen for ``damages[i]``
    (or :data:`~schemas.UNKNOWN_PART` when no part could be identified).

    Args:
        img: Original BGR image array.
        parts: Part detections from the parts YOLO model.
        damages: Damage detections from the damage YOLO model.
        damage_part_names: Resolved part name for each damage detection, in the
            same order as ``damages``.
        out_dir: Directory where the three JPEG files will be written.
            Created if it does not already exist.
        stem: Base filename stem (without extension), e.g. ``"photo1"``.

    Returns:
        A list of three :class:`~pathlib.Path` objects for the written images,
        in the order ``[parts, damage, combined]``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = img.shape[:2]
    k = max(h, w) / 1000.0
    font = max(0.4, 0.55 * k)
    line = max(2, int(round(2 * k)))

    part_colors = [_part_color(p.cls_id) for p in parts]
    part_labels = [p.cls_name for p in parts]
    dmg_colors = [_damage_color(d.cls_name, i) for i, d in enumerate(damages)]
    dmg_labels = [f"{d.cls_name} {d.confidence:.2f}" for d in damages]
    combined_labels = [
        f"{d.cls_name} {d.confidence:.2f} -> {name if name != UNKNOWN_PART else '??'}"
        for d, name in zip(damages, damage_part_names)
    ]

    parts_img = _draw(img, parts, part_colors, part_labels, 0.45, line, font)
    damage_img = _draw(img, damages, dmg_colors, dmg_labels, 0.55, line, font)

    combined = _draw(img, parts, part_colors, [""] * len(parts), 0.25, 1, font)
    # thin part names in the combined view
    for p, c in zip(parts, part_colors):
        x1, y1, _, _ = (int(v) for v in p.bbox)
        _put_label(combined, p.cls_name, (x1, y1 + int(18 * k)), c, font * 0.8)
    combined = _draw(combined, damages, dmg_colors, combined_labels, 0.55, line, font)

    paths = []
    for suffix, im in (("parts", parts_img), ("damage", damage_img), ("combined", combined)):
        p = out_dir / f"{stem}_{suffix}.jpg"
        cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(p)
    return paths
