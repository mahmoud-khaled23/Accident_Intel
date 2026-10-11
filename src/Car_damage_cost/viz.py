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

from schemas import UNKNOWN_PART, Detection

# BGR, bright and distinct from the pastel part colours
DAMAGE_COLORS = {
    "crack": (0, 0, 255),            # red
    "dent": (0, 140, 255),           # orange
    "glass shatter": (255, 255, 0),  # cyan
    "lamp broken": (0, 255, 255),    # yellow
    "scratch": (255, 0, 255),        # magenta
    "tire flat": (0, 255, 0),        # green
}


def _part_color(idx: int) -> tuple[int, int, int]:
    h = (idx * 0.137) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.45, 0.95)  # pastel
    return int(b * 255), int(g * 255), int(r * 255)


def _damage_color(name: str, idx: int) -> tuple[int, int, int]:
    return DAMAGE_COLORS.get(name, (0, 0, 255))


def _put_label(img: np.ndarray, text: str, org: tuple[int, int], color, scale: float) -> None:
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
    """damage_part_names[i] = final part chosen for damages[i] (or UNKNOWN_PART)."""
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