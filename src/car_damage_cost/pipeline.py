"""Glue: photo -> detections -> mapping -> VLM fallback -> fusion.

This module is the central orchestrator of the damage-analysis pipeline.
It ties together YOLO segmentation, part-to-damage mapping, optional VLM
refinement, debug visualisation, and cross-photo fusion into two public
functions:

* :func:`analyze_photo` — processes a single image.
* :func:`analyze_case`  — processes a list of images and fuses the results.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import cv2

from config import Config
from fusion import fuse_case
from mapping import heuristic_severity, map_damage_to_parts
from schemas import UNKNOWN_PART, CaseReport, DamageOnPart, PhotoResult
from viz import save_debug
from vlm_client import VlmClient
from yolo_wrapper import YoloSegWrapper


def analyze_photo(
    path: str | Path,
    damage_model: YoloSegWrapper,
    parts_model: YoloSegWrapper,
    vlm: VlmClient | None,
    cfg: Config,
    debug_dir: str | Path | None = None,
) -> PhotoResult:
    """Analyse a single photo and return all detected damage-on-part associations.

    Steps performed:
    1. Read and optionally blur-check the image.
    2. Optionally ask the VLM whether the photo shows a car (``cfg.vlm_quality_check``).
    3. Run the damage segmentation model.
    4. Run the parts segmentation model.
    5. Map each damage mask to its best-matching part mask using
       :func:`~mapping.map_damage_to_parts`.
    6. For any damage whose part could not be identified by overlap, optionally
       ask the VLM to guess the part (``vlm.identify_part``).
    7. Estimate damage severity heuristically, then optionally refine with the
       VLM (``cfg.vlm_severity``).
    8. Optionally write debug visualisation images.

    Args:
        path: Path to the input image file (JPEG, PNG, etc.).
        damage_model: Loaded :class:`~yolo_wrapper.YoloSegWrapper` for damage
            detection.
        parts_model: Loaded :class:`~yolo_wrapper.YoloSegWrapper` for car-part
            detection.
        vlm: Optional :class:`~vlm_client.VlmClient` for photo sanity-check,
            part identification, and severity refinement. Pass ``None`` to
            skip all VLM calls.
        cfg: Runtime configuration from :class:`~config.Config`.
        debug_dir: If given, three debug visualisation images are written to
            this directory. ``None`` disables visualisation.

    Returns:
        A :class:`~schemas.PhotoResult` with ``usable=True`` and populated
        ``items`` on success. Returns ``usable=False`` when the image cannot
        be read or the VLM determines it does not show a car.
    """
    path = Path(path)
    photo_id = path.name
    img = cv2.imread(str(path))
    if img is None:
        return PhotoResult(photo_id=photo_id, usable=False, notes=["could not read image"])

    notes: list[str] = []

    # cheap blur check (warning only)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < cfg.blur_threshold:
        notes.append("photo looks blurry")

    # VLM sanity check: only reject when it clearly says "not a car"
    if vlm and cfg.vlm_quality_check:
        q = vlm.check_photo(img)
        if q and q.get("is_car") is False:
            return PhotoResult(
                photo_id=photo_id,
                usable=False,
                notes=notes + [f"not a car photo: {q.get('notes', '')}"],
            )
        if q and q.get("usable") is False:
            notes.append(f"VLM says photo may be unusable: {q.get('notes', '')}")

    damages = damage_model.predict(img)
    parts = parts_model.predict(img)
    h, w = img.shape[:2]
    mapped = map_damage_to_parts(damages, parts, image_area=h * w, threshold=cfg.overlap_threshold)

    items: list[DamageOnPart] = []
    for m in mapped:
        part, part_source = m.part, m.part_source

        # fallback: ask the VLM which part it is
        if part is None and vlm:
            guess = vlm.identify_part(img, m.det, parts_model.class_names)
            if guess:
                part, part_source = guess, "vlm"

        severity = heuristic_severity(m.det.cls_name, m.area_ratio)
        sev_source = "heuristic"
        if vlm and cfg.vlm_severity and part:
            v = vlm.assess_severity(img, m.det, part)
            if v:
                severity, sev_source = v, "vlm"

        items.append(
            DamageOnPart(
                damage_type=m.det.cls_name,
                part=part or UNKNOWN_PART,
                confidence=round(m.det.confidence, 3),
                coverage=round(m.coverage, 3),
                area_ratio=round(m.area_ratio, 4),
                severity=severity,
                severity_source=sev_source,
                part_source=part_source if part else "unknown",
                bbox=[round(v, 1) for v in m.det.bbox],
            )
        )

    if debug_dir:
        save_debug(img, parts, damages, [it.part for it in items], debug_dir, path.stem)

    return PhotoResult(
        photo_id=photo_id,
        usable=True,
        notes=notes,
        parts_found=sorted({p.cls_name for p in parts}),
        items=items,
    )


def analyze_case(
    paths: list[str | Path],
    damage_model: YoloSegWrapper,
    parts_model: YoloSegWrapper,
    vlm: VlmClient | None,
    cfg: Config,
    case_id: str | None = None,
    debug_dir: str | Path | None = None,
) -> CaseReport:
    """Analyse a collection of photos and return a fused case-level report.

    Calls :func:`analyze_photo` for each image in ``paths``, then fuses the
    per-photo results with :func:`~fusion.fuse_case` to produce a deduplicated
    list of cross-photo damage groups.

    Case-level warnings are added when:
    * No photos were usable.
    * No damage was detected in any usable photo.

    Args:
        paths: List of image file paths (strings or :class:`~pathlib.Path`
            objects).
        damage_model: Loaded :class:`~yolo_wrapper.YoloSegWrapper` for damage
            detection.
        parts_model: Loaded :class:`~yolo_wrapper.YoloSegWrapper` for car-part
            detection.
        vlm: Optional :class:`~vlm_client.VlmClient`. ``None`` skips all VLM
            calls.
        cfg: Runtime configuration from :class:`~config.Config`.
        case_id: Optional explicit case identifier. A random 8-hex-char ID is
            generated when not supplied.
        debug_dir: If given, debug visualisations are written for every photo.
            ``None`` disables visualisation.

    Returns:
        A :class:`~schemas.CaseReport` containing per-photo results, fused
        damage groups, and any case-level warnings.
    """
    photos = [analyze_photo(p, damage_model, parts_model, vlm, cfg, debug_dir) for p in paths]

    warnings: list[str] = []
    usable = [p for p in photos if p.usable]
    if not usable:
        warnings.append("no usable photos")
    elif not any(p.items for p in usable):
        warnings.append("no damage detected in any photo")

    return CaseReport(
        case_id=case_id or uuid.uuid4().hex[:8],
        photos=photos,
        fused=fuse_case(photos),
        warnings=warnings,
    )
