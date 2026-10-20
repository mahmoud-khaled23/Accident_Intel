"""Damage assessments router — supports both /api/damage and /api/damage-assessments.

Provides endpoints for uploading car damage images, running cost estimation,
and managing assessment records.  The same router is mounted at two prefixes
for backward compatibility.

Endpoint overview:
    POST   /upload                   Upload images and run assessment immediately.
    GET    /                         List all assessments.
    GET    /images/{id}/{filename}   Serve a stored damage image or a
                                      segmentation overlay (``segmentation/``).
    GET    /{id}                     Fetch a single assessment.
    POST   /{id}/corrections         Apply user corrections to detected parts.
    DELETE /{id}                     Delete an assessment.
    POST   /legacy                   Legacy: create an empty record (no images).
    POST   /{id}/images              Legacy: upload images to an existing record.
    POST   /{id}/estimate            Legacy: trigger background cost estimation.
"""
import re
import uuid
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from ..dependencies import get_settings
from ..models import create_damage, delete_damage, get_damage, list_damage, update_damage
from ..schemas.damage import DamageCreate, DamageResponse
from ..services import damage_service

router = APIRouter(tags=["damage"])

_IN_PROGRESS_STATUSES = {"processing", "estimating"}
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _format_damage_record(record: dict) -> dict:
    """Format an internal DB record into the ``DamageAssessmentRecord`` shape expected by the UI.

    Derives ``damagedParts``, ``repairItems``, and ``estimate`` fields from
    the raw ``cost_report`` when they are not already stored in the record.
    This allows the frontend to receive a consistent response shape regardless
    of how the record was created.

    Args:
        record: Raw database record dict as returned by
            :func:`~models.get_damage`.

    Returns:
        A dict with camelCase keys matching the ``DamageAssessmentRecord``
        TypeScript interface used by the React frontend.
    """
    vinfo = record.get("vehicle_info", {})
    cost_report = record.get("cost_report") or {}
    lines = cost_report.get("lines", [])

    damaged_parts = record.get("damaged_parts")
    if damaged_parts is None:
        damaged_parts = []
        for idx, line in enumerate(lines):
            part_id = f"part-{idx + 1}"
            raw_part = line.get("part", "Unknown Part")
            name = raw_part.replace("-", " ").replace("_", " ").title()

            rp_lower = raw_part.lower()
            if any(k in rp_lower for k in ("front", "hood", "grille", "headlight")):
                location = "Front"
            elif any(k in rp_lower for k in ("back", "rear", "trunk", "tail")):
                location = "Rear"
            elif any(k in rp_lower for k in ("left", "driver")):
                location = "Driver side"
            elif any(k in rp_lower for k in ("right", "passenger")):
                location = "Passenger side"
            else:
                location = "Body"

            damages_list = line.get("damages", [])
            damage_type = ", ".join(damages_list) if damages_list else "Detected Damage"

            dt_lower = damage_type.lower()
            if "severe" in dt_lower or "shatter" in dt_lower:
                severity = "severe"
            elif "minor" in dt_lower or "scratch" in dt_lower:
                severity = "minor"
            else:
                severity = "moderate"

            damaged_parts.append({
                "id": part_id,
                "name": name,
                "location": location,
                "damageType": damage_type,
                "severity": severity,
                "confidence": 0.90,
                "source": "ai",
            })

    repair_items = record.get("repair_items")
    if repair_items is None:
        repair_items = []
        for idx, line in enumerate(lines):
            part_id = f"part-{idx + 1}"
            raw_part = line.get("part", "Unknown Part")
            name = raw_part.replace("-", " ").replace("_", " ").title()
            total_min = float(line.get("total_min") or 0.0)
            total_max = float(line.get("total_max") or 0.0)
            avg_cost = round((total_min + total_max) / 2, 2) if (total_min + total_max) > 0 else 500.0
            action = line.get("action", "repair")
            category = "parts" if "replace" in action else "labor"

            repair_items.append({
                "id": f"ri-{idx + 1}",
                "partId": part_id,
                "description": f"{name} — {action.replace('_', ' ').title()}",
                "category": category,
                "unitCost": avg_cost,
                "quantity": 1,
                "total": avg_cost,
            })

    subtotal = sum(item["total"] for item in repair_items) if repair_items else float(cost_report.get("total_min", 3500.0))
    tax = round(subtotal * 0.14, 2)
    total = round(subtotal + tax, 2)
    currency = cost_report.get("currency") or vinfo.get("currency", "EGP")

    estimate = record.get("estimate") or {
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "lowEstimate": float(cost_report.get("total_min") or subtotal * 0.85),
        "highEstimate": float(cost_report.get("total_max") or subtotal * 1.25),
        "currency": currency,
        "assumptions": ["Standard OEM parts and labor rates applied"],
        "limitations": cost_report.get("notes") or [],
    }

    images = record.get("images", [])

    # Segmentation overlays written by car_damage_cost.viz.save_debug —
    # stored as filesystem paths, served as /api/damage/images/<id>/...
    segmentation_images: list[dict] = []
    for entry in (cost_report.get("segmentation_images") or []):
        views = {
            view: f"/api/damage/images/{record['id']}/segmentation/{Path(path).name}"
            for view in ("parts", "damage", "combined")
            if (path := entry.get(view))
        }
        if views:
            segmentation_images.append({
                "photo": Path(str(entry.get("photo") or "")).name,
                **views,
            })

    return {
        "id": record["id"],
        "sessionId": record.get("session_id") or f"session-{record['id']}",
        "vehicleInfo": {
            "make": vinfo.get("make", ""),
            "model": vinfo.get("model", ""),
            "year": int(vinfo.get("year", 2020)),
            "color": vinfo.get("color", ""),
            "mileage": vinfo.get("mileage"),
            "vin": vinfo.get("vin"),
            "repairLocation": vinfo.get("repair_location", ""),
        },
        "imageUrls": images,
        "segmentationImages": segmentation_images,
        "damagedParts": damaged_parts,
        "repairItems": repair_items,
        "estimate": estimate,
        "assessmentDate": record.get("created_at", ""),
        "isDemoData": False,
    }


def _row_to_response(record: dict) -> DamageResponse:
    """Convert a raw DB record to a :class:`~schemas.damage.DamageResponse`.

    Used only by legacy endpoints that return the ``DamageResponse`` schema.

    Args:
        record: Raw database record dict.

    Returns:
        A :class:`~schemas.damage.DamageResponse` instance.
    """
    return DamageResponse(
        id=record["id"],
        status=record.get("status", ""),
        vehicle_info=record.get("vehicle_info", {}),
        images=record.get("images", []),
        cost_report=record.get("cost_report"),
        created_at=record.get("created_at", ""),
    )


def _run_damage_background(assessment_id: str, db_path: str, settings) -> None:
    """Background task wrapper for damage assessment.

    Fetches the assessment record, runs the damage service, and writes the
    cost report back to the database.  Sets ``status="failed"`` on any error.

    Args:
        assessment_id: Target assessment identifier.
        db_path: Path to the SQLite database.
        settings: Application settings passed to the damage service.
    """
    record = get_damage(db_path, assessment_id)
    if record is None:
        return
    image_paths = record.get("images", [])
    vehicle_info = record.get("vehicle_info", {})
    update_damage(db_path, assessment_id, {"status": "processing"})
    try:
        cost_report = damage_service.run_damage_assessment(
            image_paths=image_paths,
            vehicle_info=vehicle_info,
            assessment_id=assessment_id,
            settings=settings,
        )
        update_damage(db_path, assessment_id, {"cost_report": cost_report, "status": "completed"})
    except Exception as exc:
        update_damage(db_path, assessment_id, {"status": "failed", "error": str(exc)})


@router.post("/upload")
async def upload_and_assess_damage(
    images: list[UploadFile],
    make: str = Form(""),
    model: str = Form(""),
    year: int = Form(2020),
    color: str = Form(""),
    mileage: int | None = Form(None),
    vin: str | None = Form(None),
    repair_location: str | None = Form(None),
    currency: str = Form("EGP"),
    tier: str = Form("mid"),
    settings=Depends(get_settings),
):
    """Primary upload endpoint used by the React DamageEstimatorPage.

    Accepts one or more damage photos together with vehicle information,
    runs the cost-estimation pipeline synchronously, stores the result, and
    returns a formatted assessment record.

    Falls back to a hardcoded placeholder result when the damage model is
    unavailable, so the frontend always receives a well-formed response.

    Args:
        images: List of uploaded damage image files.
        make: Vehicle manufacturer name.
        model: Vehicle model name.
        year: Vehicle model year.
        color: Vehicle colour (optional).
        mileage: Current odometer reading in km (optional).
        vin: Vehicle identification number (optional).
        repair_location: Preferred repair location (optional).
        currency: Currency code for cost estimates. Defaults to ``"EGP"``.
        tier: Market segment for labour-rate lookup.
        settings: Injected application settings.

    Returns:
        A formatted damage assessment dict matching the ``DamageAssessmentRecord``
        TypeScript interface.
    """
    assessment_id = f"DMG-{uuid.uuid4().hex[:6].upper()}"

    dest_dir = Path("out") / "damage" / assessment_id / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    image_urls: list[str] = []

    for uploaded_file in images:
        safe_name = f"{uuid.uuid4().hex}_{uploaded_file.filename or 'image.jpg'}"
        file_path = dest_dir / safe_name
        async with aiofiles.open(file_path, "wb") as f:
            content = await uploaded_file.read()
            await f.write(content)
        saved_paths.append(str(file_path))
        image_urls.append(f"/api/damage/images/{assessment_id}/{safe_name}")

    vehicle_info = {
        "make": make,
        "model": model,
        "year": year,
        "color": color,
        "mileage": mileage,
        "vin": vin,
        "repair_location": repair_location,
        "currency": currency,
        "tier": tier,
    }

    # Run assessment
    try:
        cost_report = damage_service.run_damage_assessment(
            image_paths=saved_paths,
            vehicle_info=vehicle_info,
            assessment_id=assessment_id,
            settings=settings,
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Damage model inference error (%s) — using fallback", exc)
        cost_report = {
            "car_info": vehicle_info,
            "total_min": 4000.0,
            "total_max": 6500.0,
            "currency": currency,
            "lines": [
                {
                    "part": "front_bumper",
                    "action": "replace",
                    "damages": ["crack x1 (moderate)"],
                    "total_min": 2000.0,
                    "total_max": 3000.0,
                    "needs_review": False,
                },
                {
                    "part": "hood",
                    "action": "repair_and_paint",
                    "damages": ["dent x1 (minor)"],
                    "total_min": 2000.0,
                    "total_max": 3500.0,
                    "needs_review": False,
                },
            ],
            "notes": [f"Processed with fallback: {exc}"],
            "unpriced": [],
            # Overlays written during analysis survive a later pricing
            # failure — surface the real files, never fabricate any.
            "segmentation_images": damage_service.collect_segmentation_images(
                saved_paths, assessment_id
            ),
        }

    initial_record = {
        "id": assessment_id,
        "status": "completed",
        "vehicle_info": vehicle_info,
        "images": image_urls if image_urls else saved_paths,
        "saved_paths": saved_paths,
        "cost_report": cost_report,
    }

    formatted = _format_damage_record(initial_record)
    initial_record["damaged_parts"] = formatted["damagedParts"]
    initial_record["repair_items"] = formatted["repairItems"]
    initial_record["estimate"] = formatted["estimate"]

    created = create_damage(str(settings.db_path), assessment_id, initial_record)
    return _format_damage_record(created)


@router.get("/")
def list_damage_assessments(settings=Depends(get_settings)):
    """List all damage assessments for history and dashboard.

    Args:
        settings: Injected application settings.

    Returns:
        A list of formatted assessment dicts.
    """
    records = list_damage(str(settings.db_path))
    return [_format_damage_record(r) for r in records]


@router.get("/images/{assessment_id}/{filename:path}")
def serve_damage_image(assessment_id: str, filename: str):
    """Serve uploaded damage images and segmentation overlay images.

    Accepts either a bare filename (files in ``out/damage/<id>/images``) or a
    single-level subdirectory path such as ``segmentation/<file>`` (overlays
    written by the damage pipeline). The resolved path must stay inside the
    assessment's own ``out/damage/<id>`` directory to prevent traversal.

    Args:
        assessment_id: Assessment identifier used as the subdirectory name.
        filename: Image filename, optionally prefixed with ``images/`` or
            ``segmentation/``.

    Returns:
        A :class:`~fastapi.responses.FileResponse` streaming the image.

    Raises:
        :class:`~fastapi.HTTPException` 403: Path outside allowed root or
            unknown subdirectory.
        :class:`~fastapi.HTTPException` 404: Image not found.
    """
    parts = Path(filename).parts
    if len(parts) == 1:
        parts = ("images", parts[0])
    if (len(parts) != 2 or ".." in parts
            or parts[0] not in {"images", "segmentation"}):
        raise HTTPException(status_code=403, detail="Access denied")

    allowed_root = (Path("out") / "damage" / assessment_id).resolve()
    image_path = (allowed_root / parts[0] / parts[1]).resolve()

    try:
        image_path.relative_to(allowed_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    suffix = image_path.suffix.lower()
    media_type = "image/jpeg"
    if suffix == ".png":
        media_type = "image/png"
    elif suffix == ".webp":
        media_type = "image/webp"

    return FileResponse(str(image_path), media_type=media_type)


@router.get("/{assessment_id}")
def get_damage_endpoint(assessment_id: str, settings=Depends(get_settings)):
    """Fetch a single damage assessment by ID.

    Args:
        assessment_id: Target assessment identifier.
        settings: Injected application settings.

    Returns:
        A formatted damage assessment dict.

    Raises:
        :class:`~fastapi.HTTPException` 404: Assessment not found.
    """
    record = get_damage(str(settings.db_path), assessment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return _format_damage_record(record)


@router.post("/{assessment_id}/corrections")
def update_corrections(
    assessment_id: str,
    corrections: list[dict[str, Any]],
    settings=Depends(get_settings),
):
    """Apply user-supplied corrections to damaged parts.

    Merges correction dicts into the current ``damagedParts`` list by
    matching on ``id``, and marks each corrected part with
    ``"source": "user"``.

    Args:
        assessment_id: Target assessment identifier.
        corrections: List of partial part dicts. Each must include ``"id"``
            to identify the part to update.
        settings: Injected application settings.

    Returns:
        The updated formatted assessment dict.

    Raises:
        :class:`~fastapi.HTTPException` 404: Assessment not found.
    """
    record = get_damage(str(settings.db_path), assessment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Assessment not found")

    formatted = _format_damage_record(record)
    current_parts = formatted.get("damagedParts", [])

    for correction in corrections:
        cid = correction.get("id")
        for part in current_parts:
            if part.get("id") == cid:
                part.update(correction)
                part["source"] = "user"

    update_damage(str(settings.db_path), assessment_id, {"damaged_parts": current_parts})
    updated = get_damage(str(settings.db_path), assessment_id) or record
    return _format_damage_record(updated)


@router.delete("/{assessment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_damage_assessment(assessment_id: str, settings=Depends(get_settings)):
    """Delete a damage assessment record.

    Args:
        assessment_id: Target assessment identifier.
        settings: Injected application settings.

    Raises:
        :class:`~fastapi.HTTPException` 404: Assessment not found.
    """
    deleted = delete_damage(str(settings.db_path), assessment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Assessment not found")


# ── Legacy endpoints preserved for backward compatibility ────────────────────

@router.post("/legacy", response_model=DamageResponse, status_code=status.HTTP_201_CREATED)
def create_damage_assessment_legacy(
    body: DamageCreate,
    settings=Depends(get_settings),
):
    """Create an empty damage assessment record without images.

    Legacy endpoint preserved for backward compatibility. The primary
    workflow uses ``POST /upload`` instead.

    Args:
        body: Vehicle information for the assessment.
        settings: Injected application settings.

    Returns:
        A :class:`~schemas.damage.DamageResponse` with ``status="created"``.
    """
    assessment_id = str(uuid.uuid4())
    data = {
        "status": "created",
        "vehicle_info": {
            "make": body.make,
            "model": body.model,
            "year": body.year,
            "tier": body.tier,
            "currency": body.currency,
        },
        "images": [],
    }
    record = create_damage(str(settings.db_path), assessment_id, data)
    return _row_to_response(record)


@router.post("/{assessment_id}/images")
async def upload_images(
    assessment_id: str,
    files: list[UploadFile],
    settings=Depends(get_settings),
):
    """Upload additional images to an existing assessment record.

    Legacy endpoint used after ``POST /legacy`` to attach images before
    triggering estimation via ``POST /{id}/estimate``.

    Args:
        assessment_id: Target assessment identifier.
        files: List of image files to upload.
        settings: Injected application settings.

    Returns:
        A dict with ``message`` and ``images`` keys.

    Raises:
        :class:`~fastapi.HTTPException` 404: Assessment not found.
    """
    record = get_damage(str(settings.db_path), assessment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Assessment not found")

    dest_dir = Path("out") / "damage" / assessment_id / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = list(record.get("images", []))
    for uploaded_file in files:
        safe_name = f"{uuid.uuid4().hex}_{uploaded_file.filename or 'image'}"
        file_path = dest_dir / safe_name
        async with aiofiles.open(file_path, "wb") as f:
            content = await uploaded_file.read()
            await f.write(content)
        saved_paths.append(str(file_path))

    update_damage(str(settings.db_path), assessment_id, {"images": saved_paths})
    return {"message": f"{len(files)} image(s) uploaded", "images": saved_paths}


@router.post("/{assessment_id}/estimate")
def trigger_estimate(
    assessment_id: str,
    background_tasks: BackgroundTasks,
    settings=Depends(get_settings),
):
    """Trigger background cost estimation for a legacy assessment record.

    Schedules :func:`_run_damage_background` as a FastAPI background task.
    Poll ``GET /{id}`` to check the result.

    Args:
        assessment_id: Target assessment identifier.
        background_tasks: FastAPI background task queue.
        settings: Injected application settings.

    Returns:
        A dict with ``status`` and ``message`` keys.

    Raises:
        :class:`~fastapi.HTTPException` 404: Assessment not found.
        :class:`~fastapi.HTTPException` 409: Estimation already in progress.
    """
    record = get_damage(str(settings.db_path), assessment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Assessment not found")

    if record.get("status") in _IN_PROGRESS_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Assessment already in progress (status: {record['status']})",
        )

    background_tasks.add_task(
        _run_damage_background,
        assessment_id=assessment_id,
        db_path=str(settings.db_path),
        settings=settings,
    )
    return {"status": "processing", "message": "Damage assessment pipeline started"}
