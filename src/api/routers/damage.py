"""Damage assessments router."""
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status

from ..dependencies import get_settings
from ..models import create_damage, get_damage, update_damage
from ..schemas.damage import DamageCreate, DamageResponse
from ..services import damage_service

router = APIRouter(prefix="/api/damage-assessments", tags=["damage"])

_IN_PROGRESS_STATUSES = {"processing", "estimating"}


def _row_to_response(record: dict) -> DamageResponse:
    return DamageResponse(
        id=record["id"],
        status=record.get("status", ""),
        vehicle_info=record.get("vehicle_info", {}),
        images=record.get("images", []),
        cost_report=record.get("cost_report"),
        created_at=record.get("created_at", ""),
    )


def _run_damage_background(assessment_id: str, db_path: str, settings) -> None:
    """Background task wrapper for damage assessment."""
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


@router.post("/", response_model=DamageResponse, status_code=status.HTTP_201_CREATED)
def create_damage_assessment(
    body: DamageCreate,
    settings=Depends(get_settings),
):
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


@router.get("/{assessment_id}", response_model=DamageResponse)
def get_damage_endpoint(assessment_id: str, settings=Depends(get_settings)):
    record = get_damage(str(settings.db_path), assessment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return _row_to_response(record)
