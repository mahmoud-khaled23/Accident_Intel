"""Accidents router."""
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from ..dependencies import get_settings
from ..models import (
    create_accident,
    delete_accident,
    get_accident,
    list_accidents,
    update_accident,
)
from ..schemas.accident import AccidentCreate, AccidentResponse, AnalysisStatus
from ..schemas.common import PaginatedResponse
from ..services import accident_service

router = APIRouter(prefix="/api/accidents", tags=["accidents"])

_IN_PROGRESS_STATUSES = {"processing", "tracking", "vlm_analysis", "law_retrieval", "reasoning"}


def _row_to_response(record: dict) -> AccidentResponse:
    return AccidentResponse(
        id=record["id"],
        status=record.get("status", ""),
        location=record.get("location", ""),
        date_time=record.get("date_time", ""),
        description=record.get("description", ""),
        notes=record.get("notes", ""),
        created_at=record.get("created_at", ""),
        tracking_data=record.get("tracking_data"),
        vlm_description=record.get("vlm_description"),
        laws=record.get("laws"),
        report=record.get("report"),
    )


# NOTE: /upload must be registered BEFORE /{accident_id} to avoid routing ambiguity.

@router.get("/", response_model=PaginatedResponse[AccidentResponse])
def list_accidents_endpoint(
    page: int = 1,
    limit: int = 20,
    status: str | None = None,
    settings=Depends(get_settings),
):
    items, total = list_accidents(str(settings.db_path), page=page, limit=limit, status=status)
    return PaginatedResponse(
        items=[_row_to_response(r) for r in items],
        total=total,
        page=page,
        limit=limit,
        page_size=limit,
    )


@router.post("/", response_model=AccidentResponse, status_code=status.HTTP_201_CREATED)
def create_accident_endpoint(
    body: AccidentCreate,
    settings=Depends(get_settings),
):
    accident_id = str(uuid.uuid4())
    data = {
        "status": "created",
        "location": body.location,
        "date_time": body.date_time,
        "description": body.description,
        "notes": body.notes,
    }
    record = create_accident(str(settings.db_path), accident_id, data)
    return _row_to_response(record)


@router.post("/upload", response_model=AccidentResponse, status_code=status.HTTP_201_CREATED)
async def upload_accident(
    file: UploadFile,
    location: Optional[str] = Form(None),
    incident_date: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    settings=Depends(get_settings),
):
    """Combined endpoint: create accident record AND save video in one request."""
    accident_id = str(uuid.uuid4())

    # Determine storage path using settings.data_dir
    dest_dir = settings.data_dir / "accidents" / accident_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    video_path = dest_dir / "video.mp4"

    # Save the uploaded video atomically
    async with aiofiles.open(video_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Create the DB record with all metadata + video_path in one write
    data = {
        "status": "uploaded",
        "location": location or "",
        "date_time": incident_date or "",
        "description": description or "",
        "notes": notes or "",
        "video_path": str(video_path),
        "video_filename": file.filename or "video.mp4",
    }
    record = create_accident(str(settings.db_path), accident_id, data)
    return _row_to_response(record)


@router.post("/{accident_id}/video")
async def upload_video(
    accident_id: str,
    file: UploadFile,
    settings=Depends(get_settings),
):
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")

    dest_dir = settings.data_dir / "accidents" / accident_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    video_path = dest_dir / "video.mp4"

    async with aiofiles.open(video_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    update_accident(
        str(settings.db_path),
        accident_id,
        {"status": "uploaded", "video_path": str(video_path)},
    )
    return {"message": "Video uploaded successfully", "path": str(video_path)}


@router.post("/{accident_id}/analyze", response_model=AnalysisStatus)
def trigger_analysis(
    accident_id: str,
    background_tasks: BackgroundTasks,
    settings=Depends(get_settings),
):
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")

    if record.get("status") in _IN_PROGRESS_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Analysis already in progress (status: {record['status']})",
        )

    video_path = record.get("video_path", "")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=400, detail="No video file found — upload a video first")

    background_tasks.add_task(
        accident_service.run_analysis_pipeline,
        accident_id=accident_id,
        video_path=video_path,
        db_path=str(settings.db_path),
        settings=settings,
    )

    return AnalysisStatus(status="processing", message="Analysis pipeline started")


@router.get("/{accident_id}", response_model=AccidentResponse)
def get_accident_endpoint(accident_id: str, settings=Depends(get_settings)):
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    return _row_to_response(record)


@router.delete("/{accident_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_accident_endpoint(accident_id: str, settings=Depends(get_settings)):
    existed = delete_accident(str(settings.db_path), accident_id)
    if not existed:
        raise HTTPException(status_code=404, detail="Accident not found")


@router.get("/{accident_id}/status")
def get_status(accident_id: str, settings=Depends(get_settings)):
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    return {
        "id": record["id"],
        "status": record.get("status", ""),
        "error": record.get("error"),
    }


@router.get("/{accident_id}/video")
def get_video(accident_id: str, settings=Depends(get_settings)):
    """Serve the stored video file for an accident."""
    # Guard against path traversal
    safe_id = Path(accident_id).name
    if safe_id != accident_id:
        raise HTTPException(status_code=400, detail="Invalid accident ID")

    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")

    video_path = record.get("video_path", "")
    if not video_path:
        raise HTTPException(status_code=404, detail="No video file for this accident")

    resolved = Path(video_path).resolve()
    # Ensure the resolved path is within the configured data_dir
    allowed_root = settings.data_dir.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    return FileResponse(str(resolved), media_type="video/mp4")


@router.get("/{accident_id}/tracking")
def get_tracking(accident_id: str, settings=Depends(get_settings)):
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    tracking = record.get("tracking_data")
    if tracking is None:
        raise HTTPException(status_code=404, detail="Tracking data not yet available")
    return tracking


@router.get("/{accident_id}/description")
def get_description(accident_id: str, settings=Depends(get_settings)):
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    return {"description": record.get("vlm_description")}


@router.get("/{accident_id}/vlm")
def get_vlm(accident_id: str, settings=Depends(get_settings)):
    """Alias for /description — returns VLM analysis description."""
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    vlm_desc = record.get("vlm_description")
    if vlm_desc is None:
        raise HTTPException(status_code=404, detail="VLM analysis not yet available")
    return {"description": vlm_desc}


@router.get("/{accident_id}/laws")
def get_laws(accident_id: str, settings=Depends(get_settings)):
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    return {"laws": record.get("laws", [])}


@router.get("/{accident_id}/report")
def get_report(accident_id: str, settings=Depends(get_settings)):
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    return {"report": record.get("report")}


@router.get("/{accident_id}/liability")
def get_liability(accident_id: str, settings=Depends(get_settings)):
    """Alias for /report — returns liability assessment."""
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    return {"report": record.get("report")}
