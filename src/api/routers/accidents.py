"""Accidents router.

Provides CRUD endpoints for accident records and orchestrates the
upload-and-analysis workflow.  All routes are prefixed with
``/api/accidents``.

Endpoint overview:
    GET    /                         List all accidents (paginated).
    POST   /                         Create an accident record.
    POST   /upload                   Create a record AND upload a video.
    POST   /{id}/video               Replace the video for an existing record.
    POST   /{id}/analyze             Trigger the background analysis pipeline.
    GET    /{id}                     Fetch a single accident record.
    DELETE /{id}                     Delete an accident record.
    GET    /{id}/status              Fetch the current pipeline status.
    GET    /{id}/video               Serve the stored video file.
    GET    /{id}/tracking            Return tracking trajectory data.
    GET    /{id}/description         Return the VLM scene description.
    GET    /{id}/vlm                 Alias for /description.
    GET    /{id}/laws                Return retrieved traffic laws.
    GET    /{id}/report              Return the liability reasoning report.
    GET    /{id}/liability           Alias for /report.
"""
import uuid
import mimetypes
import shutil
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from ..dependencies import get_settings
from ..models import (
    create_accident,
    delete_accident,
    get_accident,
    list_accidents,
    update_accident,
    claim_analysis,
    ACTIVE_STATUSES,
)
from ..schemas.accident import AccidentCreate, AccidentResponse, AnalysisStatus
from ..schemas.common import PaginatedResponse
from ..services import accident_service
from ..services.video_service import inspect_video
from src.video_understanding.backend import VideoBackend

router = APIRouter(prefix="/api/accidents", tags=["accidents"])

_IN_PROGRESS_STATUSES = ACTIVE_STATUSES


async def _save_video(file: UploadFile, dest_dir: Path, settings) -> tuple[Path, dict]:
    """Stream an uploaded video to disk with size and format validation.

    Writes the file via a ``.part`` temporary file that is renamed atomically
    on completion, ensuring no partial files are left if an error occurs.

    Args:
        file: The uploaded video file from the FastAPI request.
        dest_dir: Directory where the video should be saved.
        settings: Application settings (used for ``max_upload_mb``).

    Returns:
        A tuple ``(destination_path, metadata)`` where ``destination_path``
        is the final :class:`~pathlib.Path` of the saved file and ``metadata``
        is the dict returned by :func:`~services.video_service.inspect_video`.

    Raises:
        :class:`~fastapi.HTTPException` 415: Unsupported video format.
        :class:`~fastapi.HTTPException` 413: File exceeds the size limit.
        :class:`~fastapi.HTTPException` 400: Empty file or invalid video.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm"}:
        await file.close()
        raise HTTPException(415, "Unsupported video format. Upload MP4, MOV, or WebM.")
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = dest_dir / f"video-{uuid.uuid4().hex}{suffix}"
    temporary = destination.with_suffix(suffix + ".part")
    size = 0
    try:
        async with aiofiles.open(temporary, "wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, f"Video exceeds {settings.max_upload_mb} MB limit.")
                await output.write(chunk)
        if size == 0:
            raise HTTPException(400, "The uploaded video is empty.")
        try:
            metadata = await run_in_threadpool(inspect_video, str(temporary))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        temporary.replace(destination)
        return destination, metadata
    finally:
        temporary.unlink(missing_ok=True)
        await file.close()


def _row_to_response(record: dict) -> AccidentResponse:
    """Convert a raw database record dict to an :class:`~schemas.accident.AccidentResponse`.

    Args:
        record: Dict as returned by :func:`~models.get_accident` or
            :func:`~models.create_accident`.

    Returns:
        A populated :class:`~schemas.accident.AccidentResponse` instance.
    """
    return AccidentResponse(
        id=record["id"],
        status=record.get("status", ""),
        location=record.get("location", ""),
        date_time=record.get("date_time", ""),
        description=record.get("description", ""),
        notes=record.get("notes", ""),
        created_at=record.get("created_at", ""),
        video_filename=record.get("video_filename", ""),
        tracking_data=record.get("tracking_data"),
        vlm_description=record.get("vlm_description"),
        laws=record.get("laws"),
        report=record.get("report"),
        error=record.get("error"),
        failed_stage=record.get("failed_stage"),
        is_demo_data=record.get("is_demo_data", False),
        video_metadata=record.get("video_metadata"),
        video_vlm_api=record.get("video_vlm_api", "auto"),
        video_vlm=record.get("video_vlm"),
    )


# NOTE: /upload must be registered BEFORE /{accident_id} to avoid routing ambiguity.

@router.get("/", response_model=PaginatedResponse[AccidentResponse])
def list_accidents_endpoint(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    page_size: int | None = Query(None, ge=1, le=200),
    status: str | None = None,
    settings=Depends(get_settings),
):
    """Return a paginated list of accident records.

    Args:
        page: 1-based page number.
        limit: Records per page (ignored when ``page_size`` is provided).
        page_size: Alias for ``limit`` accepted by the frontend.
        status: Optional filter by pipeline status string.
        settings: Injected application settings.

    Returns:
        A :class:`~schemas.common.PaginatedResponse` of
        :class:`~schemas.accident.AccidentResponse` objects.
    """
    effective_limit = page_size if page_size is not None else limit
    items, total = list_accidents(str(settings.db_path), page=page, limit=effective_limit, status=status)
    return PaginatedResponse(
        items=[_row_to_response(r) for r in items],
        total=total,
        page=page,
        limit=effective_limit,
        page_size=effective_limit,
    )


@router.post("/", response_model=AccidentResponse, status_code=status.HTTP_201_CREATED)
def create_accident_endpoint(
    body: AccidentCreate,
    settings=Depends(get_settings),
):
    """Create a new accident record without uploading a video.

    A video can be uploaded separately via ``POST /{id}/video``.

    Args:
        body: Accident creation request with location, date, and description.
        settings: Injected application settings.

    Returns:
        The newly created :class:`~schemas.accident.AccidentResponse`.
    """
    accident_id = str(uuid.uuid4())
    data = {
        "status": "created",
        "location": body.location,
        "date_time": body.date_time,
        "description": body.description,
        "notes": body.notes,
        "video_vlm_api": body.video_vlm_api or settings.video_vlm_api,
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
    video_vlm_api: VideoBackend | None = Form(None),
    settings=Depends(get_settings),
):
    """Combined endpoint: create accident record AND save video in one request.

    Streams the video to disk with size/format validation, then writes the
    database record atomically.  Cleans up the on-disk directory if the
    database write fails.

    Args:
        file: Uploaded video file (MP4, MOV, or WebM).
        location: Optional location string.
        incident_date: Optional ISO 8601 date-time string.
        description: Optional free-text description.
        notes: Optional notes.
        settings: Injected application settings.

    Returns:
        The newly created :class:`~schemas.accident.AccidentResponse` with
        ``status="uploaded"`` and video metadata populated.
    """
    accident_id = str(uuid.uuid4())

    # Determine storage path using settings.data_dir
    dest_dir = settings.data_dir / "accidents" / accident_id
    try:
        video_path, metadata = await _save_video(file, dest_dir, settings)
    except Exception:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise

    # Create the DB record with all metadata + video_path in one write
    data = {
        "status": "uploaded",
        "location": location or "",
        "date_time": incident_date or "",
        "description": description or "",
        "notes": notes or "",
        "video_path": str(video_path),
        "video_filename": file.filename or "video.mp4",
        "video_metadata": metadata,
        "video_vlm_api": video_vlm_api or settings.video_vlm_api,
    }
    try:
        record = create_accident(str(settings.db_path), accident_id, data)
    except Exception:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise
    return _row_to_response(record)


@router.post("/{accident_id}/video")
async def upload_video(
    accident_id: str,
    file: UploadFile,
    settings=Depends(get_settings),
):
    """Replace the video for an existing accident record.

    Blocked when analysis is currently in progress.  On success, any
    previous pipeline results (tracking, VLM, laws, report) are cleared so
    the new video can be re-analysed.

    Args:
        accident_id: Target accident identifier.
        file: New video file.
        settings: Injected application settings.

    Returns:
        A dict with ``message`` and ``path`` keys.

    Raises:
        :class:`~fastapi.HTTPException` 404: Accident not found.
        :class:`~fastapi.HTTPException` 409: Analysis is in progress.
    """
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")

    if record.get("status") in _IN_PROGRESS_STATUSES:
        raise HTTPException(409, "Cannot replace video while analysis is running.")

    dest_dir = settings.data_dir / "accidents" / accident_id
    video_path, metadata = await _save_video(file, dest_dir, settings)

    updated = update_accident(
        str(settings.db_path),
        accident_id,
        {"status": "uploaded", "video_path": str(video_path),
         "video_filename": file.filename or video_path.name, "video_metadata": metadata,
         "tracking_data": None, "vlm_description": None, "laws": None, "report": None,
         "error": None, "failed_stage": None, "video_vlm": None, "is_demo_data": False},
        only_if_idle=True,
    )
    if updated is None:
        video_path.unlink(missing_ok=True)
        raise HTTPException(409, "Analysis changed while uploading. Wait for it to finish before replacing the video.")
    previous = Path(record.get("video_path", ""))
    if previous.is_file() and previous.resolve().parent == dest_dir.resolve():
        previous.unlink()
    return {"message": "Video uploaded successfully", "path": str(video_path)}


@router.post("/{accident_id}/analyze", response_model=AnalysisStatus)
def trigger_analysis(
    accident_id: str,
    background_tasks: BackgroundTasks,
    video_vlm_api: VideoBackend | None = Query(None),
    settings=Depends(get_settings),
):
    """Trigger the background analysis pipeline for a video.

    The pipeline runs asynchronously via FastAPI's ``BackgroundTasks``.
    The record status is polled via ``GET /{id}/status``.

    Args:
        accident_id: Target accident identifier.
        background_tasks: FastAPI background task queue.
        settings: Injected application settings.

    Returns:
        An :class:`~schemas.accident.AnalysisStatus` confirming the job was
        queued.

    Raises:
        :class:`~fastapi.HTTPException` 404: Accident not found.
        :class:`~fastapi.HTTPException` 400: No video uploaded yet.
        :class:`~fastapi.HTTPException` 409: Analysis already in progress.
    """
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

    is_demo = any((settings.tracking_mock, settings.vlm_mock, settings.rag_mock, settings.reasoning_mock))
    if not claim_analysis(str(settings.db_path), accident_id, is_demo, video_vlm_api=video_vlm_api):
        raise HTTPException(409, "Analysis already in progress.")
    # A replacement upload may have completed between the first read and claim.
    # Once claimed, replacement/deletion are blocked until the job finishes.
    video_path = get_accident(str(settings.db_path), accident_id)["video_path"]

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
    """Fetch a single accident record by ID.

    Args:
        accident_id: Target accident identifier.
        settings: Injected application settings.

    Returns:
        The :class:`~schemas.accident.AccidentResponse` for this accident.

    Raises:
        :class:`~fastapi.HTTPException` 404: Accident not found.
    """
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    return _row_to_response(record)


@router.delete("/{accident_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_accident_endpoint(accident_id: str, settings=Depends(get_settings)):
    """Delete an accident record and its associated files.

    Also removes the on-disk directory under ``data_dir/accidents/{id}``.

    Args:
        accident_id: Target accident identifier.
        settings: Injected application settings.

    Raises:
        :class:`~fastapi.HTTPException` 404: Accident not found.
        :class:`~fastapi.HTTPException` 409: Analysis is in progress.
    """
    record = get_accident(str(settings.db_path), accident_id)
    if record and record.get("status") in _IN_PROGRESS_STATUSES:
        raise HTTPException(409, "Cannot delete an analysis while it is running.")
    try:
        existed = delete_accident(str(settings.db_path), accident_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if not existed:
        raise HTTPException(status_code=404, detail="Accident not found")
    shutil.rmtree(settings.data_dir / "accidents" / accident_id, ignore_errors=True)


@router.get("/{accident_id}/status")
def get_status(accident_id: str, settings=Depends(get_settings)):
    """Return the current pipeline status for an accident.

    Args:
        accident_id: Target accident identifier.
        settings: Injected application settings.

    Returns:
        A dict with ``id``, ``status``, and ``error`` keys.

    Raises:
        :class:`~fastapi.HTTPException` 404: Accident not found.
    """
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
    """Serve the stored video file for an accident.

    Validates the path is within the configured data directory to prevent
    path traversal.

    Args:
        accident_id: Target accident identifier.
        settings: Injected application settings.

    Returns:
        A :class:`~fastapi.responses.FileResponse` streaming the video.

    Raises:
        :class:`~fastapi.HTTPException` 400: Invalid accident ID.
        :class:`~fastapi.HTTPException` 403: Path outside allowed root.
        :class:`~fastapi.HTTPException` 404: Accident or video not found.
    """
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

    return FileResponse(str(resolved), media_type=mimetypes.guess_type(resolved.name)[0] or "video/mp4")


@router.get("/{accident_id}/tracking")
def get_tracking(accident_id: str, settings=Depends(get_settings)):
    """Return tracking trajectory data for an accident.

    Args:
        accident_id: Target accident identifier.
        settings: Injected application settings.

    Returns:
        The tracking data dict produced by the tracking service.

    Raises:
        :class:`~fastapi.HTTPException` 404: Accident not found, or tracking not yet available.
    """
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    tracking = record.get("tracking_data")
    if tracking is None:
        raise HTTPException(status_code=404, detail="Tracking data not yet available")
    return tracking


@router.get("/{accident_id}/description")
def get_description(accident_id: str, settings=Depends(get_settings)):
    """Return the VLM scene description for an accident.

    Args:
        accident_id: Target accident identifier.
        settings: Injected application settings.

    Returns:
        A dict with a ``description`` key.

    Raises:
        :class:`~fastapi.HTTPException` 404: Accident not found.
    """
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    return {"description": record.get("vlm_description")}


@router.get("/{accident_id}/vlm")
def get_vlm(accident_id: str, settings=Depends(get_settings)):
    """Alias for /description — returns VLM analysis description.

    Args:
        accident_id: Target accident identifier.
        settings: Injected application settings.

    Returns:
        A dict with a ``description`` key.

    Raises:
        :class:`~fastapi.HTTPException` 404: Accident not found, or VLM result not yet available.
    """
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    vlm_desc = record.get("vlm_description")
    if vlm_desc is None:
        raise HTTPException(status_code=404, detail="VLM analysis not yet available")
    return {"description": vlm_desc}


@router.get("/{accident_id}/laws")
def get_laws(accident_id: str, settings=Depends(get_settings)):
    """Return retrieved traffic laws for an accident.

    Args:
        accident_id: Target accident identifier.
        settings: Injected application settings.

    Returns:
        A dict with a ``laws`` key containing a list of law entries.

    Raises:
        :class:`~fastapi.HTTPException` 404: Accident not found.
    """
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    return {"laws": record.get("laws", [])}


@router.get("/{accident_id}/report")
def get_report(accident_id: str, settings=Depends(get_settings)):
    """Return the liability reasoning report for an accident.

    Args:
        accident_id: Target accident identifier.
        settings: Injected application settings.

    Returns:
        A dict with a ``report`` key.

    Raises:
        :class:`~fastapi.HTTPException` 404: Accident not found.
    """
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    return {"report": record.get("report")}


@router.get("/{accident_id}/liability")
def get_liability(accident_id: str, settings=Depends(get_settings)):
    """Alias for /report — returns liability assessment.

    Args:
        accident_id: Target accident identifier.
        settings: Injected application settings.

    Returns:
        A dict with a ``report`` key.

    Raises:
        :class:`~fastapi.HTTPException` 404: Accident not found.
    """
    record = get_accident(str(settings.db_path), accident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    return {"report": record.get("report")}
