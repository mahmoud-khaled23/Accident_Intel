"""POST /analyze and GET /images/{analysis_id}/{path:path} endpoints."""
from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

# Make existing modules importable
_SRC = str(Path(__file__).parent.parent.parent / "src" / "Car_damage_cost")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Base output directory
BASE_OUT = Path(__file__).parent.parent.parent / "out" / "damage_cases"

# Allowed file extensions and max upload size
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

# Media type map for serving images
_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

router = APIRouter(tags=["analyze"])


@router.post("/analyze")
async def analyze(
    files: list[UploadFile],
    make: str = Form(""),
    model_name: str = Form(""),
    year: int = Form(2020),
    tier: str = Form("mid"),
    use_vlm: bool = Form(True),
    use_web: bool = Form(False),
):
    # --- validation ---
    if not files:
        raise HTTPException(status_code=422, detail="At least one image file is required.")

    # Read all file contents first so we can validate and then save
    file_contents: list[tuple[str, bytes]] = []
    for upload in files:
        name = upload.filename or "upload"
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_EXTS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type '{suffix}' for file '{name}'. "
                       f"Allowed: {', '.join(sorted(ALLOWED_EXTS))}",
            )
        content = await upload.read()
        if len(content) > MAX_SIZE_BYTES:
            raise HTTPException(
                status_code=422,
                detail=f"File '{name}' exceeds the 20 MB size limit.",
            )
        file_contents.append((name, content))

    # --- generate analysis ID and directories ---
    analysis_id = uuid.uuid4().hex[:12]
    upload_dir = BASE_OUT / analysis_id / "uploads"
    debug_dir = BASE_OUT / analysis_id / "debug"
    upload_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # --- save uploads with safe ASCII filenames ---
    upload_paths: list[Path] = []
    safe_names: list[str] = []
    for original_name, content in file_contents:
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(original_name).name)
        # Ensure no path separators sneak through
        safe_name = safe_name.replace("/", "_").replace("\\", "_")
        dest = upload_dir / safe_name
        dest.write_bytes(content)
        upload_paths.append(dest)
        safe_names.append(safe_name)

    try:
        from damage_backend.models_cache import get_models
        from pipeline import analyze_case
        from review_logic import fused_to_rows

        cfg, damage_model, parts_model, vlm_client = get_models()

        report = analyze_case(
            upload_paths,
            damage_model,
            parts_model,
            vlm_client if use_vlm else None,
            cfg,
            case_id=analysis_id,
            debug_dir=debug_dir,
        )

        rows = fused_to_rows(report.fused)

        # Build per-photo image URL dicts
        photos_out = []
        for photo in report.photos:
            stem = Path(photo.photo_id).stem
            image_urls = {
                "original": f"/api/images/{analysis_id}/uploads/{photo.photo_id}",
                "combined": f"/api/images/{analysis_id}/debug/{stem}_combined.jpg",
                "parts": f"/api/images/{analysis_id}/debug/{stem}_parts.jpg",
                "damage": f"/api/images/{analysis_id}/debug/{stem}_damage.jpg",
            }
            photo_dict = photo.model_dump()
            photo_dict["image_urls"] = image_urls
            photos_out.append(photo_dict)

        response = {
            "analysis_id": analysis_id,
            "case_id": report.case_id,
            "warnings": report.warnings,
            "photos": photos_out,
            "fused": [f.model_dump() for f in report.fused],
            "rows": rows,
        }

        # Save meta.json for later reference by cost endpoint
        meta = {
            "analysis_id": analysis_id,
            "upload_paths": [str(p) for p in upload_paths],
            "rows": rows,
            "fused": [f.model_dump() for f in report.fused],
        }
        meta_path = BASE_OUT / analysis_id / "meta.json"
        meta_path.write_text(
            json.dumps(meta, default=str, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        from damage_backend.serializers import make_json_safe

        return JSONResponse(content=make_json_safe(response))

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/images/{analysis_id}/{path:path}")
def serve_image(analysis_id: str, path: str):
    # Validate analysis_id is purely alphanumeric (no traversal)
    if not re.fullmatch(r"[a-zA-Z0-9]+", analysis_id):
        raise HTTPException(status_code=400, detail="Invalid analysis ID.")

    case_dir = (BASE_OUT / analysis_id).resolve()
    full_path = (BASE_OUT / analysis_id / path).resolve()

    # Reject directory traversal
    try:
        full_path.relative_to(case_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied.")

    suffix = full_path.suffix.lower()
    if suffix not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")

    media_type = _MEDIA_TYPES.get(suffix, "image/jpeg")
    return FileResponse(str(full_path), media_type=media_type)
