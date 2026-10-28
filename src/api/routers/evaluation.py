"""Evaluation results sharing and reporting router."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

BASE_EVAL_DIR = Path("evaluation/results")


@router.get("/latest")
def get_latest_evaluation() -> dict[str, Any]:
    """Return the summary and manifest from the latest evaluation run."""
    latest_dir = BASE_EVAL_DIR / "latest"
    summary_file = latest_dir / "summary.json"
    manifest_file = latest_dir / "manifest.json"

    if not summary_file.is_file() or not manifest_file.is_file():
        return {
            "status": "no_results",
            "message": "No evaluation results available yet. Run 'uv run python -m evaluation.runner' to generate results.",
        }

    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        return {
            "status": "available",
            "summary": summary,
            "manifest": manifest,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation results: {e}")


@router.get("/history")
def list_evaluation_runs() -> list[dict[str, Any]]:
    """List historical evaluation runs."""
    runs_dir = BASE_EVAL_DIR / "runs"
    if not runs_dir.is_dir():
        return []

    runs = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if d.is_dir() and (d / "summary.json").is_file():
            try:
                summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
                manifest = {}
                if (d / "manifest.json").is_file():
                    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                runs.append({
                    "run_id": d.name,
                    "created_at_utc": summary.get("created_at_utc"),
                    "stages": {k: v.get("status") for k, v in summary.get("stages", {}).items()},
                    "git_commit": manifest.get("git", {}).get("commit", "unknown")[:8],
                })
            except Exception:
                continue
    return runs


@router.get("/artifacts/{filename}")
def get_evaluation_artifact(filename: str):
    """Download or view a generated evaluation artifact from the latest run."""
    # Sanitize filename
    safe_name = Path(filename).name
    latest_dir = BASE_EVAL_DIR / "latest"
    file_path = latest_dir / safe_name

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact '{safe_name}' not found.")

    if safe_name.endswith(".md"):
        return PlainTextResponse(file_path.read_text(encoding="utf-8"), media_type="text/markdown")
    if safe_name.endswith(".json"):
        return FileResponse(file_path, media_type="application/json", filename=safe_name)
    if safe_name.endswith(".csv"):
        return FileResponse(file_path, media_type="text/csv", filename=safe_name)
    if safe_name.endswith(".png"):
        return FileResponse(file_path, media_type="image/png")

    return FileResponse(file_path, filename=safe_name)

