"""Exporters for machine-readable evaluation artifacts (JSON, CSV, JSONL)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..manifest import RunManifest
from ..stages.base import StageResult


def export_evaluation_artifacts(
    manifest: RunManifest,
    stage_results: dict[str, StageResult],
    output_dir: Path | str,
    latest_dir: Path | str | None = None,
) -> dict[str, str]:
    """Export all machine-readable evaluation artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    # 1. manifest.json
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    artifacts["manifest_json"] = str(manifest_path)

    # 2. summary.json
    summary_data = {
        "run_id": manifest.run_id,
        "created_at_utc": manifest.created_at_utc,
        "stages": {k: v.to_dict() for k, v in stage_results.items()},
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    artifacts["summary_json"] = str(summary_path)

    # 3. metrics.csv
    csv_rows: list[dict[str, Any]] = []
    for stage_name, res in stage_results.items():
        for metric_name, metric_val in res.metrics.items():
            if isinstance(metric_val, (int, float, str, bool)):
                csv_rows.append({
                    "run_id": manifest.run_id,
                    "stage": stage_name,
                    "status": res.status,
                    "metric": metric_name,
                    "value": metric_val,
                })

    csv_path = out / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["run_id", "stage", "status", "metric", "value"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    artifacts["metrics_csv"] = str(csv_path)

    # 4. per_sample_results.jsonl
    jsonl_path = out / "per_sample_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for stage_name, res in stage_results.items():
            for item in res.per_sample_results:
                row = dict(item)
                row["_stage"] = stage_name
                row["_run_id"] = manifest.run_id
                f.write(json.dumps(row) + "\n")
    artifacts["per_sample_jsonl"] = str(jsonl_path)

    # If latest_dir is provided, copy/mirror the latest files there too
    if latest_dir:
        latest = Path(latest_dir)
        latest.mkdir(parents=True, exist_ok=True)
        for fname in ["manifest.json", "summary.json", "metrics.csv", "per_sample_results.jsonl"]:
            src = out / fname
            if src.exists():
                dst = latest / fname
                dst.write_bytes(src.read_bytes())

    return artifacts

