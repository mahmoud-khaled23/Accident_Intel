"""Vehicle tracking service — wraps src/vehicle_tracking/tracker.py.

Provides two public functions used by the accidents pipeline:

* :func:`run_tracking` — run YOLO-based vehicle tracking on a video file.
* :func:`summarize_tracking` — compress tracking data to a size safe to
  pass to an LLM context window.
"""
from pathlib import Path

def run_tracking(
    video_path: str,
    accident_id: str,
    output_dir: str,
    mock: bool = False,
    model_path: str = "weights/vehicle_tracking/yolo11x.pt",
) -> dict:
    """Run vehicle tracking on a video and return a trajectory dict.

    When ``mock=True`` a realistic fake response is returned immediately
    without loading any models. When ``mock=False`` the real YOLO tracker is
    invoked via :func:`vehicle_tracking.tracker.process_video`.

    The output dict always contains:
    * ``vehicle_count`` — number of unique tracked vehicles.
    * ``duration_seconds`` — video duration.
    * ``fps`` — video frame rate.
    * ``trajectories`` — list of per-vehicle trajectory dicts, each with
      ``id``, ``class``, and a ``frames`` list.

    In real mode, an ``accident_candidates`` key is also present with
    heuristic collision candidates from
    :func:`vehicle_tracking.accident_candidates.identify_accident_candidates`.

    Args:
        video_path: Absolute path to the input video file.
        accident_id: Unique identifier used to name output files.
        output_dir: Directory where the annotated video and trajectory JSON
            are written.
        mock: Return mock data without running inference.
        model_path: YOLO model file path or name (e.g. ``"yolo11x.pt"``).

    Returns:
        A dict with tracking results as described above.

    Raises:
        RuntimeError: If ``mock=False`` and the ``vehicle_tracking`` package
            is not importable.
    """
    if mock:
        return {
            "vehicle_count": 3,
            "duration_seconds": 12.5,
            "fps": 25.0,
            "trajectories": [
                {
                    "id": 1,
                    "class": "car",
                    "frames": [
                        {"frame": 0, "bbox": [100, 200, 250, 350], "confidence": 0.92},
                    ],
                },
                {
                    "id": 2,
                    "class": "car",
                    "frames": [
                        {"frame": 0, "bbox": [400, 180, 560, 340], "confidence": 0.88},
                    ],
                },
                {
                    "id": 3,
                    "class": "motorcycle",
                    "frames": [
                        {"frame": 5, "bbox": [50, 300, 150, 430], "confidence": 0.75},
                    ],
                },
            ],
        }

    # Real mode — lazy import to avoid loading heavy deps at startup
    try:
        from src.vehicle_tracking.tracker import process_video
    except ImportError as exc:
        raise RuntimeError(
            f"vehicle_tracking package not importable — is ultralytics installed? Original error: {exc}"
        ) from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video = str(out_dir / f"{accident_id}_tracked.mp4")
    output_json = str(out_dir / f"{accident_id}_trajectories.json")

    from src.vehicle_tracking.accident_candidates import identify_accident_candidates

    from .video_service import inspect_video
    metadata = inspect_video(video_path)
    raw = process_video(video_path, output_video, output_json, model_path=model_path)

    try:
        from src.api.model_cost.usage import record_usage  # noqa: PLC0415
        from pathlib import Path as _Path  # noqa: PLC0415
        tracking_model_id = _Path(model_path).stem
        record_usage(
            model_id=tracking_model_id,
            stage="vehicle_tracking",
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            job_id=str(accident_id) if accident_id else None,
        )
    except Exception as exc:
        import logging  # noqa: PLC0415
        logging.getLogger(__name__).warning(
            "model_cost: instrumentation skipped: %s", exc
        )

    for info in raw.values():
        for fdata in info.get("trajectory", {}).values():
            box = fdata.get("box", [])
            if "center" not in fdata and len(box) == 4:
                fdata["center"] = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    trajectories = []
    for track_id, info in raw.items():
        frames = []
        for frame_str, fdata in info.get("trajectory", {}).items():
            box = fdata.get("box", [])
            frames.append({"frame": int(frame_str), "timestamp": int(frame_str) / metadata["fps"],
                           "bbox": box, "confidence": fdata.get("confidence", 0.0)})
        frames.sort(key=lambda frame: frame["frame"])
        trajectories.append(
            {"id": int(track_id), "class": info.get("class", "vehicle"), "frames": frames}
        )

    return {
        "vehicle_count": len(trajectories),
        **metadata,
        "trajectories": trajectories,
        "accident_candidates": identify_accident_candidates(raw),
    }

def summarize_tracking(data: dict) -> dict:
    """Compress tracking data to a size safe for an LLM context window.

    Limits to 100 vehicles and 8 sampled frames per vehicle. Never invents
    speed values — only positional bounding-box data is included.

    Args:
        data: Full tracking result dict as returned by :func:`run_tracking`.

    Returns:
        A condensed dict with ``vehicle_count``, ``duration_seconds``,
        ``width``, ``height``, ``vehicles`` (sampled), ``accident_candidates``
        (up to 20), and a ``limitations`` disclaimer string.
    """
    vehicles = []
    for track in data.get("trajectories", [])[:100]:
        frames = track.get("frames", [])
        sample = [frames[round(i * (len(frames) - 1) / 7)] for i in range(8)] if len(frames) > 8 else frames
        vehicles.append({"id": track["id"], "class": track["class"],
                         "first_seen": frames[0].get("timestamp") if frames else None,
                         "last_seen": frames[-1].get("timestamp") if frames else None,
                         "sampled_positions": sample})
    candidates = data.get("accident_candidates") or []
    return {"vehicle_count": data.get("vehicle_count", 0),
            "duration_seconds": data.get("duration_seconds"),
            "width": data.get("width"), "height": data.get("height"),
            "vehicles": vehicles,
            "accident_candidates": [
                {
                    "vehicle_ids": item.get("vehicle_ids"),
                    "suspicion_score": item.get("suspicion_score"),
                    "evidence": item.get("evidence"),
                    "overlap_frame": item.get("overlap_frame"),
                }
                for item in candidates[:20]
            ],
            "limitations": "Boxes are image pixels, not calibrated speeds. Track IDs may switch or fragment. "
                           "Candidate overlaps are pixel-space heuristics, not confirmed collisions."}
