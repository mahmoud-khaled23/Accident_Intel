"""Vehicle tracking service — wraps src/vehicle_tracking/tracker.py."""
import sys
from pathlib import Path

# Ensure project root and src/ are on sys.path
_project_root = str(Path(__file__).parents[3])   # Accident_Intel/
_src_path = str(Path(__file__).parents[3] / "src")
for _p in (_project_root, _src_path):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def run_tracking(
    video_path: str,
    accident_id: str,
    output_dir: str,
    mock: bool = False,
    model_path: str = "yolo11x.pt",
) -> dict:
    """Run vehicle tracking and return a trajectory dict.

    When mock=True a realistic fake response is returned immediately.
    When mock=False the real YOLO tracker is invoked.
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
        from vehicle_tracking.tracker import process_video
    except ImportError as exc:
        raise RuntimeError(
            f"vehicle_tracking package not importable — is ultralytics installed? Original error: {exc}"
        ) from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video = str(out_dir / f"{accident_id}_tracked.mp4")
    output_json = str(out_dir / f"{accident_id}_trajectories.json")

    from vehicle_tracking.accident_candidates import identify_accident_candidates

    from .video_service import inspect_video
    metadata = inspect_video(video_path)
    raw = process_video(video_path, output_video, output_json, model_path=model_path)

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
    """Bound the model context regardless of video length; never invent speeds."""
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
