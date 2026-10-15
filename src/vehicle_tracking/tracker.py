"""Vehicle detection and tracking (YOLO11x + BoT-SORT + Supervision).

Heavy libraries (ultralytics, supervision) are imported lazily inside
``process_video`` so that importing this module is cheap and never
triggers a model download.
"""

import json
from pathlib import Path

import cv2

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parents[2] / "weights/vehicle_tracking/yolo11x.pt")
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO ids)
IMGSZ = 1024
CONF = 0.25
IOU = 0.5
TRACKER = "botsort.yaml"


def process_video(
    input_path,
    output_video,
    output_json,
    model_path=DEFAULT_MODEL_PATH,
    model=None,
):
    """Detect and track vehicles in a video.

    Saves an annotated video (boxes, class labels, track IDs, traces) and a
    trajectories JSON file.

    Args:
        input_path: Path to the input video.
        output_video: Path of the annotated output video (.mp4).
        output_json: Path of the trajectories JSON file.
        model_path: YOLO weights to load (ignored if ``model`` is given).
        model: Optional pre-loaded ``YOLO`` instance, to reuse across videos.
            Note: tracker state persists inside the model, so use a fresh
            model (or ``model=None``) for each independent video.

    Returns:
        dict: The trajectories, in the same structure that is written to JSON.

    Raises:
        FileNotFoundError: If the input video does not exist.
        RuntimeError: If the video cannot be opened or the writer fails.
        ValueError: If the video metadata (fps / width / height) is invalid.
    """
    input_path = Path(input_path)
    output_video = Path(output_video)
    output_json = Path(output_json)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    import supervision as sv
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(input_path))
    out = None
    trajectories = {}

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not fps > 0 or width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid video metadata: fps={fps}, width={width}, height={height}"
            )

        if model is None:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            model = YOLO(model_path)

        output_video.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not out.isOpened():
            raise RuntimeError(f"Could not create output video: {output_video}")

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        trace_annotator = sv.TraceAnnotator(trace_length=50, thickness=2)

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                imgsz=IMGSZ,
                persist=True,
                tracker=TRACKER,
                conf=CONF,
                iou=IOU,
                classes=VEHICLE_CLASSES,
                verbose=False,
            )

            result = results[0]
            detections = sv.Detections.from_ultralytics(result)

            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()

                for box, track_id in zip(result.boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
                    cls = int(box.cls[0])

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    if track_id not in trajectories:
                        trajectories[track_id] = {
                            "class": model.names[cls],
                            "trajectory": {},
                        }

                    trajectories[track_id]["trajectory"][str(frame_idx)] = {
                        "center": [cx, cy],
                        "box": [x1, y1, x2, y2],
                        "confidence": float(box.conf[0]),
                    }

            labels = []
            if detections.tracker_id is not None:
                for class_id, tracker_id in zip(
                    detections.class_id, detections.tracker_id
                ):
                    labels.append(f"{model.names[class_id]} ID:{tracker_id}")
            else:
                labels = [model.names[class_id] for class_id in detections.class_id]

            annotated_frame = frame.copy()
            # Traces need tracker IDs; with none present there is nothing to draw.
            if detections.tracker_id is not None:
                annotated_frame = trace_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            out.write(annotated_frame)
            frame_idx += 1
    finally:
        cap.release()
        if out is not None:
            out.release()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=4)

    return trajectories
