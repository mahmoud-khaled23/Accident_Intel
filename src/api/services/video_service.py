"""Validate stored uploads without loading any inference models."""
import math


def inspect_video(path: str) -> dict:
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError("The uploaded file is not a readable video.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not all(math.isfinite(v) and v > 0 for v in (fps, frame_count, width, height)):
            raise ValueError("Video has invalid frame rate, duration, or dimensions.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("The video contains no decodable frames.")
        return {"fps": fps, "frame_count": int(frame_count), "width": int(width),
                "height": int(height), "duration_seconds": frame_count / fps}
    finally:
        cap.release()
