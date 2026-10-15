#!/usr/bin/env python3
"""Build a contact sheet of evenly sampled video frames (for visual inspection).

Example:
    python contact_sheet.py --video trimmed.mp4 --n-frames 32 -o sheet.png
"""
import argparse
import math

import cv2
from PIL import Image, ImageDraw


def sample_frames(video_path: str, n_frames: int):
    """Extract evenly spaced frames from a video file.

    Computes ``n_frames`` evenly spaced frame indices across the full video
    duration and reads each frame with OpenCV.  Frames that cannot be read
    (e.g. due to a corrupted position) are silently skipped.

    Args:
        video_path: Path to the video file to sample.
        n_frames: Number of frames to extract.  Must be at least 1.

    Returns:
        list[tuple[float, PIL.Image.Image]]: A list of ``(timestamp_seconds,
        RGB_image)`` pairs in presentation order.  The list may be shorter
        than ``n_frames`` if some frames cannot be decoded.

    Raises:
        ValueError: If the video cannot be opened or has invalid metadata
            (fps ≤ 0 or frame count ≤ 0).

    Example:
        >>> frames = sample_frames("clip.mp4", 8)
        >>> len(frames)
        8
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        cap.release()
        raise ValueError("Could not read video.")

    if n_frames == 1:
        indices = [0]
    else:
        indices = [round(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx / fps, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    cap.release()
    return frames


def make_sheet(frames, cols=4, thumb_w=320, thumb_h=190, label_h=25):
    """Assemble a grid of thumbnail images into a single contact sheet.

    Each cell contains a resized frame thumbnail followed by a text label
    showing the 1-based frame number and timestamp in seconds.

    Args:
        frames: List of ``(timestamp_seconds, PIL.Image.Image)`` pairs as
            returned by :func:`sample_frames`.
        cols: Number of thumbnail columns in the grid.  Defaults to ``4``.
        thumb_w: Maximum thumbnail width in pixels.  Defaults to ``320``.
        thumb_h: Maximum thumbnail height in pixels.  Defaults to ``190``.
        label_h: Height in pixels reserved below each thumbnail for the text
            label.  Defaults to ``25``.

    Returns:
        PIL.Image.Image: A white RGB image containing the assembled grid.

    Example:
        >>> sheet = make_sheet(frames, cols=4)
        >>> sheet.save("out.png")
    """
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (ts, frame) in enumerate(frames):
        x = (i % cols) * thumb_w
        y = (i // cols) * (thumb_h + label_h)
        frame.thumbnail((thumb_w, thumb_h))
        sheet.paste(frame, (x, y))
        draw.text((x + 5, y + thumb_h + 3), f"Frame {i + 1} | {ts:.2f}s", fill="black")
    return sheet


def main():
    """Parse CLI arguments, build the contact sheet, and save it to disk.

    Accepts ``--video``, ``--n-frames``, ``--cols``, and ``-o`` / ``--output``
    flags.  Samples frames from the video via :func:`sample_frames`, assembles
    them with :func:`make_sheet`, and writes the resulting PNG to the output
    path.

    Returns:
        None
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--n-frames", type=int, default=32)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("-o", "--output", default="contact_sheet.png")
    args = p.parse_args()

    sheet = make_sheet(sample_frames(args.video, args.n_frames), cols=args.cols)
    sheet.save(args.output)
    print(f"Saved contact sheet to {args.output}")


if __name__ == "__main__":
    main()
