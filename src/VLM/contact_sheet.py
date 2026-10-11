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
