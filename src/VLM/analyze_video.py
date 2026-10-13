#!/usr/bin/env python3
"""Analyze a CCTV video with Qwen2-VL and print a traffic-incident report.

Examples:
    python analyze_video.py --video clip_07.mp4
    python analyze_video.py --video clip_07.mp4 --prompt-mode json --n-frames 16
    python analyze_video.py --video clip_07.mp4 --prompt-file my_prompt.txt -o report.txt
"""
import argparse
from pathlib import Path

import cv2
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from prompts import SYSTEM_PROMPT, TRAFFIC_ANALYSIS_PROMPT, build_accident_prompt
except ImportError:
    from .prompts import SYSTEM_PROMPT, TRAFFIC_ANALYSIS_PROMPT, build_accident_prompt

DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="Path to the input video")
    p.add_argument("--prompt-mode", choices=["json", "report"], default="report",
                   help="'json' = structured checklist prompt, 'report' = traffic analysis prompt")
    p.add_argument("--prompt-file", help="Text file with a custom prompt (overrides --prompt-mode)")
    p.add_argument("--n-frames", type=int, default=32, help="Number of sampled frames")
    p.add_argument("--max-new-tokens", type=int, default=700, help="Max generated tokens")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("-o", "--output", help="Optional file to save the report")
    args = p.parse_args()

    if args.n_frames < 1:
        p.error("--n-frames must be at least 1")
    if args.max_new_tokens < 1:
        p.error("--max-new-tokens must be at least 1")
    return args


def load_model(model_id: str):
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto"
    )
    print("Model loaded successfully!")
    print("Device mapping:", model.hf_device_map)
    return model, processor


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or total_frames <= 0:
        raise ValueError(f"Could not read video metadata. Check path: {video_path}")
    return total_frames / fps


def build_prompt(args, duration_s: float) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt_mode == "json":
        return build_accident_prompt(args.n_frames, duration_s)
    return TRAFFIC_ANALYSIS_PROMPT


def analyze(model, processor, video_path: str, prompt: str, n_frames: int, max_new_tokens: int) -> str:
    # Qwen video sampling requires an even count that fits in the source clip.
    cap = cv2.VideoCapture(video_path)
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if frame_count < 2:
        raise ValueError("The local visual model requires a video with at least two frames.")
    n_frames = min(max(2, n_frames), frame_count) // 2 * 2
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "nframes": n_frames},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def main():
    args = parse_args()
    duration_s = get_video_duration(args.video)
    prompt = build_prompt(args, duration_s)

    model, processor = load_model(args.model_id)
    report = analyze(model, processor, args.video, prompt, args.n_frames, args.max_new_tokens)

    print("-" * 70)
    print(f"Prompt mode: {args.prompt_mode} | Video duration: {duration_s:.2f}s | Requested frames: {args.n_frames}")
    print(report)
    print("-" * 70)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
