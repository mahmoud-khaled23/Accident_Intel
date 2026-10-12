#!/usr/bin/env python3
"""Analyze a CCTV video with Qwen2-VL and print a traffic-incident report.

Examples:
    python analyze_video.py --video clip_07.mp4
    python analyze_video.py --video clip_07.mp4 --prompt-mode json --n-frames 16
    python analyze_video.py --video clip_07.mp4 --load-4bit # Fits comfortably on T4 (16GB) on Lightning AI
    python analyze_video.py --video clip_07.mp4 --model-id Qwen/Qwen2-VL-2B-Instruct # Fast lightweight model
    python analyze_video.py --video clip_07.mp4 --prompt-file my_prompt.txt -o report.txt
"""
import argparse
from pathlib import Path
import sys

# Support running directly or from repository root
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

import cv2
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from prompts import SYSTEM_PROMPT, TRAFFIC_ANALYSIS_PROMPT, build_accident_prompt
except ImportError:
    from src.VLM.prompts import SYSTEM_PROMPT, TRAFFIC_ANALYSIS_PROMPT, build_accident_prompt

DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_MIN_PIXELS = 128 * 28 * 28
DEFAULT_MAX_PIXELS = 512 * 28 * 28

# Backwards compatibility aliases
MIN_PIXELS = DEFAULT_MIN_PIXELS
MAX_PIXELS = DEFAULT_MAX_PIXELS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="Path to the input video")
    p.add_argument("--prompt-mode", choices=["json", "report"], default="report",
                   help="'json' = structured checklist prompt, 'report' = traffic analysis prompt")
    p.add_argument("--prompt-file", help="Text file with a custom prompt (overrides --prompt-mode)")
    p.add_argument("--n-frames", type=int, default=32, help="Number of sampled frames")
    p.add_argument("--max-new-tokens", type=int, default=700, help="Max generated tokens")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                   help="Model ID (e.g. Qwen/Qwen2-VL-7B-Instruct or Qwen/Qwen2-VL-2B-Instruct)")
    p.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto",
                   help="Torch precision. 'auto' selects bfloat16 on Ampere+ GPUs (A10G/A100/L4), float16 on T4.")
    p.add_argument("--load-4bit", action="store_true",
                   help="Load model in 4-bit (NF4) via bitsandbytes to save VRAM on 16GB GPUs (T4).")
    p.add_argument("--load-8bit", action="store_true",
                   help="Load model in 8-bit via bitsandbytes.")
    p.add_argument("--attn-implementation", choices=["auto", "flash_attention_2", "sdpa", "eager"], default="auto",
                   help="Attention implementation (flash_attention_2 requires flash-attn).")
    p.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS,
                   help="Minimum pixels per frame for vision processor")
    p.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS,
                   help="Maximum pixels per frame for vision processor")
    p.add_argument("--cache-dir", default=None,
                   help="Custom Hugging Face cache directory (recommended for persistent disk on Lightning AI)")
    p.add_argument("-o", "--output", help="Optional file to save the report")
    args = p.parse_args()

    if args.n_frames < 1:
        p.error("--n-frames must be at least 1")
    if args.max_new_tokens < 1:
        p.error("--max-new-tokens must be at least 1")
    return args


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    torch_dtype: str = "auto",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    attn_implementation: str = "auto",
    cache_dir: str | None = None,
):
    processor_kwargs = {
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }
    if cache_dir:
        processor_kwargs["cache_dir"] = cache_dir

    processor = AutoProcessor.from_pretrained(
        model_id, **processor_kwargs
    )

    # Resolve torch dtype
    if torch_dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            resolved_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            resolved_dtype = torch.float16
        else:
            resolved_dtype = torch.float32
    elif torch_dtype == "bfloat16":
        resolved_dtype = torch.bfloat16
    elif torch_dtype == "float16":
        resolved_dtype = torch.float16
    elif torch_dtype == "float32":
        resolved_dtype = torch.float32
    else:
        resolved_dtype = "auto"

    model_kwargs = {
        "device_map": "auto",
    }
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir

    # Attention implementation
    if attn_implementation and attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation
    elif torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

    # Quantization handling
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print("Enabling 4-bit (NF4) quantization via bitsandbytes.")
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print("Enabling 8-bit quantization via bitsandbytes.")
    else:
        model_kwargs["torch_dtype"] = resolved_dtype

    print(f"Loading {model_id} (dtype={model_kwargs.get('torch_dtype', 'quantized')}, device_map='auto')...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, **model_kwargs
    )
    print("Model loaded successfully!")
    if hasattr(model, "hf_device_map"):
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
    )
    device = getattr(model, "device", None) or next(model.parameters()).device
    inputs = inputs.to(device)

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

    model, processor = load_model(
        model_id=args.model_id,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_4bit,
        load_in_8bit=args.load_8bit,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        attn_implementation=args.attn_implementation,
        cache_dir=args.cache_dir,
    )
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

