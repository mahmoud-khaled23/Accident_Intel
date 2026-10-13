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
import os
from pathlib import Path
import sys

# Prevent Transformers async weight-loading deadlock on cloud/NFS file systems (Lightning AI)
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Support running directly or from repository root
_CURRENT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _CURRENT_DIR.parent
_REPO_ROOT = _SRC_DIR.parent

for _p in [str(_CURRENT_DIR), str(_SRC_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import torch

# Fix PyAV >= 19 compatibility with torchvision.io.read_video:
# PyAV 19+ removed metadata_errors and metadata_encoding, but torchvision still passes them.
try:
    import av
    _orig_av_open = av.open

    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        kwargs.pop("metadata_encoding", None)
        return _orig_av_open(*args, **kwargs)

    av.open = _safe_av_open
except (ImportError, AttributeError):
    pass

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


try:
    from prompts import SYSTEM_PROMPT, TRAFFIC_ANALYSIS_PROMPT, build_accident_prompt
except ImportError:
    from src.VLM.prompts import SYSTEM_PROMPT, TRAFFIC_ANALYSIS_PROMPT, build_accident_prompt

# Optional tracking integration
_TRACKING_IMPORT_ERR = None
try:
    from vehicle_tracking.tracker import process_video
    from vehicle_tracking.accident_candidates import format_tracking_summary, identify_accident_candidates
except Exception as _e:
    try:
        from src.vehicle_tracking.tracker import process_video
        from src.vehicle_tracking.accident_candidates import format_tracking_summary, identify_accident_candidates
    except Exception as _e2:
        process_video = None
        format_tracking_summary = None
        identify_accident_candidates = None
        _TRACKING_IMPORT_ERR = _e2


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

    # Tracking & Detector pre-processing flags
    p.add_argument(
        "--paradigm",
        choices=["without-tracking", "with-tracking", "compare"],
        default="without-tracking",
        help="Analysis paradigm: 'without-tracking' (pure VLM), 'with-tracking' (YOLO+BoT-SORT grounded VLM), or 'compare' (runs both side-by-side).",
    )
    p.add_argument("--with-tracking", action="store_true",
                   help="Shortcut for --paradigm with-tracking.")
    p.add_argument("--use-annotated-video", action="store_true", default=True,
                   help="Feed the tracker's annotated video (with bounding boxes, track IDs, traces) into the VLM in Paradigm 2 (default: True).")
    p.add_argument("--raw-video-only", action="store_true",
                   help="Force Paradigm 2 to use raw unannotated video instead of the annotated video.")
    p.add_argument("--smart-sampling", action="store_true", default=True,
                   help="Concentrate frame sampling around detected candidate events (default: True).")
    p.add_argument("--no-smart-sampling", action="store_true",
                   help="Disable smart event-biased frame sampling and use uniform sampling.")
    p.add_argument("--yolo-model", default="yolo11x.pt",
                   help="YOLO model path or weights for tracking (default: yolo11x.pt).")
    p.add_argument("--repetition-penalty", type=float, default=1.15,
                   help="Repetition penalty for VLM generation (default: 1.15 to prevent repetitive loops).")
    p.add_argument("--min-track-frames", type=int, default=6,
                   help="Minimum number of frames a vehicle track must be active to be included in telemetry (filters out ghost detections). Default: 6.")

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


def get_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or total_frames <= 0:
        raise ValueError(f"Could not read video metadata. Check path: {video_path}")
    return total_frames / fps, fps


def get_video_duration(video_path: str) -> float:
    duration_s, _ = get_video_info(video_path)
    return duration_s


def compute_smart_frame_indices(
    total_frames: int,
    fps: float,
    event_times: list[float] | None = None,
    n_frames: int = 32,
    window_before: float = 2.0,
    window_after: float = 2.5,
) -> list[int]:
    """Compute exactly n_frames indices with event-window concentration."""
    if total_frames <= n_frames:
        return list(range(total_frames))

    if not event_times:
        return [int(round(i * (total_frames - 1) / (n_frames - 1))) for i in range(n_frames)]

    t_event = event_times[0]
    duration_s = total_frames / fps if fps > 0 else 0.0

    w_start = max(0.0, t_event - window_before)
    w_end = min(duration_s, t_event + window_after)

    start_f = max(0, int(w_start * fps))
    end_f = min(total_frames - 1, int(w_end * fps))

    n_event = n_frames // 2
    n_global = n_frames - n_event

    global_indices = set(int(round(i * (total_frames - 1) / (n_global - 1))) for i in range(n_global))
    if end_f > start_f:
        event_indices = set(int(round(start_f + i * (end_f - start_f) / (n_event - 1))) for i in range(n_event))
    else:
        event_indices = set()

    combined = sorted(global_indices | event_indices)

    # If count is less than n_frames, fill in from event window
    if len(combined) < n_frames:
        for f in range(start_f, end_f + 1):
            if f not in combined:
                combined.append(f)
            if len(combined) == n_frames:
                break

    # If still less, fill from all frames
    if len(combined) < n_frames:
        for f in range(total_frames):
            if f not in combined:
                combined.append(f)
            if len(combined) == n_frames:
                break

    combined.sort()

    # If greater, take evenly spaced subset of size n_frames
    if len(combined) > n_frames:
        indices = [combined[int(round(i * (len(combined) - 1) / (n_frames - 1)))] for i in range(n_frames)]
        return sorted(list(dict.fromkeys(indices)))

    return combined


def create_event_biased_clip(
    input_video: str,
    output_clip: str,
    event_times: list[float] | None = None,
    n_frames: int = 32,
) -> str:
    """Extract exactly n_frames with event-window concentration and save as a compact MP4 clip."""
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        return input_video

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= n_frames or width <= 0 or height <= 0:
        cap.release()
        return input_video

    indices = compute_smart_frame_indices(total_frames, fps, event_times, n_frames=n_frames)
    target_set = set(indices)
    max_target = max(indices)

    Path(output_clip).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_clip, cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (width, height))
    if not out.isOpened():
        cap.release()
        return input_video

    curr_f = 0
    while curr_f <= max_target:
        ret, frame = cap.read()
        if not ret:
            break
        if curr_f in target_set:
            out.write(frame)
        curr_f += 1

    cap.release()
    out.release()
    return output_clip


def build_prompt(args, duration_s: float, tracking_summary: str | None = None) -> str:
    if args.prompt_file:
        base_prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    elif args.prompt_mode == "json":
        base_prompt = build_accident_prompt(args.n_frames, duration_s)
    else:
        base_prompt = TRAFFIC_ANALYSIS_PROMPT

    if tracking_summary:
        telemetry_block = (
            "======================================================================\n"
            "GROUND TRUTH TELEMETRY FROM YOLO DETECTOR & BoT-SORT TRACKER:\n"
            f"{tracking_summary}\n"
            "======================================================================\n"
            "MANDATORY FORENSIC GROUNDING INSTRUCTIONS:\n"
            "1. VEHICLE ID BINDING (STRICT): Whenever referring to any road user anywhere in your report, you MUST "
            "explicitly cite its Track ID in parentheses alongside its visual description (e.g. 'the white van (Vehicle ID 2)', "
            "'the leading black car (Vehicle ID 1)', 'the blue truck (Vehicle ID 9)'). Never refer to a tracked vehicle without its Vehicle ID.\n"
            "2. OBJECT PERMANENCE: The Track IDs above track the SAME physical entities over time. Do NOT invent phantom vehicles "
            "or claim an overturned vehicle is a separate car from the one tracked in that lane.\n"
            "3. TIMESTAMP ANCHORING: In 'Chronological Events', structure each event phase with timestamp brackets grounded "
            "in the telemetry and clip duration (e.g. '[0.00s - 0.83s]: ...', '[0.83s - 1.80s]: ...').\n"
            "4. KINEMATIC SYNCHRONIZATION: Cross-reference your deceleration and interaction descriptions with the measured speed drops "
            "and collision alerts flagged in the telemetry above.\n"
            "5. VERDICT SUMMARY: Complete the required Verdict Summary section at the end of the report, specifying the involved Vehicle IDs.\n"
            "6. NO REPETITIVE LOOPS: State observations clearly and concisely without looping or repeating sentences.\n\n"
        )
        return telemetry_block + base_prompt
    return base_prompt


def analyze(
    model,
    processor,
    video_path: str,
    prompt: str,
    n_frames: int,
    max_new_tokens: int,
    repetition_penalty: float = 1.15,
) -> str:
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

    proc_kwargs = {
        "text": [text_prompt],
        "images": image_inputs,
        "videos": video_inputs,
        "padding": True,
        "return_tensors": "pt",
    }
    if video_inputs is not None:
        try:
            proc_kwargs["cap_pixels_per_frame"] = True
        except Exception:
            pass

    inputs = processor(**proc_kwargs)
    device = getattr(model, "device", None) or next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def run_paradigm_without_tracking(model, processor, video_path: str, args, duration_s: float) -> str:
    """Paradigm 1: Direct, ungrounded VLM analysis (relies purely on visual model)."""
    prompt = build_prompt(args, duration_s, tracking_summary=None)
    return analyze(
        model,
        processor,
        video_path,
        prompt,
        args.n_frames,
        args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )


def run_paradigm_with_tracking(
    model, processor, video_path: str, args, duration_s: float, fps: float
):
    """Paradigm 2: Neuro-symbolic grounded VLM analysis (YOLO detector + BoT-SORT tracking)."""
    trajectories = None
    tracked_video_p = None

    if args.trajectories:
        print(f"[Tracker] Loading precomputed trajectories from {args.trajectories}...")
        import json
        with open(args.trajectories, "r", encoding="utf-8") as f:
            trajectories = json.load(f)
    else:
        if process_video is None:
            raise ImportError(
                f"Vehicle tracking module could not be loaded ({_TRACKING_IMPORT_ERR}). "
                "Run: pip install ultralytics supervision"
            )
        video_p = Path(video_path)
        out_tracked_video = video_p.parent / f"{video_p.stem}_tracked.mp4"
        out_trajectories = video_p.parent / f"{video_p.stem}_trajectories.json"

        print(f"[Tracker] Running YOLO ({args.yolo_model}) + BoT-SORT on {video_path}...")
        trajectories = process_video(
            input_path=str(video_p),
            output_video=str(out_tracked_video),
            output_json=str(out_trajectories),
            model_path=args.yolo_model,
            min_track_frames=args.min_track_frames,
        )
        print(f"[Tracker] Annotated video saved to: {out_tracked_video}")
        print(f"[Tracker] Trajectories saved to: {out_trajectories}")
        tracked_video_p = str(out_tracked_video)

    candidates = (
        identify_accident_candidates(trajectories, min_track_frames=args.min_track_frames)
        if identify_accident_candidates
        else None
    )
    tracking_summary = (
        format_tracking_summary(
            trajectories,
            fps=fps,
            candidates=candidates,
            min_track_frames=args.min_track_frames,
        )
        if format_tracking_summary
        else None
    )

    if tracking_summary:
        print("-" * 70)
        print("Ground Truth Telemetry from Tracker:")
        print(tracking_summary)
        print("-" * 70)

    use_annotated = (not getattr(args, "raw_video_only", False)) and bool(tracked_video_p)
    video_for_vlm = tracked_video_p if use_annotated else video_path
    if use_annotated:
        print("[Tracker] Using annotated video with bounding boxes & track IDs for VLM visual input.")

    # Smart event-biased temporal zooming
    is_smart_sampling = getattr(args, "smart_sampling", True) and not getattr(args, "no_smart_sampling", False)
    if is_smart_sampling and candidates:
        event_times = []
        for cand in candidates:
            if cand.get("overlap_frame") is not None and fps > 0:
                event_times.append(cand["overlap_frame"] / fps)
            for d in cand.get("deceleration", {}).values():
                if d.get("detected") and d.get("frame") is not None and fps > 0:
                    event_times.append(d["frame"] / fps)

        if event_times:
            video_p = Path(video_path)
            smart_clip = str(video_p.parent / f"{video_p.stem}_smart_{args.n_frames}f.mp4")
            print(f"[Sampling] Generating smart event-biased clip concentrated around {event_times[0]:.2f}s...")
            video_for_vlm = create_event_biased_clip(
                video_for_vlm,
                smart_clip,
                event_times=event_times,
                n_frames=args.n_frames,
            )
            print(f"[Sampling] Smart event-concentrated clip ready: {smart_clip}")

    prompt = build_prompt(args, duration_s, tracking_summary=tracking_summary)

    report = analyze(
        model,
        processor,
        video_for_vlm,
        prompt,
        args.n_frames,
        args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    return report, tracking_summary


def main():
    args = parse_args()
    if args.with_tracking and args.paradigm == "without-tracking":
        args.paradigm = "with-tracking"

    duration_s, fps = get_video_info(args.video)
    video_stem = Path(args.video).stem

    print(f"Video duration: {duration_s:.2f}s | FPS: {fps:.2f} | Frames sampled: {args.n_frames}")
    print(f"Selected Paradigm: {args.paradigm.upper()}")

    # Load VLM model once
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

    # ── Paradigm Execution ──────────────────────────────────────────────────
    if args.paradigm == "without-tracking":
        print("\n" + "=" * 70)
        print("PARADIGM 1: PURE VLM (WITHOUT TRACKING)")
        print("=" * 70)
        report = run_paradigm_without_tracking(model, processor, args.video, args, duration_s)
        print(report)
        print("=" * 70)

        out_file = args.output or f"{video_stem}_vlm_without_tracking.txt"
        Path(out_file).write_text(report, encoding="utf-8")
        print(f"Report saved to: {out_file}")

    elif args.paradigm == "with-tracking":
        print("\n" + "=" * 70)
        print("PARADIGM 2: GROUNDED VLM (WITH YOLO + BoT-SORT TRACKING)")
        print("=" * 70)
        report, summary = run_paradigm_with_tracking(model, processor, args.video, args, duration_s, fps)
        print(report)
        print("=" * 70)

        out_file = args.output or f"{video_stem}_vlm_with_tracking.txt"
        Path(out_file).write_text(report, encoding="utf-8")
        print(f"Report saved to: {out_file}")

    elif args.paradigm == "compare":
        print("\n" + "=" * 70)
        print("RUNNING DUAL-PARADIGM BENCHMARK ON SAME FOOTAGE")
        print("=" * 70)

        print("\n>>> [1/2] Executing Paradigm 1: Pure VLM (Without Tracking)...")
        report_pure = run_paradigm_without_tracking(model, processor, args.video, args, duration_s)

        print("\n>>> [2/2] Executing Paradigm 2: Grounded VLM (With Detector & Tracker)...")
        report_tracked, summary = run_paradigm_with_tracking(model, processor, args.video, args, duration_s, fps)

        print("\n" + "=" * 70)
        print("COMPARISON: PARADIGM 1 (WITHOUT TRACKING) vs PARADIGM 2 (WITH TRACKING)")
        print("=" * 70)

        print("\n--- [PARADIGM 1: PURE VLM] ---")
        print(report_pure)

        print("\n--- [PARADIGM 2: TRACKING-GROUNDED VLM] ---")
        print(report_tracked)
        print("=" * 70)

        out_pure = f"{video_stem}_without_tracking.txt"
        out_tracked = f"{video_stem}_with_tracking.txt"
        Path(out_pure).write_text(report_pure, encoding="utf-8")
        Path(out_tracked).write_text(report_tracked, encoding="utf-8")
        print(f"[Saved] Paradigm 1 Report -> {out_pure}")
        print(f"[Saved] Paradigm 2 Report -> {out_tracked}")


if __name__ == "__main__":
    main()



