#!/usr/bin/env python3
"""Analyze a CCTV video with Qwen2-VL and print a traffic-incident report.

Examples:
    uv run python -m src.video_understanding.analyze_video --video clip_07.mp4
    uv run python -m src.video_understanding.analyze_video --video clip_07.mp4 --prompt-mode json --n-frames 16
    uv run python -m src.video_understanding.analyze_video --video clip_07.mp4 --prompt-file my_prompt.txt -o report.txt
"""
import argparse
from pathlib import Path

import cv2
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

if __package__:
    from .prompts import SYSTEM_PROMPT, TRAFFIC_ANALYSIS_PROMPT, build_accident_prompt
else:  # Direct script execution; prefer uv run python -m src.video_understanding.analyze_video.
    from prompts import SYSTEM_PROMPT, TRAFFIC_ANALYSIS_PROMPT, build_accident_prompt

DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28


def parse_args():
    """Parse and validate command-line arguments for the VLM analyzer.

    Builds an :class:`argparse.ArgumentParser` with options for the input
    video, prompt selection, sampling parameters, model choice, and output
    file.  Exits with an error message if ``--n-frames`` or
    ``--max-new-tokens`` are less than 1.

    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            - ``video`` (str): Path to the input video.
            - ``prompt_mode`` (str): ``"json"`` or ``"report"``.
            - ``prompt_file`` (str | None): Path to a custom prompt file.
            - ``n_frames`` (int): Number of frames to sample.
            - ``max_new_tokens`` (int): Generation budget.
            - ``model_id`` (str): HuggingFace model identifier.
            - ``output`` (str | None): Optional path to save the report.
    """
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
    """Download (if necessary) and load the Qwen2-VL model and its processor.

    Requires CUDA and loads ``Qwen2VLForConditionalGeneration`` in half-precision (``float16``). ``device_map``
    is set to ``"auto"`` so multi-GPU and CPU-offload are handled automatically
    by the ``accelerate`` library.

    Args:
        model_id: HuggingFace model repository identifier, e.g.
            ``"Qwen/Qwen2-VL-7B-Instruct"``.

    Returns:
        tuple[Qwen2VLForConditionalGeneration, AutoProcessor]: The loaded model
        and its matching processor, ready for inference.

    Example:
        >>> model, processor = load_model("Qwen/Qwen2-VL-7B-Instruct")
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Local Qwen2-VL requires a CUDA GPU. Use the hosted API backend on CPU.")
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    print("Model loaded successfully!")
    print("Device mapping:", model.hf_device_map)
    return model, processor


def get_video_duration(video_path: str) -> float:
    """Return the total duration of a video in seconds.

    Opens the video with OpenCV, reads its FPS and total frame count, then
    computes ``total_frames / fps``.

    Args:
        video_path: Path to the video file.

    Returns:
        float: Duration in seconds.

    Raises:
        ValueError: If the video cannot be opened or its metadata (FPS or
            frame count) is invalid (≤ 0).

    Example:
        >>> get_video_duration("clip.mp4")
        12.5
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or total_frames <= 0:
        raise ValueError(f"Could not read video metadata. Check path: {video_path}")
    return total_frames / fps


def build_prompt(args, duration_s: float) -> str:
    """Select or construct the user-turn prompt for the VLM.

    Priority order:
    1. If ``args.prompt_file`` is set, read and return the file's text verbatim.
    2. If ``args.prompt_mode`` is ``"json"``, return a structured accident-report
       prompt rendered for the actual frame count and duration via
       :func:`~prompts.build_accident_prompt`.
    3. Otherwise return :data:`~prompts.TRAFFIC_ANALYSIS_PROMPT`.

    Args:
        args: Parsed CLI arguments (as returned by :func:`parse_args`).
        duration_s: Video duration in seconds, used when building the
            structured prompt.

    Returns:
        str: The prompt string to send as the user turn.
    """
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt_mode == "json":
        return build_accident_prompt(args.n_frames, duration_s)
    return TRAFFIC_ANALYSIS_PROMPT


def analyze(model, processor, video_path: str, prompt: str, n_frames: int, max_new_tokens: int, model_id: str = DEFAULT_MODEL_ID) -> str:
    """Run the Qwen2-VL model on a video and return the generated text report.

    Clamps ``n_frames`` to the actual frame count and rounds it down to an
    even number (required by Qwen's video sampler).  Builds a two-turn
    message list with the system prompt, the video, and the user prompt; then
    runs greedy decoding with ``do_sample=False``.

    Args:
        model: A loaded ``Qwen2VLForConditionalGeneration`` instance.
        processor: The matching ``AutoProcessor`` instance.
        video_path: Path to the input video file.
        prompt: The user-turn prompt text (from :func:`build_prompt`).
        n_frames: Requested number of frames to sample; automatically clamped
            to the valid range ``[2, frame_count]`` and rounded to even.
        max_new_tokens: Maximum number of tokens the model may generate.

    Returns:
        str: The decoded text of the model's response with special tokens
        stripped.

    Raises:
        ValueError: If the video has fewer than two frames, which is the
            minimum required by Qwen's video encoder.

    Example:
        >>> report = analyze(model, processor, "clip.mp4", prompt, 32, 700)
    """
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
    try:
        from src.api.model_cost.usage import record_usage  # noqa: PLC0415
        prompt_tok = inputs.input_ids.shape[1]
        completion_tok = len(trimmed[0])
        record_usage(
            model_id=model_id,
            stage="vlm_local",
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
        )
    except Exception as exc:
        import logging  # noqa: PLC0415
        logging.getLogger(__name__).warning(
            "model_cost: instrumentation skipped: %s", exc
        )
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def main():
    """Entry point for the VLM video analyzer CLI.

    Parses arguments, measures the video duration, selects the appropriate
    prompt, loads the model, runs inference, prints the report, and
    optionally saves it to a file.

    Returns:
        None
    """
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
