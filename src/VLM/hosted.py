"""OpenAI-compatible hosted vision analysis of sampled video frames."""
from __future__ import annotations

import base64
import io
import math


def analyze_hosted(
    video_path: str,
    prompt: str,
    *,
    model: str,
    api_key: str,
    base_url: str | None = None,
    n_frames: int = 16,
    max_images: int = 3,
    max_tokens: int = 1400,
    extra_params: dict | None = None,
) -> str:
    """Sample frames, group them into timestamped contact sheets, and call a vision chat model."""
    import openai

    try:
        from contact_sheet import make_sheet, sample_frames
        from prompts import SYSTEM_PROMPT
    except ImportError:
        from .contact_sheet import make_sheet, sample_frames
        from .prompts import SYSTEM_PROMPT

    if not model:
        raise RuntimeError("Set VLM_MODEL to a vision-capable model for VLM_API=openai.")
    if not api_key:
        raise RuntimeError("Set VLM_API_KEY for the hosted visual model.")

    frames = sample_frames(video_path, n_frames)
    if not frames:
        raise RuntimeError("No frames could be extracted from this video.")

    content = [
        {
            "type": "text",
            "text": (
                prompt
                + "\nImages may be contact sheets: read panels left-to-right, top-to-bottom. "
                "Each panel is a different time, labelled in seconds; they are not simultaneous objects."
            ),
        }
    ]
    group_size = math.ceil(len(frames) / max_images)
    for offset in range(0, len(frames), group_size):
        group = frames[offset : offset + group_size]
        if len(group) == 1:
            ts, picture = group[0]
            picture.thumbnail((768, 768))
            caption = f"Frame at {ts:.3f} seconds"
        else:
            picture = make_sheet(group, cols=math.ceil(math.sqrt(len(group))), thumb_w=384, thumb_h=216)
            caption = "Video frames at " + ", ".join(f"{ts:.3f} seconds" for ts, _ in group)
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.append({"type": "text", "text": caption})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})

    with openai.OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=180.0,
        max_retries=1,
    ) as client:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=max_tokens,
            extra_body=extra_params or None,
        )
    return response.choices[0].message.content or ""
