"""Select local CUDA inference or an OpenAI-compatible vision API."""
from functools import lru_cache
from typing import Literal

VideoBackend = Literal["auto", "local", "openai"]


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, OSError):
        return False


def select_backend(requested: VideoBackend) -> Literal["local", "openai"]:
    if requested == "auto":
        return "local" if cuda_available() else "openai"
    return requested


def configuration_error(settings, backend: str) -> str | None:
    if backend == "local" and not cuda_available():
        return "Local video analysis requires a CUDA GPU. Select API or Auto on a CPU server."
    if backend == "openai":
        if not settings.vlm_model.strip() or not settings.vlm_api_key.strip():
            return "Configure VIDEO_VLM_MODEL and VIDEO_VLM_API_KEY on the server for API video analysis."
    return None
