"""Central configuration. Values come from .env / environment variables."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # --- models ---
    damage_weights: str = "weights/damage_seg.pt"
    parts_weights: str = "weights/parts_segmentation.pt"

    # --- VLM (Ollama) ---
    vlm_base_url: str = "http://localhost:11434"
    vlm_model: str = "qwen3-vl:4b-instruct"
    vlm_timeout: float = 120.0
    vlm_api: str = "ollama"        # "ollama" or "openai" (OpenAI-compatible hosted API)
    vlm_api_key: str = ""
    vlm_extra: dict = field(default_factory=dict)   # e.g. {"reasoning_effort": "none"} for Groq

    # --- cost / pricing ---
    prices_path: str = "data/prices.json"
    price_cache_path: str = "data/price_cache.json"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-lite-latest"   # set GEMINI_MODEL in .env to your exact model id

    # --- detection thresholds ---
    damage_conf: float = 0.25
    parts_conf: float = 0.25
    parts_imgsz: int = 768   # as recommended for the parts model
    parts_iou: float = 0.7
    # per-class override for the damage model (reflective surfaces => noisy)
    class_conf: dict[str, float] = field(
        default_factory=lambda: {"scratch": 0.40, "crack": 0.40}
    )

    # --- mapping ---
    # min fraction of the damage mask that must lie inside a part mask
    overlap_threshold: float = 0.30

    # --- photo checks ---
    blur_threshold: float = 60.0        # Laplacian variance; below => warning
    vlm_quality_check: bool = True      # ask the VLM "is this a car photo?"
    vlm_severity: bool = False          # refine severity with VLM (slower)

    @classmethod
    def from_env(cls) -> "Config":
        c = cls()
        c.damage_weights = os.getenv("DAMAGE_WEIGHTS", c.damage_weights)
        c.parts_weights = os.getenv("PARTS_WEIGHTS", c.parts_weights)
        c.vlm_base_url = os.getenv("VLM_BASE_URL", c.vlm_base_url)
        c.vlm_model = os.getenv("VLM_MODEL", c.vlm_model)
        c.vlm_api = os.getenv("VLM_API", c.vlm_api).lower()
        c.vlm_api_key = os.getenv("VLM_API_KEY", c.vlm_api_key)
        raw = os.getenv("VLM_EXTRA_PARAMS", "").strip()
        if raw:
            try:
                c.vlm_extra = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f'VLM_EXTRA_PARAMS must be valid JSON, e.g. {{"reasoning_effort": "none"}} ({e})')
        c.prices_path = os.getenv("PRICES_PATH", c.prices_path)
        c.gemini_api_key = os.getenv("GEMINI_API_KEY", c.gemini_api_key)
        c.gemini_model = os.getenv("GEMINI_MODEL", c.gemini_model)
        c.overlap_threshold = float(os.getenv("OVERLAP_THRESHOLD", c.overlap_threshold))
        c.vlm_severity = os.getenv("VLM_SEVERITY", "0") == "1"
        c.vlm_quality_check = os.getenv("VLM_QUALITY_CHECK", "1") == "1"
        return c