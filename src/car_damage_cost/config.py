"""Central configuration for the Car Damage Cost module.

Reads all tunable parameters from environment variables (via a `.env` file or
the shell environment) and exposes them as a single :class:`Config` dataclass.

Key classes:
    Config: Dataclass holding paths, thresholds, and API credentials.

Usage::

    cfg = Config.from_env()
    model = YoloSegWrapper(cfg.damage_weights, conf=cfg.damage_conf)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Config:
    """All runtime configuration for the damage-analysis pipeline.

    Default values are applied first; any env-var that is present overrides
    them. Call :meth:`from_env` rather than instantiating directly.

    Attributes:
        damage_weights: Path to the YOLO segmentation weights file for damage
            detection (e.g. ``weights/car_damage_cost/damage_segmentation.pt``).
        parts_weights: Path to the YOLO segmentation weights file for car-part
            detection (e.g. ``weights/car_damage_cost/parts_segmentation.pt``).
        vlm_base_url: Base URL of the VLM server (Ollama or OpenAI-compatible).
        vlm_model: Model name/tag to use for VLM calls.
        vlm_timeout: HTTP timeout in seconds for each VLM request.
        vlm_api: Protocol to use — ``"ollama"`` for a local Ollama server or
            ``"openai"`` for any OpenAI-compatible hosted API.
        vlm_api_key: Bearer token for OpenAI-compatible hosted APIs; empty for
            Ollama (which needs no auth).
        vlm_extra: Extra JSON fields merged into every hosted-API request body,
            e.g. ``{"reasoning_effort": "none"}`` for Groq.
        prices_path: Path to the JSON file containing placeholder part prices
            and labour-rate tables.
        price_cache_path: Path to the on-disk JSON cache used by
            :class:`~price_agent.PriceAgent` to avoid repeated web searches.
        gemini_api_key: Google Gemini API key used by the price agent.
        gemini_model: Gemini model identifier (e.g. ``"gemini-flash-lite-latest"``).
        damage_conf: Global confidence threshold for the damage YOLO model.
        parts_conf: Global confidence threshold for the parts YOLO model.
        parts_imgsz: Input image size (px) for the parts model inference.
        parts_iou: IoU threshold for non-maximum suppression in the parts model.
        class_conf: Per-class confidence overrides for the damage model.
            Defaults to higher thresholds for ``scratch`` and ``crack`` which
            tend to produce many false positives on reflective surfaces.
        overlap_threshold: Minimum fraction of a damage mask that must overlap
            a part mask for the part to be considered a match.
        blur_threshold: Laplacian-variance threshold below which a photo is
            flagged as potentially blurry.
        vlm_quality_check: Whether to ask the VLM to confirm that each photo
            actually shows a car before running segmentation.
        vlm_severity: Whether to ask the VLM to refine the heuristic severity
            estimate (slower; disabled by default).
    """

    # --- models ---
    damage_weights: str = "weights/car_damage_cost/damage_segmentation.pt"
    parts_weights: str = "weights/car_damage_cost/parts_segmentation.pt"

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
        """Build a :class:`Config` by reading environment variables.

        Starts from the dataclass defaults and overrides each field with the
        corresponding environment variable when it is set. Also resolves weight
        and data file paths against several candidate locations so the module
        works whether it is run from the repo root or from within
        ``src/car_damage_cost/``.

        Returns:
            A fully populated :class:`Config` instance.

        Raises:
            ValueError: If ``VLM_EXTRA_PARAMS`` is set but contains invalid JSON.
        """
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

        package_root = Path(__file__).resolve().parent
        for attr in ("damage_weights", "parts_weights", "prices_path", "price_cache_path"):
            value = Path(getattr(c, attr).replace("\\", "/")).expanduser()
            if not value.is_absolute():
                # Candidate locations: the repo root (weights/, data/) and
                # this package's own directory — e.g. the price tables ship
                # at src/car_damage_cost/data/prices.json while weights live
                # at the repo root. Pick the first candidate that exists so
                # the module works from any entry point.
                candidates = (PROJECT_ROOT / value, package_root / value)
                existing = next((p for p in candidates if p.exists()), None)
                value = existing if existing is not None else PROJECT_ROOT / value
            setattr(c, attr, str(value.resolve()))
        return c
