"""Singleton model cache — load YOLO + VLM once per backend process."""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

# Make the existing Car_damage_cost package importable.
_SRC = str(Path(__file__).parent.parent / "src" / "Car_damage_cost")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Absolute paths to weights so Config.from_env() resolves them regardless of CWD.
_WEIGHTS_DIR = Path(__file__).parent.parent / "src" / "Car_damage_cost" / "weights"
os.environ.setdefault("DAMAGE_WEIGHTS", str(_WEIGHTS_DIR / "damage_segmentation.pt"))
os.environ.setdefault("PARTS_WEIGHTS", str(_WEIGHTS_DIR / "parts_segmentation.pt"))

# Absolute paths to data files.
_DATA_DIR = Path(__file__).parent.parent / "src" / "Car_damage_cost" / "data"
os.environ.setdefault("PRICES_PATH", str(_DATA_DIR / "prices.json"))
os.environ.setdefault("PRICE_CACHE_PATH", str(_DATA_DIR / "price_cache.json"))

_lock: threading.Lock = threading.Lock()
_cache: dict = {}


def get_models():
    """Return (cfg, damage_model, parts_model, vlm_client), loading once."""
    global _cache
    if _cache:
        return (
            _cache["cfg"],
            _cache["damage_model"],
            _cache["parts_model"],
            _cache["vlm_client"],
        )
    with _lock:
        # Double-checked locking
        if _cache:
            return (
                _cache["cfg"],
                _cache["damage_model"],
                _cache["parts_model"],
                _cache["vlm_client"],
            )

        from config import Config
        from vlm_client import VlmClient
        from yolo_wrapper import YoloSegWrapper

        cfg = Config.from_env()

        # Resolve weight paths — handle relative paths from config
        def _resolve_weight(path_str: str) -> str:
            p = Path(path_str)
            if p.is_absolute():
                return str(p)
            return str(Path(__file__).parent.parent / "src" / "Car_damage_cost" / p)

        damage_model = YoloSegWrapper(
            weights=_resolve_weight(cfg.damage_weights),
            conf=cfg.damage_conf,
            class_conf=cfg.class_conf,
        )
        parts_model = YoloSegWrapper(
            weights=_resolve_weight(cfg.parts_weights),
            conf=cfg.parts_conf,
            iou=cfg.parts_iou,
            imgsz=cfg.parts_imgsz,
        )
        vlm_client = VlmClient(
            base_url=cfg.vlm_base_url,
            model=cfg.vlm_model,
            timeout=cfg.vlm_timeout,
            api=cfg.vlm_api,
            api_key=cfg.vlm_api_key,
            extra=cfg.vlm_extra,
        )

        _cache = {
            "cfg": cfg,
            "damage_model": damage_model,
            "parts_model": parts_model,
            "vlm_client": vlm_client,
        }
        return cfg, damage_model, parts_model, vlm_client


def get_price_agent():
    """Return a PriceAgent instance or None if GEMINI_API_KEY is not set."""
    if not os.getenv("GEMINI_API_KEY"):
        return None

    from config import Config
    from price_agent import PriceAgent

    cfg = Config.from_env()

    cache_path = cfg.price_cache_path
    p = Path(cache_path)
    if not p.is_absolute():
        cache_path = str(Path(__file__).parent.parent / "src" / "Car_damage_cost" / p)

    return PriceAgent(
        api_key=cfg.gemini_api_key,
        model=cfg.gemini_model,
        cache_path=cache_path,
    )
