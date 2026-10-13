from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict as SettingsConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    api_title: str = "Accident-Intel API"
    api_version: str = "0.1.0"
    data_dir: Path = PROJECT_ROOT / "out"
    db_path: Path = PROJECT_ROOT / "out/accident_intel.db"
    openai_api_key: str = ""
    openai_base_url: str = ""
    reasoning_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    vlm_api: Literal["local", "openai"] = "local"
    vlm_base_url: str = ""
    vlm_model: str = ""
    vlm_api_key: str = ""
    vlm_model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    vlm_n_frames: int = Field(default=16, ge=1, le=128)
    vlm_max_images: int = Field(default=3, ge=1, le=32)
    vlm_max_tokens: int = Field(default=1400, ge=1)
    vlm_extra_params: dict = Field(default_factory=dict)
    vlm_mock: bool = False
    tracking_mock: bool = False
    tracking_model_path: str = "yolo11x.pt"
    reasoning_mock: bool = False
    rag_mock: bool = False
    damage_mock: bool = False
    rag_out_dir: Path = PROJECT_ROOT / "out"
    rag_mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    rag_rerank: bool = True
    max_upload_mb: int = Field(default=500, ge=1)
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:4173"]
    cors_extra_origin_pattern: str = r"https://.*\.(?:cloudspaces\.litng\.ai|litng\.ai)$"
    model_config = SettingsConfig(env_file=PROJECT_ROOT / ".env", extra="ignore")
