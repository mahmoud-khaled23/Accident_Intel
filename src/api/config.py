"""API configuration settings loaded from environment variables and a .env file.

Uses Pydantic ``BaseSettings`` so every field can be overridden by setting the
corresponding environment variable.  The ``.env`` file at the project root is
read automatically.

Key classes:
    Settings: All runtime configuration for the FastAPI application.
"""
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict as SettingsConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""Absolute path to the repository root (two levels above this file)."""


class Settings(BaseSettings):
    """Runtime configuration for the Accident-Intel API.

    All fields can be overridden via environment variables of the same name
    (case-insensitive) or via the ``.env`` file at the project root.

    Attributes:
        api_title: Human-readable title shown in the OpenAPI docs.
        api_version: API semantic version string.
        data_dir: Root directory for all runtime data (uploads, database).
        db_path: Path to the SQLite database file.
        openai_api_key: API key for the OpenAI-compatible reasoning endpoint.
        openai_base_url: Base URL for the OpenAI-compatible reasoning endpoint.
            Empty string means the default OpenAI URL is used.
        reasoning_model: Model name passed to the reasoning service
            (e.g. ``"gpt-4o-mini"``).
        gemini_api_key: Google Gemini API key used by the price agent.
        gemini_model: Gemini model identifier (e.g. ``"gemini-1.5-flash"``).
        video_vlm_api: Selects the video VLM backend. ``"local"`` uses the
            local Qwen2-VL model; ``"openai"`` uses an OpenAI-compatible
            hosted vision API.
        vlm_base_url: Base URL for the hosted OpenAI-compatible VLM endpoint.
            Empty when using the local model.
        vlm_model: Model name for the hosted OpenAI-compatible vision endpoint.
        vlm_api_key: Bearer token for the hosted vision API.
        vlm_model_id: Hugging Face model ID for the local VLM
            (e.g. ``"Qwen/Qwen2-VL-7B-Instruct"``).
        vlm_n_frames: Number of video frames extracted and sent to the VLM.
        vlm_max_images: Maximum number of images per VLM request (hosted API).
        vlm_max_tokens: Maximum token budget for each VLM response.
        vlm_extra_params: Additional JSON fields merged into hosted VLM
            requests (e.g. ``{"reasoning_effort": "none"}``).
        vlm_mock: When ``True``, the VLM service returns a canned mock
            description instead of running inference.
        tracking_mock: When ``True``, the tracking service returns mock
            trajectory data.
        tracking_model_path: Path or name of the YOLO model used for vehicle
            tracking (e.g. ``"yolo11x.pt"``).
        reasoning_mock: When ``True``, the reasoning service returns a canned
            mock liability report.
        rag_mock: When ``True``, the law-retrieval service returns a small set
            of hard-coded mock laws.
        damage_mock: When ``True``, the damage service returns mock cost data.
        rag_out_dir: Directory containing the chunked traffic-law corpus and
            the Chroma vector index used by the RAG retriever.
        rag_mode: Retrieval mode passed to the RAG backend — ``"hybrid"``,
            ``"dense"``, or ``"bm25"``.
        rag_rerank: Whether to apply a cross-encoder reranker after
            retrieval.
        max_upload_mb: Maximum allowed video upload size in megabytes.
        cors_origins: List of allowed CORS origins for the standard
            ``CORSMiddleware``.
        cors_extra_origin_pattern: Regex pattern for additional CORS origins
            (supports wildcard subdomains not matchable by the standard list).
    """

    api_title: str = "Accident-Intel API"
    api_version: str = "0.1.0"
    data_dir: Path = PROJECT_ROOT / "data"
    db_path: Path = PROJECT_ROOT / "data/accident_intel.db"
    openai_api_key: str = ""
    openai_base_url: str = ""
    reasoning_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    # Video selection is independent of VLM_API in the repair-cost configuration.
    # The upload pipeline uses src/video_understanding/analyze_video.py unless explicitly overridden.
    video_vlm_api: Literal["local", "openai"] = "local"
    vlm_base_url: str = ""
    vlm_model: str = ""
    vlm_api_key: str = ""
    vlm_model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    # Conservative hosted defaults fit providers with a 7k input-token limit.
    # They can be increased after confirming the provider's vision quota.
    vlm_n_frames: int = Field(default=4, ge=1, le=128)
    vlm_max_images: int = Field(default=1, ge=1, le=32)
    vlm_max_tokens: int = Field(default=700, ge=1)
    vlm_extra_params: dict = Field(default_factory=dict)
    vlm_mock: bool = False
    tracking_mock: bool = False
    tracking_model_path: str = "yolo11x.pt"
    reasoning_mock: bool = False
    rag_mock: bool = False
    damage_mock: bool = False
    # The checked-in law corpus and its Chroma index live together here.
    # Runtime uploads and SQLite state live under `out/` and must not be used
    # as a retrieval index unless an index has explicitly been built there.
    rag_out_dir: Path = PROJECT_ROOT / "data/traffic_laws/chunks"
    rag_mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    rag_rerank: bool = True
    max_upload_mb: int = Field(default=500, ge=1)
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:4173"]
    cors_extra_origin_pattern: str = r"https://.*\.(?:cloudspaces\.litng\.ai|litng\.ai)$"
    model_config = SettingsConfig(env_file=PROJECT_ROOT / ".env", extra="ignore")
