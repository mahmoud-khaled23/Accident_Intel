from pathlib import Path
import dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict as SettingsConfig

dotenv.load_dotenv()


class Settings(BaseSettings):
    api_title: str = "Accident-Intel API"
    api_version: str = "0.1.0"
    data_dir: Path = Path("out")
    db_path: Path = Path("out/accident_intel.db")
    openai_api_key: str = ""
    reasoning_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    vlm_api: str = "openai"
    vlm_base_url: str = ""
    vlm_model: str = ""
    vlm_api_key: str = ""
    vlm_model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    vlm_n_frames: int = 32
    vlm_max_tokens: int = 700
    vlm_mock: bool = True
    tracking_mock: bool = True
    reasoning_mock: bool = True
    rag_mock: bool = False
    damage_mock: bool = True
    rag_out_dir: Path = Path("out")
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:4173"]
    cors_extra_origin_pattern: str = r"https://.*\.(?:cloudspaces\.litng\.ai|litng\.ai)$"
    model_config = SettingsConfig(env_file=".env", extra="ignore")
