# sys.path surgery MUST happen before any domain imports so that all submodules
# (vehicle_tracking, VLM, rag, Car_damage_cost) are importable as bare packages.
import sys
from pathlib import Path

_src_dir = str(Path(__file__).parents[2])             # Accident_Intel/
_src_src_dir = str(Path(__file__).parents[2] / "src") # Accident_Intel/src/
_car_damage_dir = str(Path(__file__).parents[2] / "src" / "Car_damage_cost")
for _p in (_src_dir, _src_src_dir, _car_damage_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .models import init_db
from .routers.accidents import router as accidents_router
from .routers.damage import router as damage_router
from .routers.laws import router as laws_router


def _get_settings() -> Settings:
    from .dependencies import get_settings
    return get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = _get_settings()

    # Create required output directories
    for subdir in ("", "accidents", "damage"):
        Path("out", subdir).mkdir(parents=True, exist_ok=True)

    # Initialise the SQLite database
    init_db(str(settings.db_path))

    yield


def create_app() -> FastAPI:
    settings = _get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(accidents_router)
    app.include_router(damage_router)
    app.include_router(laws_router)

    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok", "version": settings.api_version}

    return app


app = create_app()
