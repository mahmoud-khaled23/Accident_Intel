# sys.path surgery MUST happen before any domain imports so that all submodules
# (vehicle_tracking, VLM, rag, Car_damage_cost) are importable as bare packages.
import re
import sys
from pathlib import Path

_src_dir = str(Path(__file__).parents[2])             # Accident_Intel/
_src_src_dir = str(Path(__file__).parents[2] / "src") # Accident_Intel/src/
_car_damage_dir = str(Path(__file__).parents[2] / "src" / "Car_damage_cost")
_vlm_dir = str(Path(__file__).parents[2] / "src" / "VLM")
_rag_dir = str(Path(__file__).parents[2] / "src" / "rag")
for _p in (_src_dir, _src_src_dir, _car_damage_dir, _vlm_dir, _rag_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .config import Settings
from .models import init_db, recover_interrupted_analyses
from .routers.accidents import router as accidents_router
from .routers.damage import router as damage_router
from .routers.laws import router as laws_router


def _get_settings() -> Settings:
    from .dependencies import get_settings
    return get_settings()


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    """Handle wildcard-subdomain origins that CORSMiddleware cannot match."""

    def __init__(self, app, pattern: str) -> None:
        super().__init__(app)
        self._pattern = re.compile(pattern)

    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin", "")
        if origin and self._pattern.match(origin):
            # Preflight
            if request.method == "OPTIONS":
                response = Response(status_code=204)
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Allow-Methods"] = "*"
                response.headers["Access-Control-Allow-Headers"] = "*"
                return response
            response = await call_next(request)
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            return response
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = _get_settings()

    # Create required output directories
    for subdir in ("", "accidents", "damage"):
        (settings.data_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Initialise the SQLite database
    init_db(str(settings.db_path))
    recover_interrupted_analyses(str(settings.db_path))

    yield


def create_app() -> FastAPI:
    settings = _get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        lifespan=lifespan,
    )

    # Dynamic CORS for wildcard subdomains (must be added before CORSMiddleware)
    app.add_middleware(
        DynamicCORSMiddleware,
        pattern=settings.cors_extra_origin_pattern,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(accidents_router)
    app.include_router(damage_router, prefix="/api/damage")
    app.include_router(damage_router, prefix="/api/damage-assessments")
    app.include_router(laws_router)

    @app.get("/health", tags=["health"])
    @app.get("/api/health", tags=["health"])
    def health():
        return {"status": "ok", "version": settings.api_version,
                "pipeline": {name: "demo" if getattr(settings, f"{name}_mock") else "real"
                             for name in ("tracking", "vlm", "rag", "reasoning")}}

    return app


app = create_app()
