"""FastAPI application factory for the Accident-Intel API.

Sets up ``sys.path`` so that all domain packages (``vehicle_tracking``,
``VLM``, ``rag``, ``reasoning``, ``car_damage_cost``) are importable by their
bare names.  Registers routers, CORS middleware, and the lifespan handler that
initialises the SQLite database.

Key exports:
    app: The ready-to-serve :class:`~fastapi.FastAPI` application instance.
    create_app: Factory function used by the lifespan and routers.
    DynamicCORSMiddleware: Starlette middleware that handles wildcard-subdomain
        CORS origins that the standard ``CORSMiddleware`` cannot match.
"""
# sys.path surgery MUST happen before any domain imports so that video-analysis
# packages (vehicle_tracking, VLM, rag, reasoning) are importable as bare names.
import re
import sys
from pathlib import Path

_src_dir = str(Path(__file__).parents[2])             # Accident_Intel/
_src_src_dir = str(Path(__file__).parents[2] / "src") # Accident_Intel/src/
_vlm_dir = str(Path(__file__).parents[2] / "src" / "VLM")
_rag_dir = str(Path(__file__).parents[2] / "src" / "rag")
_reasoning_dir = str(Path(__file__).parents[2] / "src" / "reasoning")
_car_damage_dir = str(Path(__file__).parents[2] / "src" / "car_damage_cost")
for _p in (_src_dir, _src_src_dir, _vlm_dir, _rag_dir, _reasoning_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _car_damage_dir not in sys.path:
    sys.path.append(_car_damage_dir)

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
    """Return the cached :class:`~config.Settings` singleton.

    Returns:
        The application settings instance.
    """
    from .dependencies import get_settings
    return get_settings()


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    """Handle wildcard-subdomain origins that CORSMiddleware cannot match.

    Matches incoming ``Origin`` headers against a compiled regex pattern.
    When a match is found, the middleware injects the appropriate
    ``Access-Control-Allow-*`` headers and handles pre-flight ``OPTIONS``
    requests directly.

    Attributes:
        _pattern: Compiled regular expression for allowed origin URLs.
    """

    def __init__(self, app, pattern: str) -> None:
        """Initialise the middleware with the given origin pattern.

        Args:
            app: The ASGI application to wrap.
            pattern: A regex string that will be compiled and matched against
                the ``Origin`` request header.
        """
        super().__init__(app)
        self._pattern = re.compile(pattern)

    async def dispatch(self, request: Request, call_next):
        """Process a single request, injecting CORS headers for matching origins.

        Args:
            request: The incoming HTTP request.
            call_next: Callable that forwards the request to the next
                middleware or route handler.

        Returns:
            The HTTP response, potentially with CORS headers added.
        """
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
    """FastAPI lifespan context manager — runs startup and shutdown logic.

    On startup:
    * Creates the required output directory structure under ``settings.data_dir``.
    * Initialises the SQLite database schema.
    * Marks any in-progress analyses as failed (they cannot survive a restart).

    Args:
        app: The FastAPI application instance (provided by the framework).

    Yields:
        Control to the application after startup is complete.
    """
    settings = _get_settings()

    # Create required output directories
    for subdir in ("", "accidents", "damage"):
        (settings.data_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Initialise the SQLite database
    init_db(str(settings.db_path))
    recover_interrupted_analyses(str(settings.db_path))

    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Registers CORS middleware (dynamic wildcard first, then standard),
    includes all routers, and adds the ``/health`` and ``/api/health``
    health-check endpoints.

    Returns:
        A fully configured :class:`~fastapi.FastAPI` application instance.
    """
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
        """Return API health status and pipeline mode (real vs demo).

        Returns:
            A dict with ``status``, ``version``, per-pipeline mode flags, and
            video VLM backend information.
        """
        from .services.vlm_service import backend_info

        return {"status": "ok", "version": settings.api_version,
                "pipeline": {name: "demo" if getattr(settings, f"{name}_mock") else "real"
                             for name in ("tracking", "vlm", "rag", "reasoning")},
                "video_vlm": backend_info(settings)}

    return app


app = create_app()
