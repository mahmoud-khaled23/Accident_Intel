"""FastAPI application for the real upload-to-report pipeline."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

    Registers CORS middleware with exact and regex origin support,
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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_extra_origin_pattern or None,
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
                "video_vlm": backend_info(settings),
                "max_upload_mb": settings.max_upload_mb,
                "rag": {"mode": settings.rag_mode, "rerank": settings.rag_rerank,
                        "corpus_available": all((settings.rag_out_dir / name).is_file()
                                                for name in ("parents.json", "children.jsonl"))},
                "reasoning_configured": bool(settings.openai_api_key.strip() and settings.reasoning_model.strip())}

    return app


app = create_app()
