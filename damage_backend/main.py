"""damage_backend FastAPI application entry point.

Start with:
    uvicorn damage_backend.main:app --port 8001
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from damage_backend.routers import analyze, config_api, cost


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load AI models once on startup."""
    from damage_backend.models_cache import get_models

    get_models()
    yield
    # No explicit cleanup needed


app = FastAPI(
    title="Car Damage Cost Estimator API",
    version="1.0.0",
    description="FastAPI wrapper around the car damage analysis pipeline.",
    lifespan=lifespan,
)

# CORS — allow local dev servers and Lightning AI Studio
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"https://.*\.lightning\.ai",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Include routers — all routes are defined without /api prefix inside the router
# and mounted here with the /api prefix.
app.include_router(config_api.router, prefix="/api")
app.include_router(analyze.router, prefix="/api")
app.include_router(cost.router, prefix="/api")
