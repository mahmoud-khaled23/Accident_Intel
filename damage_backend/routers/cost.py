"""POST /cost-estimate endpoint."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Make existing modules importable
_SRC = str(Path(__file__).parent.parent.parent / "src" / "Car_damage_cost")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

router = APIRouter(tags=["cost"])


class CostRequest(BaseModel):
    analysis_id: str
    rows: list[dict[str, Any]]
    make: str
    model_name: str
    year: int
    tier: str
    use_web: bool = False


@router.post("/cost-estimate")
def cost_estimate(request: CostRequest):
    try:
        import os

        from config import Config
        from cost_engine import estimate_cost, load_tables
        from review_logic import rows_to_fused
        from schemas import CarInfo, Tier

        from damage_backend.models_cache import get_price_agent
        from damage_backend.serializers import make_json_safe

        cfg = Config.from_env()

        fused_list, row_warnings = rows_to_fused(request.rows)

        try:
            tier_enum = Tier(request.tier)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid tier '{request.tier}'. Choices: economy, mid, luxury.",
            )

        car = CarInfo(
            make=request.make,
            model=request.model_name,
            year=request.year,
            tier=tier_enum,
        )

        # Resolve prices_path to absolute
        prices_path = cfg.prices_path
        p = Path(prices_path)
        if not p.is_absolute():
            prices_path = str(
                Path(__file__).parent.parent.parent / "src" / "Car_damage_cost" / p
            )

        tables = load_tables(prices_path)

        agent = None
        if request.use_web and os.getenv("GEMINI_API_KEY"):
            agent = get_price_agent()

        cost_report = estimate_cost(fused_list, car, tables, agent)

        response = {
            "cost": make_json_safe(cost_report.model_dump()),
            "warnings": row_warnings,
        }

        return JSONResponse(content=make_json_safe(response))

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
