"""GET /health and GET /config endpoints."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter

# Make existing modules importable
_SRC = str(Path(__file__).parent.parent.parent / "src" / "Car_damage_cost")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

router = APIRouter(tags=["config"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/config")
def get_config():
    from review_logic import DAMAGE_TYPES, PART_CHOICES, SEVERITIES
    from schemas import Tier

    return {
        "part_choices": PART_CHOICES,
        "damage_types": DAMAGE_TYPES,
        "severities": SEVERITIES,
        "tiers": [t.value for t in Tier],
    }
