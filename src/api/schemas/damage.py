from typing import Any

from pydantic import BaseModel


class DamageCreate(BaseModel):
    make: str
    model: str
    year: int
    tier: str = "mid"
    currency: str = "EGP"


class DamageResponse(BaseModel):
    id: str
    status: str
    vehicle_info: dict = {}
    images: list[str] = []
    cost_report: dict | None = None
    created_at: str = ""
