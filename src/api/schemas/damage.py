from typing import Any

from pydantic import BaseModel, Field


class DamageCreate(BaseModel):
    make: str
    model: str
    year: int
    tier: str = "mid"
    currency: str = "EGP"


class DamageResponse(BaseModel):
    id: str
    status: str
    vehicle_info: dict = Field(default_factory=dict)
    images: list[str] = Field(default_factory=list)
    cost_report: dict | None = None
    created_at: str = ""
