"""Pydantic schemas for the damage assessment API endpoints.

Provides the request and response models used by the damage router.
"""
from typing import Any

from pydantic import BaseModel, Field


class DamageCreate(BaseModel):
    """Request body for creating a legacy damage assessment record.

    Used by the ``POST /api/damage/legacy`` endpoint. The primary workflow
    uses the ``/upload`` endpoint instead, which accepts images directly.

    Attributes:
        make: Vehicle manufacturer name (e.g. ``"Toyota"``).
        model: Vehicle model name (e.g. ``"Corolla"``).
        year: Vehicle model year.
        tier: Market segment for labour-rate lookup — ``"economy"``,
            ``"mid"``, or ``"luxury"``. Defaults to ``"mid"``.
        currency: Currency code for cost estimates. Defaults to ``"EGP"``.
    """

    make: str
    model: str
    year: int
    tier: str = "mid"
    currency: str = "EGP"


class DamageResponse(BaseModel):
    """Response schema for a damage assessment record (legacy endpoints).

    Attributes:
        id: Unique assessment identifier.
        status: Current status of the assessment (e.g. ``"completed"``).
        vehicle_info: Dict of vehicle details (make, model, year, tier, etc.).
        images: List of image file paths or URLs associated with this
            assessment.
        cost_report: The cost estimate dict produced by the damage service,
            or ``None`` if the assessment has not been completed yet.
        created_at: ISO 8601 timestamp of record creation.
    """

    id: str
    status: str
    vehicle_info: dict = Field(default_factory=dict)
    images: list[str] = Field(default_factory=list)
    cost_report: dict | None = None
    created_at: str = ""
