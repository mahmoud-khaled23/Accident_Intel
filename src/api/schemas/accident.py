from typing import Any, Literal

from pydantic import BaseModel, Field


class AccidentCreate(BaseModel):
    location: str
    date_time: str
    description: str = ""
    notes: str = ""


class AccidentResponse(BaseModel):
    id: str
    status: str
    location: str = ""
    date_time: str = ""
    description: str = ""
    notes: str = ""
    created_at: str = ""
    video_filename: str = ""
    tracking_data: dict | None = None
    vlm_description: str | None = None
    laws: list[Any] | None = None
    report: dict | None = None
    error: str | None = None
    failed_stage: str | None = None
    is_demo_data: bool = False
    video_metadata: dict | None = None


class AnalysisStatus(BaseModel):
    status: str
    message: str


class ReportVehicle(BaseModel):
    id: str | int
    role: str = "insufficient_evidence"
    violations: list[str] = Field(default_factory=list)


class ReportViolation(BaseModel):
    law_section: str = ""
    description: str
    severity: Literal["minor", "moderate", "major"] = "minor"


class ReasoningReport(BaseModel):
    overall_assessment: str = Field(min_length=1)
    overall_level: Literal["potential_responsibility", "shared_responsibility", "insufficient_evidence"] = "insufficient_evidence"
    vehicles: list[ReportVehicle] = Field(default_factory=list)
    violations: list[ReportViolation] = Field(default_factory=list)
    confidence: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    limitations: list[str] = Field(default_factory=list)
