"""Pydantic schemas for the accident analysis API endpoints.

Defines the request and response models used by the accidents router,
including the structured liability-reasoning report produced by the
reasoning service.
"""
from typing import Any, Literal

from pydantic import BaseModel, Field


class AccidentCreate(BaseModel):
    """Request body for creating a new accident record.

    Attributes:
        location: Human-readable location of the accident
            (e.g. ``"Main St and 5th Ave, Cairo"``).
        date_time: ISO 8601 date-time string for when the accident occurred.
        description: Optional free-text description of the incident.
        notes: Optional additional notes or context.
    """

    location: str
    date_time: str
    description: str = ""
    notes: str = ""


class AccidentResponse(BaseModel):
    """Response schema for a single accident record.

    Combines the stored accident metadata with the results of each pipeline
    stage (tracking, VLM description, retrieved laws, and reasoning report).

    Attributes:
        id: Unique accident identifier (UUID string).
        status: Current lifecycle status (see :class:`~common.StatusEnum`).
        location: Location string provided at creation.
        date_time: Date-time string provided at creation.
        description: Description provided at creation.
        notes: Notes provided at creation.
        created_at: ISO 8601 timestamp of record creation.
        video_filename: Original filename of the uploaded video.
        tracking_data: Trajectory data produced by the tracking service,
            or ``None`` if the step has not run yet.
        vlm_description: Scene description produced by the VLM service,
            or ``None`` if the step has not run yet.
        laws: Retrieved traffic law entries, or ``None`` if not yet retrieved.
        report: Liability reasoning report dict, or ``None`` if not yet produced.
        error: Human-readable error message if the pipeline failed.
        failed_stage: Name of the pipeline stage that failed, if any.
        is_demo_data: ``True`` when the analysis used mock (demo) services.
        video_metadata: Video technical metadata (duration, fps, dimensions),
            or ``None`` if no video has been uploaded.
    """

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
    """Response body for the trigger-analysis endpoint.

    Attributes:
        status: Short status keyword (e.g. ``"processing"``).
        message: Human-readable message describing what was started.
    """

    status: str
    message: str


class ReportVehicle(BaseModel):
    """Per-vehicle entry in the reasoning report.

    Attributes:
        id: Vehicle identifier matching the tracking track ID.
        role: Assigned role in the incident (e.g. ``"primary_at_fault"``,
            ``"secondary"``, ``"insufficient_evidence"``).
        violations: List of traffic violation code strings attributed to this
            vehicle.
    """

    id: str | int
    role: str = "insufficient_evidence"
    violations: list[str] = Field(default_factory=list)


class ReportViolation(BaseModel):
    """A single traffic-law violation identified in the reasoning report.

    Attributes:
        law_section: Traffic law section identifier (e.g. ``"1110"``).
        description: Human-readable description of the violation.
        severity: Severity classification — ``"minor"``, ``"moderate"``, or
            ``"major"``.
    """

    law_section: str = ""
    description: str
    severity: Literal["minor", "moderate", "major"] = "minor"


class ReasoningReport(BaseModel):
    """Structured liability assessment produced by the reasoning service.

    All fields are validated to ensure the reasoning service returned a
    well-formed response before it is stored in the database.

    Attributes:
        overall_assessment: Narrative summary of the incident and liability
            determination. Must be non-empty.
        overall_level: High-level liability classification.
        vehicles: Per-vehicle role and violation assignments.
        violations: Specific traffic-law violations identified.
        confidence: Model confidence in the assessment, in ``[0, 1]``.
        limitations: List of caveats or limitations of the analysis.
    """

    overall_assessment: str = Field(min_length=1)
    overall_level: Literal["potential_responsibility", "shared_responsibility", "insufficient_evidence"] = "insufficient_evidence"
    vehicles: list[ReportVehicle] = Field(default_factory=list)
    violations: list[ReportViolation] = Field(default_factory=list)
    confidence: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    limitations: list[str] = Field(default_factory=list)
