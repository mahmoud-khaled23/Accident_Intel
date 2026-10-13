from typing import Any

from pydantic import BaseModel


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


class AnalysisStatus(BaseModel):
    status: str
    message: str
