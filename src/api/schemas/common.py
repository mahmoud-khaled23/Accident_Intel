from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class StatusEnum(str, Enum):
    created = "created"
    uploaded = "uploaded"
    processing = "processing"
    tracking = "tracking"
    vlm_analysis = "vlm_analysis"
    law_retrieval = "law_retrieval"
    reasoning = "reasoning"
    completed = "completed"
    failed = "failed"


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    limit: int
