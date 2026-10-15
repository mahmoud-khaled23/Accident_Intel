"""Shared Pydantic schemas used across multiple API routers.

Provides the :class:`StatusEnum` for accident/analysis lifecycle states and
the generic :class:`PaginatedResponse` wrapper for list endpoints.
"""
from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class StatusEnum(str, Enum):
    """Lifecycle status values for an accident analysis job.

    Attributes:
        created: Record created but no video uploaded yet.
        uploaded: Video file has been successfully uploaded.
        processing: Analysis job has been claimed and is about to start.
        tracking: Vehicle tracking step is running.
        vlm_analysis: VLM scene description step is running.
        law_retrieval: Traffic-law RAG retrieval step is running.
        reasoning: Liability reasoning step is running.
        completed: All pipeline steps finished successfully.
        failed: One or more pipeline steps encountered an error.
    """

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
    """Generic paginated list response wrapper.

    Attributes:
        items: The records on the current page.
        total: Total number of matching records across all pages.
        page: Current 1-based page number.
        limit: Maximum number of records per page as requested.
        page_size: Alias for ``limit``; populated by endpoints so the
            frontend can use either field name.
    """

    items: list[T]
    total: int
    page: int
    limit: int
    page_size: int | None = None  # alias for limit — populated by endpoints

    @property
    def pageSize(self) -> int:
        """Return ``page_size`` if set, otherwise fall back to ``limit``.

        Returns:
            Effective page size as an integer.
        """
        return self.page_size if self.page_size is not None else self.limit
