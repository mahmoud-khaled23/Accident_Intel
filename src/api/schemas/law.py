"""Pydantic schemas for the traffic-law API endpoints.

Defines the response models used by the laws router for individual law
entries and search result collections.
"""
from pydantic import BaseModel


class LawResult(BaseModel):
    """A single traffic law entry returned by the laws router.

    Attributes:
        section_id: Unique section identifier (e.g. ``"VTL-1140"``).
        title: Short human-readable title of the law section.
        content: Full text of the law section.
        score: Relevance score from the RAG retriever in ``[0, 1]``.
            ``0.0`` when the entry was retrieved by ID rather than by query.
        metadata: Additional key-value pairs from the law corpus, such as
            jurisdiction and category tags.
    """

    section_id: str
    title: str = ""
    content: str
    score: float = 0.0
    metadata: dict = {}


class LawSearchResponse(BaseModel):
    """Response wrapper for a traffic-law search query.

    Attributes:
        results: List of matching law entries, ordered by relevance score.
        query: The search query string that produced these results.
        total: Total number of results returned (``len(results)``).
    """

    results: list[LawResult]
    query: str
    total: int
