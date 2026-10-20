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
            article number and category tags.
    """

    section_id: str
    title: str = ""
    content: str
    score: float = 0.0
    metadata: dict = {}


class LawFacets(BaseModel):
    """Facet counts over the browseable law corpus.

    Attributes:
        categories: Map of corpus category (e.g. ``"core"``) to the number of
            law sections in that category.
        articles: Map of article number (e.g. ``"24"``) to the number of law
            sections in that article.
    """

    categories: dict[str, int] = {}
    articles: dict[str, int] = {}


class LawSearchResponse(BaseModel):
    """Response wrapper for a traffic-law browse or search query.

    Attributes:
        results: List of matching law entries. Browse results are ordered by
            article/section; search results are ordered by relevance.
        query: The search query string that produced these results (empty
            for browse responses).
        total: Total number of matches before ``limit``/``offset`` slicing,
            so clients can paginate.
        offset: Offset of the first item in ``results`` within ``total``.
        facets: Facet counts over the full corpus (browse responses only).
    """

    results: list[LawResult]
    query: str
    total: int
    offset: int = 0
    facets: LawFacets | None = None
