"""Laws router — traffic law search via the RAG index.

Provides endpoints for searching the traffic-law corpus and fetching
individual law entries by their section ID. All routes are prefixed with
``/api/laws``.
"""
from fastapi import APIRouter, Depends

from ..dependencies import get_settings
from ..schemas.law import LawResult, LawSearchResponse
from ..services import law_service

router = APIRouter(prefix="/api/laws", tags=["laws"])


@router.get("/", response_model=LawSearchResponse)
def search_laws(
    q: str = "",
    jurisdiction: str = "",
    topic: str = "",
    limit: int = 10,
    settings=Depends(get_settings),
):
    """Search the traffic-law corpus by query string.

    When no query is provided and mock mode is enabled, all mock laws are
    returned (useful for the Laws Library page).  When no query is provided
    in real mode, an empty result set is returned.

    Args:
        q: Free-text search query (e.g. ``"right of way at intersections"``).
        jurisdiction: Optional jurisdiction filter (not currently applied by
            the RAG backend; reserved for future use).
        topic: Optional topic filter (reserved for future use).
        limit: Maximum number of results to return.
        settings: Injected application settings.

    Returns:
        A :class:`~schemas.law.LawSearchResponse` containing matching law
        entries, the query string, and the total count.
    """
    if not q.strip():
        # Return all mock laws when no query provided (useful for the Laws Library page)
        if settings.rag_mock:
            from ..services.law_service import _MOCK_LAWS
            results = [
                LawResult(
                    section_id=item.get("section_id", ""),
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    score=float(item.get("score") or 0.0),
                    metadata=item.get("metadata", {}),
                )
                for item in _MOCK_LAWS[:limit]
            ]
            return LawSearchResponse(results=results, query=q, total=len(results))
        return LawSearchResponse(results=[], query=q, total=0)

    raw = law_service.retrieve_laws(q, str(settings.rag_out_dir), settings)

    results = [
        LawResult(
            section_id=item.get("section_id", ""),
            title=item.get("title", ""),
            content=item.get("content", ""),
            score=float(item.get("score") or 0.0),
            metadata=item.get("metadata", {}),
        )
        for item in raw[:limit]
    ]
    return LawSearchResponse(results=results, query=q, total=len(results))


@router.get("/{section_id}", response_model=LawResult)
def get_law(section_id: str, settings=Depends(get_settings)):
    """Fetch a single traffic law entry by its section ID.

    Args:
        section_id: The law section identifier (e.g. ``"VTL-1140"``).
        settings: Injected application settings.

    Returns:
        A :class:`~schemas.law.LawResult` for the requested section.

    Raises:
        :class:`~fastapi.HTTPException` 404: Section ID not found.
    """
    item = law_service.get_law_by_id(section_id, str(settings.rag_out_dir), settings)
    if item is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Law section '{section_id}' not found")
    return LawResult(
        section_id=item["section_id"],
        title=item.get("title", ""),
        content=item.get("content", ""),
        score=float(item.get("score") or 0.0),
        metadata=item.get("metadata", {}),
    )
