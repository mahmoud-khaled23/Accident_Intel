"""Laws router — traffic law browsing and search via the RAG corpus.

Provides endpoints for browsing the full traffic-law corpus (the same
``parents.json`` records the RAG pipeline retrieves from), searching it by
query string, and fetching individual law entries by their section ID.
All routes are prefixed with ``/api/laws``.
"""
from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import get_settings
from ..schemas.law import LawFacets, LawResult, LawSearchResponse
from ..services import law_service

router = APIRouter(prefix="/api/laws", tags=["laws"])


def _to_law_result(item: dict) -> LawResult:
    """Convert a law-service entry dict into a :class:`LawResult`.

    Args:
        item: Entry with ``section_id``, ``title``, ``content``, ``score``
            and ``metadata`` keys.

    Returns:
        LawResult: Pydantic model for the API response.
    """
    return LawResult(
        section_id=item.get("section_id", ""),
        title=item.get("title", ""),
        content=item.get("content", ""),
        score=float(item.get("score") or 0.0),
        metadata=item.get("metadata", {}),
    )


@router.get("/", response_model=LawSearchResponse)
def search_laws(
    q: str = "",
    jurisdiction: str = "",
    topic: str = "",
    category: str = "",
    article: str = "",
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    rerank: bool = Query(False, description=(
        "Apply the cross-encoder precision pass. Off by default: it costs "
        "seconds per query on CPU and is not needed for interactive search.")),
    settings=Depends(get_settings),
):
    """Browse the law corpus, or search it when a query is provided.

    * **Browse** (empty ``q``): every law section in the corpus, ordered by
      article and section number, with facet counts so clients can render
      category/article filters. ``category`` and ``article`` narrow the
      listing; ``limit``/``offset`` paginate it. In mock mode the built-in
      demo laws are returned instead.
    * **Search** (non-empty ``q``): the existing hybrid RAG retrieval is
      used and results are ordered by relevance. ``category``/``article``
      filter the retrieved sections.

    Args:
        q: Free-text search query (e.g. ``"right of way at intersections"``).
            Leave empty to browse.
        jurisdiction: Deprecated, accepted for compatibility; the corpus is
            New York State VTL only, so this filter is not applied.
        topic: Deprecated, accepted for compatibility; use ``category``.
        category: Exact corpus category filter (e.g. ``"core"``).
        article: Exact article-number filter (e.g. ``"24"``).
        limit: Maximum number of results to return.
        offset: Number of matches to skip (pagination).
        rerank: Opt into the slow cross-encoder reranking pass.
        settings: Injected application settings.

    Returns:
        A :class:`~schemas.law.LawSearchResponse` containing the law
        entries, the query string, the total number of matches, and facet
        counts for browse responses.

    Raises:
        HTTPException 503: If the corpus files are missing or retrieval
            fails; the detail explains how to fix it.
    """
    if not q.strip():
        if settings.rag_mock:
            results = [_to_law_result(item) for item in law_service._MOCK_LAWS[:limit]]
            return LawSearchResponse(results=results, query=q, total=len(results),
                                     offset=offset)
        try:
            entries, total, facets = law_service.list_laws(
                str(settings.rag_out_dir), category=category, article=article,
                limit=limit, offset=offset,
            )
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return LawSearchResponse(
            results=[_to_law_result(item) for item in entries],
            query=q, total=total, offset=offset,
            facets=LawFacets(**facets),
        )

    try:
        raw = law_service.retrieve_laws(q, str(settings.rag_out_dir), settings, rerank=rerank)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    filtered = [
        item for item in raw
        if (not category or str(item.get("metadata", {}).get("category") or "") == category)
        and (not article or str(item.get("metadata", {}).get("article") or "") == str(article))
    ]
    results = [_to_law_result(item) for item in filtered[:limit]]
    return LawSearchResponse(results=results, query=q, total=len(filtered), offset=offset)


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
    try:
        item = law_service.get_law_by_id(section_id, str(settings.rag_out_dir), settings)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=404, detail=f"Law section '{section_id}' not found")
    return _to_law_result(item)
