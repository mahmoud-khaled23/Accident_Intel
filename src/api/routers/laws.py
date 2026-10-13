"""Laws router — traffic law search via the RAG index."""
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

    try:
        raw = law_service.retrieve_laws(q, str(settings.rag_out_dir), settings)
    except Exception as exc:
        # RAG deps missing (chromadb, sentence-transformers) — fall back to mock
        import logging
        logging.getLogger(__name__).warning(
            "RAG retrieval failed (%s) — falling back to mock laws. "
            "Install chromadb and sentence-transformers to enable real retrieval.",
            exc,
        )
        from ..services.law_service import _MOCK_LAWS
        raw = _MOCK_LAWS

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
    try:
        item = law_service.get_law_by_id(section_id, str(settings.rag_out_dir), settings)
    except Exception:
        item = None
    if item is None:
        # Try mock index as fallback
        from ..services.law_service import _MOCK_LAWS_INDEX
        item = _MOCK_LAWS_INDEX.get(section_id)
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
