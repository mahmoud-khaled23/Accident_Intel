"""Laws router — traffic law search via the RAG index."""
from fastapi import APIRouter, Depends, HTTPException

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
    raw = law_service.retrieve_laws(section_id, str(settings.rag_out_dir), settings)
    for item in raw:
        if item.get("section_id", "") == section_id or item.get("section_id", "").endswith(section_id):
            return LawResult(
                section_id=item["section_id"],
                title=item.get("title", ""),
                content=item.get("content", ""),
                score=float(item.get("score") or 0.0),
                metadata=item.get("metadata", {}),
            )
    raise HTTPException(status_code=404, detail=f"Law section '{section_id}' not found")
