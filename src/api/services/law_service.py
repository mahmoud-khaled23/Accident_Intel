"""Law retrieval service — wraps src/rag/retrieval/report_retrieval.py."""
import sys
from pathlib import Path

# Ensure project root and src/ are on sys.path
_project_root = str(Path(__file__).parents[3])
_src_path = str(Path(__file__).parents[3] / "src")
for _p in (_project_root, _src_path):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_MOCK_LAWS = [
    {
        "section_id": "VTL-1140",
        "title": "Right of way at intersections",
        "content": (
            "The driver of a vehicle approaching an intersection shall yield the right of way "
            "to a vehicle which has entered the intersection from a different highway."
        ),
        "score": 0.91,
        "metadata": {"jurisdiction": "NYS VTL", "category": "core"},
    },
    {
        "section_id": "VTL-1128",
        "title": "Driving on roadways laned for traffic",
        "content": (
            "Whenever any roadway has been divided into two or more clearly marked lanes for "
            "traffic, a vehicle shall be driven as nearly as practicable entirely within a "
            "single lane and shall not be moved from such lane until the driver has first "
            "ascertained that such movement can be made with safety."
        ),
        "score": 0.87,
        "metadata": {"jurisdiction": "NYS VTL", "category": "core"},
    },
    {
        "section_id": "VTL-1163",
        "title": "Signals on starting, stopping or turning",
        "content": (
            "No person shall turn a vehicle at an intersection unless the vehicle is in proper "
            "position upon the roadway as required in this title, or turn a vehicle to enter a "
            "private road or driveway, or otherwise turn a vehicle from a direct course or move "
            "right or left upon a roadway, unless and until such movement can be made with "
            "reasonable safety."
        ),
        "score": 0.83,
        "metadata": {"jurisdiction": "NYS VTL", "category": "core"},
    },
]

# Build a direct-lookup index from the mock list so get_law_by_id is O(1).
_MOCK_LAWS_INDEX: dict[str, dict] = {item["section_id"]: item for item in _MOCK_LAWS}


def retrieve_laws(description: str, out_dir: str, settings) -> list[dict]:
    """Retrieve relevant traffic laws for the given accident description.

    Returns a list of dicts with keys: section_id, title, content, score, metadata.
    """
    if settings.rag_mock:
        return _MOCK_LAWS

    try:
        from rag.retrieval.report_retrieval import retrieve_for_report
    except ImportError as exc:
        raise RuntimeError(
            f"rag package not importable — check rag/retrieval dependencies. Original error: {exc}"
        ) from exc

    raw = retrieve_for_report(
        description=description,
        out=out_dir,
        top_k=5,
        rerank=True,
        return_details=False,
    )

    results = []
    for item in raw:
        results.append(
            {
                "section_id": item.get("id", ""),
                "title": item.get("title", ""),
                "content": item.get("text", ""),
                "score": float(item.get("score") or 0.0),
                "metadata": {
                    k: v
                    for k, v in item.items()
                    if k not in ("id", "title", "text", "score")
                },
            }
        )
    return results


def get_law_by_id(section_id: str, out_dir: str, settings) -> dict | None:
    """Return a single law entry by its section ID, or None if not found.

    In mock mode this is a direct dict lookup against the in-memory index.
    In real RAG mode a semantic search is used as a best-effort fallback because
    the RAG backend does not expose a direct-by-ID retrieval API; the caller
    should treat a None return as a definitive 404 only in mock mode.
    """
    if settings.rag_mock:
        return _MOCK_LAWS_INDEX.get(section_id)

    # Real RAG: perform a similarity search and look for an exact section_id match
    # in the returned results.  This is inherently unreliable for opaque IDs.
    results = retrieve_laws(section_id, out_dir, settings)
    for item in results:
        sid = item.get("section_id", "")
        if sid == section_id or sid.endswith(section_id):
            return item
    return None
