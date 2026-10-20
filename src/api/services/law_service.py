"""Law retrieval service — wraps src/rag/retrieval/report_retrieval.py.

Provides three public functions:

* :func:`list_laws` — enumerate the full law corpus for browsing, with
  category/article filters, facet counts, and pagination.
* :func:`retrieve_laws` — perform a semantic search over the traffic-law
  corpus and return relevant law entries.
* :func:`get_law_by_id` — look up a single law entry by its section ID.

When ``settings.rag_mock`` is ``True`` the search and lookup functions use
the hard-coded :data:`_MOCK_LAWS` list so the API works without a built RAG
index.  Browsing always reads the corpus files, never mock data.
"""
from pathlib import Path


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
"""Small set of mock traffic laws returned when ``settings.rag_mock = True``."""

# Build a direct-lookup index from the mock list so get_law_by_id is O(1).
_MOCK_LAWS_INDEX: dict[str, dict] = {item["section_id"]: item for item in _MOCK_LAWS}
"""Dict index for O(1) mock-law lookup by section ID."""


def _section_sort_key(section: str):
    """Natural-sort key for section numbers such as ``"1111-c-1"``.

    Splits the section on ``"-"`` and orders numeric tokens before letter
    suffixes, so ``1111`` sorts before ``1111-a`` and ``1111-c`` before
    ``1111-c-1``.

    Args:
        section: Raw section number string from the corpus (e.g. ``"1144-a"``).

    Returns:
        list[tuple[int, int, str]]: Sortable key of per-token tuples.
    """
    key = []
    for token in str(section).split("-"):
        key.append((0, int(token), "") if token.isdigit() else (1, 0, token))
    return key


def _parent_to_entry(item: dict) -> dict:
    """Convert a ``parents.json`` record into the API entry shape.

    Args:
        item: One parent-section record from ``parents.json``.

    Returns:
        dict: Keys ``section_id``, ``title``, ``content``, ``score`` and
        ``metadata`` (everything except the core fields).
    """
    return {
        "section_id": item.get("id", ""),
        "title": item.get("title", ""),
        "content": item.get("text", ""),
        "score": 0.0,
        "metadata": {k: v for k, v in item.items() if k not in {"id", "title", "text"}},
    }


def list_laws(out_dir: str, category: str = "", article: str = "",
              limit: int = 50, offset: int = 0) -> tuple[list[dict], int, dict]:
    """Enumerate the law corpus for the browsable Traffic Laws Library.

    Loads ``parents.json`` directly — no embedding model, BM25 index, or
    reranker is touched, so this stays fast even on the first request.

    Args:
        out_dir: Directory holding ``parents.json`` and ``children.jsonl``.
        category: Optional exact category filter (e.g. ``"core"``).
        article: Optional exact article-number filter (e.g. ``"24"``).
        limit: Maximum number of entries to return.
        offset: Number of matching entries to skip (pagination).

    Returns:
        tuple[list[dict], int, dict]: A 3-tuple of:
            - the paginated entries in article/section order, in the same
              shape as :func:`get_law_by_id` returns;
            - the total number of matches before slicing;
            - facet counts ``{"categories": {...}, "articles": {...}}``
              computed over the full corpus.

    Raises:
        RuntimeError: If the corpus files are missing or empty.
    """
    from src.rag.retrieval.index_and_retrieve import load
    try:
        parents, _ = load(Path(out_dir))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Traffic-law listing failed: {exc}") from exc

    ordered = sorted(
        parents.values(),
        key=lambda item: (int(item.get("article") or 0),
                          _section_sort_key(item.get("section") or "")),
    )

    categories: dict[str, int] = {}
    articles: dict[str, int] = {}
    for item in ordered:
        cat = str(item.get("category") or "")
        art = str(item.get("article") or "")
        if cat:
            categories[cat] = categories.get(cat, 0) + 1
        if art:
            articles[art] = articles.get(art, 0) + 1

    filtered = [
        item for item in ordered
        if (not category or str(item.get("category") or "") == category)
        and (not article or str(item.get("article") or "") == str(article))
    ]
    page = [_parent_to_entry(item) for item in filtered[offset:offset + limit]]
    return page, len(filtered), {"categories": categories, "articles": articles}


def retrieve_laws(description: str, out_dir: str, settings, rerank: bool | None = None) -> list[dict]:
    """Retrieve relevant traffic laws for the given accident description.

    In mock mode returns :data:`_MOCK_LAWS` directly.  In real mode calls
    :func:`rag.retrieval.report_retrieval.retrieve_for_report` with optional
    query expansion.

    Args:
        description: Free-text accident description used as the retrieval query.
        out_dir: Output directory path passed to the RAG retriever (the
            Chroma index must already exist here in real mode).
        settings: Application settings. ``settings.rag_mock``,
            ``settings.rag_rerank``, and ``settings.rag_mode`` are
            consulted.
        rerank: Overrides ``settings.rag_rerank`` when given.  Interactive
            callers pass ``False`` because the cross-encoder costs seconds
            per query on CPU; pipeline callers keep the setting default.

    Returns:
        A list of dicts with keys ``section_id``, ``title``, ``content``,
        ``score``, and ``metadata``.

    Raises:
        RuntimeError: If the ``rag`` package is not importable (real mode).
        RuntimeError: If the RAG retrieval itself raises an exception.
    """
    if settings.rag_mock:
        return _MOCK_LAWS
    use_rerank = settings.rag_rerank if rerank is None else rerank

    try:
        from src.rag.chunking.query_expansion import expand_queries
        from src.rag.retrieval.report_retrieval import retrieve_for_report
    except ImportError as exc:
        raise RuntimeError(
            f"rag package not importable — check rag/retrieval dependencies. Original error: {exc}"
        ) from exc

    try:
        extra_queries = expand_queries([description[:300]], method="rules")
        raw = retrieve_for_report(
            description=description,
            out=Path(out_dir),   # ensure Path, not string
            extra_queries=extra_queries,
            top_k=5,
            rerank=use_rerank,
            mode=settings.rag_mode,
            return_details=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Traffic-law retrieval failed: {exc}") from exc

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

    In mock mode this is a direct dict lookup against the in-memory
    :data:`_MOCK_LAWS_INDEX`. In real RAG mode the section is looked up directly in parents.json,
    without loading an embedding model.

    Args:
        section_id: The law section identifier to look up
            (e.g. ``"VTL-1140"``).
        out_dir: Output directory path passed to :func:`retrieve_laws` in
            real mode.
        settings: Application settings.

    Returns:
        The matching law entry dict, or ``None`` if not found.
    """
    if settings.rag_mock:
        return _MOCK_LAWS_INDEX.get(section_id)

    from src.rag.retrieval.index_and_retrieve import load
    try:
        parents, _ = load(out_dir)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Traffic-law lookup failed: {exc}") from exc
    key = section_id.strip().upper()
    if not key.startswith("VTL-"):
        key = f"VTL-{key}"
    item = parents.get(key)
    if item is None:
        return None
    return {"section_id": item["id"], "title": item.get("title", ""),
            "content": item.get("text", ""), "score": 0.0,
            "metadata": {k: v for k, v in item.items() if k not in {"id", "title", "text"}}}
