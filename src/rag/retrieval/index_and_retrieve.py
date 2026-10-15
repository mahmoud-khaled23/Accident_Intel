"""Dense + BM25 hybrid retrieval engine for the traffic-law RAG pipeline.

Supports three retrieval modes:

- ``"dense"``  — BAAI/bge-m3 embedding similarity via ChromaDB.
- ``"bm25"``   — keyword search with ``rank_bm25.BM25Okapi``.
- ``"hybrid"`` — both fused with Reciprocal Rank Fusion (default).

Key functions:
    :func:`index`    — embed child chunks and persist them to a local Chroma store.
    :func:`retrieve` — run one or more queries and return the top-k parent sections.
    :func:`load`     — load the pre-built ``parents.json`` / ``children.jsonl`` files.

Utility functions:
    :func:`tokenize`      — language-agnostic tokenizer supporting English and Arabic.
    :func:`get_bm25`      — lazily build/cache a per-process BM25 index.
    :func:`rrf_fuse`      — Reciprocal Rank Fusion combiner.
    :func:`merge_rankings` — round-robin interleaving of per-query ranked lists.

1) Index:     python index_and_retrieve.py index --out ./out
2) Retrieve:  python index_and_retrieve.py search --out ./out \\
                  -q "failure to yield right of way" -q "unsafe lane change"
              (default mode is hybrid = dense + BM25 fused with Reciprocal Rank Fusion)

pip install sentence-transformers chromadb rank-bm25
"""
import argparse
import json
import re
from functools import lru_cache
from threading import Lock
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ..chunking.build_chunks import resolve_ref
from ..chunking.query_expansion import expand_queries

MODEL_NAME = "BAAI/bge-m3"   # multilingual (English + Arabic) embedding model
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"   # optional cross-encoder used with --rerank
_index_lock = Lock()


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load and cache the BGE-M3 sentence-transformer embedding model.

    The model is loaded on CPU and cached for the lifetime of the process so
    that repeated calls never re-initialise it.

    Returns:
        SentenceTransformer: The loaded :data:`MODEL_NAME` model instance.
    """
    return SentenceTransformer(MODEL_NAME, device="cpu")


@lru_cache(maxsize=1)
def get_reranker():
    """Load and cache the BGE cross-encoder reranker model.

    Used only when ``rerank=True`` is passed to :func:`retrieve`.  Loaded on
    CPU and cached for the lifetime of the process.

    Returns:
        CrossEncoder: The loaded :data:`RERANKER_NAME` cross-encoder instance.
    """
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_NAME, device="cpu")

# ---------------------------------------------------------------- BM25 helpers
_AR_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")   # tashkeel + tatweel
_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    """Lowercase + light Arabic normalisation + word split (works for English and Arabic)."""
    t = _AR_DIACRITICS.sub("", text.lower())
    t = re.sub("[إأآٱ]", "ا", t)      # unify alef forms
    t = t.replace("ى", "ي").replace("ة", "ه")
    toks = []
    for w in _TOKEN.findall(t):
        if len(w) > 4 and w.startswith("ال"):   # strip Arabic definite article
            w = w[2:]
        toks.append(w)
    return toks


_BM25_CACHE = {}


def get_bm25(out, children):
    """Build (or return from cache) a BM25 index over the core child chunks.

    Filters ``children`` to only ``"core"`` category records (the same filter
    used by the dense search) before building the index.  The index is cached
    in :data:`_BM25_CACHE` keyed by the resolved absolute path of ``out`` so
    that multiple calls with the same directory never rebuild it.

    Args:
        out: Path to the output directory used as a cache key.
        children: List of child-chunk dicts loaded from ``children.jsonl``.

    Returns:
        tuple[BM25Okapi, list[str]]: The BM25 index and the parallel list of
        chunk IDs, both restricted to ``"core"`` chunks.
    """
    key = str(Path(out).resolve())
    if key not in _BM25_CACHE:
        from rank_bm25 import BM25Okapi
        core = [c for c in children if c["category"] == "core"]
        _BM25_CACHE[key] = (BM25Okapi([tokenize(c["text"]) for c in core]), [c["id"] for c in core])
    return _BM25_CACHE[key]


def rrf_fuse(rank_lists, weights, k=60):
    """Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Assigns each item a score of ``sum_i weights[i] / (k + rank_i(item))``
    where ``rank_i`` is the 1-based position in the ``i``-th list.  Items that
    do not appear in a list are not penalised — they simply receive no
    contribution from that list.  The constant ``k`` (default 60) controls
    how steeply rank differences are rewarded.

    Because only ranks matter, lists with incompatible scales (e.g. cosine
    distances and BM25 scores) can be combined without normalisation.

    Args:
        rank_lists: A list of ranked ID lists (each list is best-first).
        weights: A list of per-list weights (same length as ``rank_lists``).
        k: The RRF constant.  Higher values produce a flatter fusion.
            Defaults to ``60``.

    Returns:
        list[tuple[str, float]]: ``(chunk_id, rrf_score)`` pairs sorted by
        descending RRF score (best first).

    Example:
        >>> rrf_fuse([["a", "b", "c"], ["b", "a"]], [1.0, 1.0])
        [('b', ...), ('a', ...), ('c', ...)]
    """
    scores = {}
    for lst, w in zip(rank_lists, weights):
        for rank, cid in enumerate(lst, 1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------- storage
def load(out):
    """Load the pre-built parents index and children chunk list from disk.

    Expects ``<out>/parents.json`` (a dict of section records) and
    ``<out>/children.jsonl`` (one JSON object per line).

    Args:
        out: Path to the directory produced by the indexing step.

    Returns:
        tuple[dict, list[dict]]: ``(parents, children)`` where ``parents``
        maps section IDs to their full records and ``children`` is a list of
        chunk dicts.

    Raises:
        FileNotFoundError: If either ``parents.json`` or ``children.jsonl`` is
            missing from ``out``.
    """
    parents = json.loads((out / "parents.json").read_text(encoding="utf-8"))
    children = [json.loads(l) for l in open(out / "children.jsonl", encoding="utf-8")]
    return parents, children


def get_collection(out):
    """Open (or create) the Chroma collection for traffic-law chunks.

    Uses a persistent Chroma client rooted at ``<out>/chroma``.  The
    collection is named ``"traffic_laws"`` with cosine distance.

    Args:
        out: Path to the output directory containing the ``chroma/`` subdirectory.

    Returns:
        chromadb.Collection: The existing or newly created collection.
    """
    return client.get_or_create_collection("traffic_laws", metadata={"hnsw:space": "cosine"})


def index(out):
    """Embed every child chunk and upsert it into the local Chroma database.

    Only ``"core"`` category chunks are considered by retrieval, but all
    chunks are embedded here so the collection stays consistent with the full
    ``children.jsonl``.  Embeddings are computed in batches of 64 with L2
    normalisation.  Running this on an already-indexed directory is safe —
    Chroma's ``upsert`` is idempotent.

    Args:
        out: Path to the directory containing ``children.jsonl`` and the
            ``chroma/`` subdirectory.

    Returns:
        None
    """
    _, children = load(out)
    model = get_embedding_model()
    col = get_collection(out)
    B = 64
    for i in range(0, len(children), B):
        batch = children[i:i + B]
        col.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=model.encode([c["text"] for c in batch], normalize_embeddings=True).tolist(),
            metadatas=[{"parent_id": c["parent_id"], "article": str(c["article"]),
                        "section": c["section"], "sub": c["sub"],
                        "category": c["category"]} for c in batch],
        )
        print(f"indexed {min(i + B, len(children))}/{len(children)}")


def merge_rankings(ranked_lists, queries, max_total):
    """Interleave per-query ranked parent lists in round-robin order.

    Ensures every query contributes results before any single strong query
    fills all slots.  The best result from each query is collected first,
    then the second-best from each, and so on.

    Args:
        ranked_lists: A list of per-query ranked lists.  Each inner list
            contains ``(parent_id, value)`` pairs sorted best-first (lowest
            value = best when ``value`` is a distance).
        queries: The query strings corresponding to each inner list, used to
            record which queries matched a section.
        max_total: Maximum number of unique parent IDs to return.

    Returns:
        list[tuple[str, list[str], float]]: Up to ``max_total`` tuples of
        ``(parent_id, matched_queries, best_value)`` in the round-robin order
        they were first encountered.
    """
    order, matched, best = [], {}, {}
    depth = max((len(l) for l in ranked_lists), default=0)
    for rank in range(depth):
        for qi, lst in enumerate(ranked_lists):
            if rank < len(lst):
                pid, val = lst[rank]
                matched.setdefault(pid, []).append(queries[qi])
                best[pid] = min(val, best.get(pid, val))
                if pid not in order:
                    order.append(pid)
    return [(pid, matched[pid], best[pid]) for pid in order[:max_total]]


def retrieve(queries, out, top_k=5, per_query=None, expand_refs=False, refs_from_top=2,
             rerank=False, k_children=10, rerank_pool=30, debug=False,
             mode="hybrid", rrf_k=60, w_dense=1.0, w_bm25=1.0,
             expand="none", n_variants=3, glossary_path=None):
    """Retrieve the top-k parent law sections most relevant to the given queries.

    Runs a multi-query retrieval pipeline over the local vector store and BM25
    index.  Each query independently ranks small child chunks; the rankings are
    fused (optionally re-ranked) and merged in round-robin order so that every
    query contributes.

    Retrieval pipeline:

    1. Optionally expand queries using legal normalisation rules or an LLM
       (controlled by ``expand``).
    2. For each query, rank child chunks via the chosen ``mode``:

       - ``"dense"``  — cosine similarity with the BGE-M3 embedding model.
       - ``"bm25"``   — BM25Okapi keyword score.
       - ``"hybrid"`` — RRF fusion of both (default).

    3. Optionally re-score the fused candidate chunks with the BGE
       cross-encoder (``rerank=True``).
    4. Map chunk scores back to parent sections (best chunk wins per section).
    5. Merge per-query parent lists with :func:`merge_rankings`.
    6. Optionally append sections referenced by the top results
       (``expand_refs=True``).

    Note: On the first call, if the Chroma collection is empty, the index is
    built automatically — this may take several minutes.

    Args:
        queries: One or more search query strings.
        out: Path to the directory containing ``parents.json``,
            ``children.jsonl``, and ``chroma/``.
        top_k: Maximum number of parent sections to return.  Defaults to ``5``.
        per_query: Maximum sections a single query may contribute.
            Defaults to ``top_k``.
        expand_refs: If ``True``, fill remaining slots with sections referenced
            by the first ``refs_from_top`` results.  Defaults to ``False``.
        refs_from_top: How many top results to expand references from.
            Defaults to ``2``.
        rerank: Re-score candidates with the cross-encoder.  Slower but more
            accurate.  Defaults to ``False``.
        k_children: Chunks fetched per query in non-rerank mode.
            Defaults to ``10``.
        rerank_pool: Candidate chunks fetched per query when ``rerank=True``.
            Defaults to ``30``.
        debug: Print the top-15 chunks per query with their scores.
            Defaults to ``False``.
        mode: Retrieval mode — ``"dense"``, ``"bm25"``, or ``"hybrid"``.
            Defaults to ``"hybrid"``.
        rrf_k: RRF constant (see :func:`rrf_fuse`).  Defaults to ``60``.
        w_dense: Weight of the dense ranking in hybrid mode.  Defaults to ``1.0``.
        w_bm25: Weight of the BM25 ranking in hybrid mode.  Defaults to ``1.0``.
        expand: Query expansion method — ``"none"``, ``"rules"``, ``"llm"``,
            or ``"both"``.  Defaults to ``"none"``.
        n_variants: Max extra queries generated per original query when
            ``expand`` is not ``"none"``.  Defaults to ``3``.
        glossary_path: Optional path to a JSON file that extends the built-in
            legal glossary used by the ``"rules"`` expander.

    Returns:
        list[dict]: Up to ``top_k`` parent-section dicts.  Each dict is a copy
        of the record from ``parents.json`` augmented with:

        - ``"matched_queries"`` (list[str]): queries that surfaced this section.
        - ``"score"`` (float): The best score for this section across queries.
          Higher is better for rerank/BM25/hybrid; lower is better for dense.
        - ``"score_kind"`` (str): Human-readable description of what the score
          means (e.g. ``"RRF score (higher=better)"``).

    Example:
        >>> laws = retrieve(["failure to yield at intersection"], "./out", top_k=3)
        >>> laws[0]["id"]
        'VTL-1140'
    """
    per_query = per_query or top_k
    queries = expand_queries(queries, method=expand, n_variants=n_variants,
                             glossary_path=glossary_path, debug=debug)
    parents, children = load(out)
    child_by_id = {c["id"]: c for c in children}
    use_dense = mode in ("dense", "hybrid")
    use_bm25 = mode in ("bm25", "hybrid")

    model = get_embedding_model() if use_dense else None
    col = get_collection(out) if use_dense else None
    # Fresh checkouts ship law chunks; build the dense index on first use.
    if use_dense:
        with _index_lock:
            if col.count() < len(children):
                index(out)
    bm25, bm25_ids = get_bm25(out, children) if use_bm25 else (None, None)

    reranker = None
    if rerank:
        reranker = get_reranker()

    # several chunks can belong to the same section, so fetch more chunks than top_k
    n_chunks = max(rerank_pool, top_k * 5) if rerank else max(k_children, top_k * 3)

    ranked_lists = []
    for q in queries:
        dense_ids, dense_dist = [], {}
        if use_dense:
            r = col.query(
                query_embeddings=model.encode([q], normalize_embeddings=True).tolist(),
                n_results=n_chunks,
                where={"category": "core"},      # skip owner-liability / camera-program sections
            )
            dense_ids = r["ids"][0]
            dense_dist = dict(zip(dense_ids, r["distances"][0]))

        bm25_ids_ranked, bm25_score = [], {}
        if use_bm25:
            scores = bm25.get_scores(tokenize(q))
            top = sorted(range(len(scores)), key=lambda i: -scores[i])[:n_chunks]
            bm25_ids_ranked = [bm25_ids[i] for i in top if scores[i] > 0]   # score 0 = no keyword overlap
            bm25_score = {bm25_ids[i]: float(scores[i]) for i in top}

        # ---- fuse into ONE ranked list of chunk ids; value = -score so "lower = better"
        if mode == "hybrid":
            fused = rrf_fuse([dense_ids, bm25_ids_ranked], [w_dense, w_bm25], k=rrf_k)
            cand_ids = [cid for cid, _ in fused][:max(n_chunks, rerank_pool if rerank else 0)]
            base = {cid: -s for cid, s in fused}
        elif mode == "dense":
            cand_ids, base = dense_ids, dense_dist           # cosine distance, lower = better
        else:
            cand_ids = bm25_ids_ranked
            base = {cid: -bm25_score[cid] for cid in cand_ids}

        if reranker and cand_ids:                  # lower value = better, so negate the score
            docs = [child_by_id[cid]["text"] for cid in cand_ids]
            values = {cid: -float(s) for cid, s in zip(cand_ids, reranker.predict([(q, d) for d in docs]))}
        else:
            values = {cid: base[cid] for cid in cand_ids}

        if debug:                                 # show the best chunks, to see WHY a section ranks where it does
            print(f"\n[debug] query: {q}   (mode={mode}{', rerank' if reranker else ''})")
            for cid in sorted(values, key=values.get)[:15]:
                dr = dense_ids.index(cid) + 1 if cid in dense_ids else "-"
                br = bm25_ids_ranked.index(cid) + 1 if cid in bm25_ids_ranked else "-"
                print(f"   {cid:<24} value={values[cid]:.4f}  dense_rank={dr!s:<3} bm25_rank={br!s:<3}")

        best = {}  # parent_id -> best value among this query's chunks
        for cid, val in values.items():
            pid = child_by_id[cid]["parent_id"]
            best[pid] = min(val, best.get(pid, 1e9))
        ranked_lists.append(sorted(best.items(), key=lambda x: x[1])[:per_query])

    if reranker:
        kind = "rerank score (higher=better)"
    elif mode == "hybrid":
        kind = "RRF score (higher=better)"
    elif mode == "bm25":
        kind = "BM25 score (higher=better)"
    else:
        kind = "distance (lower=better)"
    flip = reranker is not None or mode in ("hybrid", "bm25")   # these were stored negated

    results = []
    for pid, matched, val in merge_rankings(ranked_lists, queries, top_k):
        p = dict(parents[pid])
        p["matched_queries"] = matched
        p["score"] = round(-val if flip else val, 4)
        p["score_kind"] = kind
        results.append(p)

    if expand_refs:                               # fill remaining free slots with referenced sections
        have = {p["id"] for p in results}
        for p in list(results[:refs_from_top]):
            for ref in p["refs"]:
                for rid in resolve_ref(ref, parents):
                    if len(results) >= top_k:
                        break
                    if rid not in have and parents[rid]["category"] == "core":
                        q = dict(parents[rid])
                        q["matched_queries"] = [f"(referenced by {p['id']})"]
                        q["score"], q["score_kind"] = None, "reference"
                        results.append(q); have.add(rid)
    return results[:top_k]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "search"])
    ap.add_argument("--out", default="./out")
    ap.add_argument("-q", action="append", default=[], help="query (repeat -q for several)")
    ap.add_argument("--top-k", type=int, default=5, help="max number of law sections returned")
    ap.add_argument("--per-query", type=int, default=None, help="max sections a single query may contribute")
    ap.add_argument("--expand-refs", action="store_true", help="use free slots for referenced sections")
    ap.add_argument("--refs-from-top", type=int, default=2, help="expand references of the first N results only")
    ap.add_argument("--rerank", action="store_true", help="re-score with a cross-encoder (more precise)")
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid",
                    help="dense = embeddings, bm25 = keywords, hybrid = both fused with RRF (default)")
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF constant (higher = flatter fusion)")
    ap.add_argument("--w-dense", type=float, default=1.0, help="weight of the dense ranking in hybrid mode")
    ap.add_argument("--w-bm25", type=float, default=1.0, help="weight of the BM25 ranking in hybrid mode")
    ap.add_argument("--expand", choices=["none", "rules", "llm", "both"], default="rules",
                    help="legal query normalization/expansion before search (default: rules)")
    ap.add_argument("--n-variants", type=int, default=3, help="max extra queries generated per query")
    ap.add_argument("--glossary", default=None, help="JSON file that extends/overrides the built-in legal glossary")
    ap.add_argument("--debug", action="store_true", help="print the top 15 chunks per query with their scores")
    a = ap.parse_args()

    if a.cmd == "index":
        index(a.out)
    else:
        res = retrieve(a.q, a.out, top_k=a.top_k, per_query=a.per_query,
                       expand_refs=a.expand_refs, refs_from_top=a.refs_from_top, rerank=a.rerank,
                       debug=a.debug, mode=a.mode, rrf_k=a.rrf_k, w_dense=a.w_dense, w_bm25=a.w_bm25,
                       expand=a.expand, n_variants=a.n_variants, glossary_path=a.glossary)
        print(f"\nReturned {len(res)} sections (top_k={a.top_k}, mode={a.mode})")
        for i, p in enumerate(res, 1):
            print(f"\n=== #{i} {p['id']} | {p['title'][:80]} ===")
            print(f"matched: {p['matched_queries']} | {p['score_kind']}: {p['score']}")
            print(p["text"][:300])
