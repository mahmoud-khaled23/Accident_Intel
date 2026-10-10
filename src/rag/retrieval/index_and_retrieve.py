"""
index_and_retrieve.py
---------------------
1) Index:     python index_and_retrieve.py index --out ./out
2) Retrieve:  python index_and_retrieve.py search --out ./out \
                  -q "failure to yield right of way" -q "unsafe lane change"
              (default mode is hybrid = dense + BM25 fused with Reciprocal Rank Fusion)

pip install sentence-transformers chromadb rank-bm25
"""
import argparse
import json
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ..chunking.build_chunks import resolve_ref
from ..chunking.query_expansion import expand_queries

MODEL_NAME = "BAAI/bge-m3"   # multilingual (English + Arabic) embedding model
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"   # optional cross-encoder used with --rerank

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
    """BM25 index over the 'core' child chunks (same filter as the dense search). Built once per process."""
    key = str(Path(out).resolve())
    if key not in _BM25_CACHE:
        from rank_bm25 import BM25Okapi
        core = [c for c in children if c["category"] == "core"]
        _BM25_CACHE[key] = (BM25Okapi([tokenize(c["text"]) for c in core]), [c["id"] for c in core])
    return _BM25_CACHE[key]


def rrf_fuse(rank_lists, weights, k=60):
    """
    Reciprocal Rank Fusion. rank_lists[i] = [chunk_id, ...] best first.
    score(id) = sum_i  weights[i] / (k + rank_i(id))      (rank starts at 1)
    Returns [(chunk_id, rrf_score), ...] best first. Only ranks matter, so dense distances and
    BM25 scores (different scales) can be combined without normalising.
    """
    scores = {}
    for lst, w in zip(rank_lists, weights):
        for rank, cid in enumerate(lst, 1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------- storage
def load(out):
    out = Path(out)
    parents = json.loads((out / "parents.json").read_text(encoding="utf-8"))
    children = [json.loads(l) for l in open(out / "children.jsonl", encoding="utf-8")]
    return parents, children


def get_collection(out):
    client = chromadb.PersistentClient(str(Path(out) / "chroma"))
    return client.get_or_create_collection("traffic_laws", metadata={"hnsw:space": "cosine"})


def index(out):
    """Embed every child chunk and store it in a local Chroma database. (BM25 needs no indexing step.)"""
    _, children = load(out)
    model = SentenceTransformer(MODEL_NAME)
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
    """
    ranked_lists[i] = [(parent_id, value), ...] for queries[i], best (lowest value) first.
    Interleave them round-robin (best of each query first, then second best of each...) so that
    every query contributes, instead of one strong query crowding out the others.
    Returns [(parent_id, [queries that matched it], best_value), ...] capped at max_total.
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
    """
    expand      : "none" | "rules" | "llm" | "both"  - legal query normalization/expansion applied BEFORE the
                  search (see query_expansion.py). Originals are served first, variants fill in.

    Returns AT MOST `top_k` full law sections (the standard RAG "top_k").

    1. For EACH query, rank the small child chunks:
         mode="dense"  : embedding similarity only
         mode="bm25"   : keyword (BM25) only
         mode="hybrid" : both, fused with Reciprocal Rank Fusion (weights w_dense / w_bm25)
       then rank the parent sections they belong to (best chunk wins).
    2. Merge the queries round-robin (best of every query first) and keep the first `top_k`.
    3. expand_refs=True: sections referenced by the first `refs_from_top` results may fill any
       slots that are still free, but the total never goes above `top_k`.

    rerank=True : the fused candidate chunks are re-scored with a cross-encoder, which reads
                  query + chunk together (more precise, slower).
    per_query   : how many sections each query may contribute (default: top_k).
    Each returned section has "matched_queries", "score" and "score_kind".
    """
    per_query = per_query or top_k
    queries = expand_queries(queries, method=expand, n_variants=n_variants,
                             glossary_path=glossary_path, debug=debug)
    parents, children = load(out)
    child_by_id = {c["id"]: c for c in children}
    use_dense = mode in ("dense", "hybrid")
    use_bm25 = mode in ("bm25", "hybrid")

    model = SentenceTransformer(MODEL_NAME) if use_dense else None
    col = get_collection(out) if use_dense else None
    bm25, bm25_ids = get_bm25(out, children) if use_bm25 else (None, None)

    reranker = None
    if rerank:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANKER_NAME)

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