"""
index_and_retrieve.py
---------------------
1) Index:     python index_and_retrieve.py index --out ./out
2) Retrieve:  python index_and_retrieve.py search --out ./out \
                  -q "failure to yield right of way" -q "unsafe lane change"

pip install sentence-transformers chromadb
"""
import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from build_chunks import resolve_ref

MODEL_NAME = "BAAI/bge-m3"   # multilingual (English + Arabic) embedding model
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"   # optional cross-encoder used with --rerank


def load(out):
    out = Path(out)
    parents = json.loads((out / "parents.json").read_text(encoding="utf-8"))
    children = [json.loads(l) for l in open(out / "children.jsonl", encoding="utf-8")]
    return parents, children


def get_collection(out):
    client = chromadb.PersistentClient(str(Path(out) / "chroma"))
    return client.get_or_create_collection("traffic_laws", metadata={"hnsw:space": "cosine"})


def index(out):
    """Embed every child chunk and store it in a local Chroma database."""
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
             rerank=False, k_children=10, rerank_pool=30, debug=False):
    """
    Returns AT MOST `top_k` full law sections (the standard RAG "top_k").

    1. For EACH query, search the small child chunks and rank the parent sections they belong to.
    2. Merge the queries round-robin (best of every query first) and keep the first `top_k`.
    3. expand_refs=True: sections referenced by the first `refs_from_top` results may fill any
       slots that are still free, but the total never goes above `top_k`.

    rerank=True : fetch more chunks per query and re-score them with a cross-encoder, which reads
                  query + chunk together (more precise, slower).
    per_query   : how many sections each query may contribute (default: top_k, so a single query
                  can fill all the slots; the round-robin merge keeps it fair with several queries).
    Each returned section has "matched_queries", "score" and "score_kind".
    """
    per_query = per_query or top_k
    parents, _ = load(out)
    model = SentenceTransformer(MODEL_NAME)
    col = get_collection(out)

    reranker = None
    if rerank:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANKER_NAME)

    # several chunks can belong to the same section, so fetch more chunks than top_k
    n_chunks = max(rerank_pool, top_k * 5) if rerank else max(k_children, top_k * 3)

    ranked_lists = []
    for q in queries:
        r = col.query(
            query_embeddings=model.encode([q], normalize_embeddings=True).tolist(),
            n_results=n_chunks,
            where={"category": "core"},          # skip owner-liability / camera-program sections
        )
        metas, docs, dists = r["metadatas"][0], r["documents"][0], r["distances"][0]
        if reranker:                              # lower value = better, so negate the score
            values = [-float(x) for x in reranker.predict([(q, d) for d in docs])]
        else:
            values = list(dists)
        if debug:                                 # show the best chunks, to see WHY a section ranks where it does
            print(f"\n[debug] query: {q}")
            for i in sorted(range(len(values)), key=lambda i: values[i])[:15]:
                extra = f"  rerank={-values[i]:.3f}" if reranker else ""
                print(f"   {r['ids'][0][i]:<24} distance={dists[i]:.3f}{extra}")
        best = {}  # parent_id -> best value among this query's chunks
        for meta, val in zip(metas, values):
            pid = meta["parent_id"]
            best[pid] = min(val, best.get(pid, 1e9))
        ranked_lists.append(sorted(best.items(), key=lambda x: x[1])[:per_query])

    kind = "rerank score (higher=better)" if rerank else "distance (lower=better)"
    results = []
    for pid, matched, val in merge_rankings(ranked_lists, queries, top_k):
        p = dict(parents[pid])
        p["matched_queries"] = matched
        p["score"] = round(-val if rerank else val, 3)
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
    ap.add_argument("--debug", action="store_true", help="print the top 15 chunks per query with their scores")
    a = ap.parse_args()

    if a.cmd == "index":
        index(a.out)
    else:
        res = retrieve(a.q, a.out, top_k=a.top_k, per_query=a.per_query,
                       expand_refs=a.expand_refs, refs_from_top=a.refs_from_top, rerank=a.rerank, debug=a.debug)
        print(f"\nReturned {len(res)} sections (top_k={a.top_k})")
        for i, p in enumerate(res, 1):
            print(f"\n=== #{i} {p['id']} | {p['title'][:80]} ===")
            print(f"matched: {p['matched_queries']} | {p['score_kind']}: {p['score']}")
            print(p["text"][:300])