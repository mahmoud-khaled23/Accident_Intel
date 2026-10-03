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


def retrieve(queries, out, k_children=10, per_query=3, max_total=8, expand_refs=False,
             refs_from_top=2, rerank=False, rerank_pool=30):
    """
    For EACH query: search the small child chunks and keep its top `per_query` parent sections.
    Then merge the queries round-robin and return the full PARENT sections (deduplicated).

    rerank=True      : fetch `rerank_pool` chunks per query and re-score them with a cross-encoder
                       (reads query + chunk together, so it knows "red light" is not an
                       emergency-vehicle red light). Slower but noticeably more precise.
    expand_refs=True : also add sections referenced by the FIRST `refs_from_top` results only,
                       so irrelevant results don't drag in more irrelevant sections.
    Each returned section has "matched_queries", "score" and "score_kind".
    """
    parents, _ = load(out)
    model = SentenceTransformer(MODEL_NAME)
    col = get_collection(out)

    reranker = None
    if rerank:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANKER_NAME)

    ranked_lists = []
    for q in queries:
        r = col.query(
            query_embeddings=model.encode([q], normalize_embeddings=True).tolist(),
            n_results=rerank_pool if rerank else k_children,
            where={"category": "core"},          # skip owner-liability / camera-program sections
        )
        metas, docs, dists = r["metadatas"][0], r["documents"][0], r["distances"][0]
        if reranker:                              # lower value = better, so negate the score
            values = [-float(x) for x in reranker.predict([(q, d) for d in docs])]
        else:
            values = list(dists)
        best = {}  # parent_id -> best value among this query's chunks
        for meta, val in zip(metas, values):
            pid = meta["parent_id"]
            best[pid] = min(val, best.get(pid, 1e9))
        ranked_lists.append(sorted(best.items(), key=lambda x: x[1])[:per_query])

    kind = "rerank score (higher=better)" if rerank else "distance (lower=better)"
    results = []
    for pid, matched, val in merge_rankings(ranked_lists, queries, max_total):
        p = dict(parents[pid])
        p["matched_queries"] = matched
        p["score"] = round(-val if rerank else val, 3)
        p["score_kind"] = kind
        results.append(p)

    if expand_refs:
        have = {p["id"] for p in results}
        for p in list(results[:refs_from_top]):
            for ref in p["refs"]:
                for rid in resolve_ref(ref, parents):
                    if rid not in have and parents[rid]["category"] == "core":
                        q = dict(parents[rid])
                        q["matched_queries"] = [f"(referenced by {p['id']})"]
                        q["score"], q["score_kind"] = None, "reference"
                        results.append(q); have.add(rid)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "search"])
    ap.add_argument("--out", default="./out")
    ap.add_argument("-q", action="append", default=[], help="query (repeat -q for several)")
    ap.add_argument("--per-query", type=int, default=3, help="sections kept from each query")
    ap.add_argument("--max-total", type=int, default=8, help="max sections returned")
    ap.add_argument("--expand-refs", action="store_true")
    ap.add_argument("--refs-from-top", type=int, default=2, help="expand references of the first N results only")
    ap.add_argument("--rerank", action="store_true", help="re-score with a cross-encoder (more precise)")
    a = ap.parse_args()

    if a.cmd == "index":
        index(a.out)
    else:
        for p in retrieve(a.q, a.out, per_query=a.per_query, max_total=a.max_total,
                          expand_refs=a.expand_refs, refs_from_top=a.refs_from_top, rerank=a.rerank):
            print(f"\n=== {p['id']} | {p['title'][:80]} ===")
            print(f"matched: {p['matched_queries']} | {p['score_kind']}: {p['score']}")
            print(p["text"][:300])