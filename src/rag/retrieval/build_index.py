#!/usr/bin/env python3
"""
build_index.py - embed title_7_clean.jsonl and store it in a local Chroma DB.

Install once:
    pip install chromadb sentence-transformers

Build the index (run once):
    python build_index.py build --data title_7_clean.jsonl --db vtl_db

Try a search:
    python build_index.py search "vehicle entered intersection facing steady red signal"
    python build_index.py search "changed lane without signaling" --all-types
"""
import argparse
import json

import chromadb

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"   # English legal text; ~440 MB, fine on CPU
COLLECTION = "vtl"
# bge models work better when the *query* (not the documents) has this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def clean_meta(meta: dict) -> dict:
    """Chroma metadata must be str/int/float/bool: no None, no lists."""
    out = {}
    for k, v in meta.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, list):
            out[k] = ",".join(v)
        else:
            out[k] = v
    return out


def build(data, db, model_name, batch=64, encode=None):
    rows = load_rows(data)
    if encode is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        encode = lambda texts: model.encode(
            texts, batch_size=batch, normalize_embeddings=True,
            show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=db)
    try:
        client.delete_collection(COLLECTION)          # rebuild from scratch
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(rows), 256):
        part = rows[i:i + 256]
        col.add(
            ids=[r["id"] for r in part],
            documents=[r["text"] for r in part],                  # shown to the LLM
            embeddings=encode([r["embed_text"] for r in part]),   # what is searched
            metadatas=[clean_meta(r["metadata"]) for r in part],
        )
    print(f"indexed {col.count()} chunks into '{db}'")
    return col


def search(query, db, model_name, k=5, doc_type="substantive_rule", encode=None):
    if encode is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        encode = lambda texts: model.encode(texts, normalize_embeddings=True).tolist()

    col = chromadb.PersistentClient(path=db).get_collection(COLLECTION)
    where = {"doc_type": doc_type} if doc_type else None
    res = col.query(query_embeddings=encode([QUERY_PREFIX + query]),
                    n_results=k, where=where)
    hits = []
    for i in range(len(res["ids"][0])):
        m = res["metadatas"][0][i]
        hits.append({
            "id": res["ids"][0][i],
            "citation": m["citation"],
            "title": m["section_title"],
            "score": 1 - res["distances"][0][i],      # cosine similarity
            "references": [r for r in m["references"].split(",") if r],
            "text": res["documents"][0][i],
        })
    return hits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--data", default="title_7_clean.jsonl")
    b.add_argument("--db", default="vtl_db")
    b.add_argument("--model", default=DEFAULT_MODEL)

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--db", default="vtl_db")
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--all-types", action="store_true",
                   help="also search camera/enforcement and procedure sections")

    a = ap.parse_args()
    if a.cmd == "build":
        build(a.data, a.db, a.model)
    else:
        for h in search(a.query, a.db, a.model, a.k,
                        doc_type=None if a.all_types else "substantive_rule"):
            print(f"{h['score']:.3f}  {h['citation']}  -  {h['title']}")
            print("       ", h["text"][:160].replace("\n", " "), "...\n")
