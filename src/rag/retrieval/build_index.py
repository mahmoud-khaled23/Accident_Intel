#!/usr/bin/env python3
"""Embed a JSONL law corpus and store it in a local ChromaDB vector database.

Provides two operations accessible both programmatically and via the CLI:

- :func:`build` — loads ``title_7_clean.jsonl``, encodes all records with a
  sentence-transformer model, and upserts them into a named Chroma collection.
- :func:`search` — queries the collection for the most similar sections given
  a free-text query string.

Utility: :func:`clean_meta` — sanitises metadata dicts to the scalar types
accepted by Chroma (no ``None`` values or lists).

Install once:
    uv sync --locked

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
    """Load a JSONL file into a list of Python dicts.

    Args:
        path: Path (str or Path-like) to a ``.jsonl`` file where each
            non-empty line is a JSON object.

    Returns:
        list[dict]: One dict per non-empty line.

    Example:
        >>> rows = load_rows("title_7_clean.jsonl")
        >>> rows[0]["id"]
        'VTL-1110-chunk-0'
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def clean_meta(meta: dict) -> dict:
    """Sanitise a metadata dict to the scalar types accepted by ChromaDB.

    ChromaDB metadata fields must be ``str``, ``int``, ``float``, or ``bool``.
    This function converts ``None`` values to empty strings and flattens
    ``list`` values to comma-joined strings.

    Args:
        meta: A raw metadata dict that may contain ``None`` or ``list`` values.

    Returns:
        dict: A copy of ``meta`` with all values converted to Chroma-compatible
        scalar types.

    Example:
        >>> clean_meta({"references": ["§1111", "§1120"], "title": None})
        {'references': '§1111,§1120', 'title': ''}
    """
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
    """Embed all records from a JSONL file and store them in a ChromaDB collection.

    Deletes and recreates the collection named :data:`COLLECTION` to ensure a
    clean rebuild.  Records are embedded in batches and upserted via
    ``col.add``.  The ``"embed_text"`` field is used for embedding (which may
    differ from the ``"text"`` shown to the LLM).

    Args:
        data: Path to the source ``.jsonl`` file.
        db: Directory path where the persistent ChromaDB will be stored.
        model_name: HuggingFace sentence-transformer model identifier used to
            compute embeddings (e.g. ``"BAAI/bge-base-en-v1.5"``).
        batch: Embedding batch size.  Defaults to ``64``.
        encode: Optional callable ``(list[str]) -> list[list[float]]`` that
            overrides the default sentence-transformer encoding.  Useful for
            testing or alternative embedding backends.

    Returns:
        chromadb.Collection: The populated ChromaDB collection.

    Example:
        >>> col = build("title_7_clean.jsonl", "vtl_db", "BAAI/bge-base-en-v1.5")
        >>> col.count()
        1234
    """
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
    """Perform a semantic similarity search against the ChromaDB collection.

    Encodes the query string with :data:`QUERY_PREFIX` prepended (as
    recommended by BGE models for asymmetric retrieval), then queries the
    collection for the ``k`` nearest neighbours filtered by ``doc_type``.

    Args:
        query: Free-text search query in natural English.
        db: Directory path of the persistent ChromaDB to search.
        model_name: Sentence-transformer model identifier matching the one
            used when the index was built.
        k: Number of results to return.  Defaults to ``5``.
        doc_type: Chroma metadata filter on the ``doc_type`` field.
            Pass ``None`` to search across all document types.
            Defaults to ``"substantive_rule"``.
        encode: Optional callable ``(list[str]) -> list[list[float]]`` that
            overrides the default encoding, useful for testing.

    Returns:
        list[dict]: Up to ``k`` result dicts, each containing:
            - ``"id"`` (str): The chunk identifier.
            - ``"citation"`` (str): Human-readable citation (e.g. ``"VTL § 1111"``).
            - ``"title"`` (str): Section title.
            - ``"score"`` (float): Cosine similarity (``1 - distance``), higher is better.
            - ``"references"`` (list[str]): Cross-references from the metadata.
            - ``"text"`` (str): The chunk text shown to the LLM.

    Example:
        >>> hits = search("ran a red light", "vtl_db", "BAAI/bge-base-en-v1.5", k=3)
        >>> hits[0]["citation"]
        'VTL § 1111(d)'
    """
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
