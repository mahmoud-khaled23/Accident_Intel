"""
Evaluate the Accident Intel RAG pipeline with RAGAS.

Judge LLM  : Groq (free tier)      – scores the pipeline
Generator  : Groq (free tier)      – produces the answer shown to RAGAS
Embeddings : BAAI/bge-m3 (local GPU) – same model as the retriever

Free Groq API key : https://console.groq.com  (no credit card)
Free models list  : https://console.groq.com/docs/models

Setup:
    pip install ragas langchain-openai langchain-community \
                sentence-transformers chromadb rank-bm25 rapidfuzz openai

How to run (from the project root):
    python -m src.rag.evaluate.ragas
    python -m src.rag.evaluate.ragas --out src/rag/chunking/out --testset src/rag/evaluate/ragas_testset.jsonl
    python -m src.rag.evaluate.ragas --retrieve-k 20 --final-k 5 --trace
"""
import argparse
import json
import sys
import time
from pathlib import Path
import dotenv
import os

dotenv.load_dotenv()  # load GROQ_API_KEY from .env if present

# ===========================================================================
# 🔑  PUT YOUR FREE GROQ API KEY HERE
#     Get one at: https://console.groq.com  (no credit card needed)
# ===========================================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Free models on Groq (as of Oct 2025) — pick one:
#   "llama-3.1-8b-instant"          ← fastest  (8B)
#   "gemma2-9b-it"                  ← good quality (9B, Google)
# Note: llama-3.3-70b-versatile requires Enterprise plan
GROQ_MODEL = "qwen/qwen3.8-27b"

# Separate model for RAGAS judge (needs stronger reasoning for claim comparison)
# Using same model for now — change if factual_correctness stays 0.0
GROQ_JUDGE_MODEL = "openai/gpt-oss-120b"

# Groq base URL — fully OpenAI-compatible
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# The same embedding model the retriever uses — keeps evaluation
# in the same vector space as retrieval.
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# ===========================================================================

# ------------------------------------------------------------------ paths
# __file__ = src/rag/evaluate/ragas.py
# .parent.parent.parent        → src/
# .parent.parent.parent.parent → project root
_PROJECT_ROOT   = Path(__file__).parent.parent.parent.parent
DEFAULT_OUT     = str(_PROJECT_ROOT / "src" / "rag" / "chunking" / "out")
DEFAULT_TESTSET = str(Path(__file__).parent / "ragas_testset.jsonl")

# ------------------------------------------------------------------ retriever
# Point to src/ so `rag` is a proper package and relative imports inside
# index_and_retrieve.py (from ..chunking.build_chunks) work correctly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # → src/
from rag.retrieval.index_and_retrieve import retrieve          # noqa: E402
from rag.chunking.query_expansion import expand_queries        # noqa: E402  (for --trace display)


# ------------------------------------------------------------------ children cache
_CHILDREN_CACHE: dict = {}  # out_path -> {parent_id: [child_ids]}
_CHILDREN_TEXT_CACHE: dict = {}  # out_path -> {child_id: child_dict}


def _load_parent_to_children(out: str) -> dict:
    """Load children.jsonl once and return {parent_id: [child_id, ...]} mapping."""
    key = str(Path(out).resolve())
    if key not in _CHILDREN_CACHE:
        mapping: dict = {}
        children_path = Path(out) / "children.jsonl"
        for line in open(children_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            child = json.loads(line)
            pid = child["parent_id"]
            mapping.setdefault(pid, []).append(child["id"])
        _CHILDREN_CACHE[key] = mapping
    return _CHILDREN_CACHE[key]


def _load_children_lookup(out: str) -> dict:
    """Load children.jsonl once and return {child_id: child_dict} mapping."""
    key = str(Path(out).resolve())
    if key not in _CHILDREN_TEXT_CACHE:
        lookup: dict = {}
        children_path = Path(out) / "children.jsonl"
        for line in open(children_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            child = json.loads(line)
            lookup[child["id"]] = child
        _CHILDREN_TEXT_CACHE[key] = lookup
    return _CHILDREN_TEXT_CACHE[key]


# ------------------------------------------------------------------ retrieval
def _retrieve_contexts(
    question: str,
    out: str,
    retrieve_k: int = 20,
    final_k: int = 5,
    trace: bool = False,
    question_index: int = 0,
    reference_ids: list = None,
    # Legacy parameter kept for backwards compatibility
    top_k: int = None,
) -> tuple:
    """
    Return (ctx_texts, ctx_ids) using the production pipeline:
      expand → retrieve retrieve_k → rerank (if final_k < retrieve_k) → slice to final_k
      → expand each parent to its child chunks.

    Both ctx_texts[i] and ctx_ids[i] refer to the SAME child chunk (e.g. VTL-1111#d.1),
    so the returned texts and IDs are consistent with each other and with
    reference_context_ids / reference_contexts in the testset (which use child chunks).

    This replaces the old approach that returned parent texts but child IDs, which caused
    NonLLMContextRecall=0.0 (text mismatch) while IDBasedContextRecall=1.0 (ID inflation).
    """
    # Handle legacy top_k parameter
    if top_k is not None:
        retrieve_k = top_k
        final_k = top_k

    do_rerank = final_k < retrieve_k

    # ── retrieval (top retrieve_k, with optional reranking inside retrieve()) ─
    results = retrieve(
        queries=[question],
        out=out,
        top_k=retrieve_k,
        mode="hybrid",   # dense (bge-m3) + BM25 fused with RRF
        expand="rules",  # light legal query normalisation
        rerank=do_rerank,
    )

    # ── optional trace: show all retrieve_k results ───────────────────────
    # NOTE: when do_rerank=True, retrieve() already returns reranked results.
    # There is no pre-rerank list exposed by retrieve(), so we show the final
    # ranked order as returned, with an accurate label.
    if trace:
        expanded = expand_queries([question], method="rules")
        print(f"\n{'=' * 60}")
        print(f"TRACE: question {question_index + 1}")
        print(f"{'=' * 60}")
        print(f"QUESTION: {question}")
        print(f"EXPANDED QUERIES: {expanded}")
        if do_rerank:
            print(f"\nRETRIEVAL + RERANK (top-{retrieve_k} retrieved and reranked, showing final ranked order):")
        else:
            print(f"\nRETRIEVAL (top-{retrieve_k}):")
        for rank, p in enumerate(results, 1):
            preview = p["text"].replace("\n", " ")[:120]
            print(f"  #{rank:<3} {p['id']:<16}  score={p.get('score', '?')!s:<8}  "
                  f"[{p.get('title', '')[:50]}]  \"{preview}…\"")

    # ── slice to final_k ──────────────────────────────────────────────────
    final_results = results[:final_k]

    # ── trace: final top-N parents before child expansion ─────────────────
    if trace:
        print(f"\nFINAL TOP-{final_k} PARENTS (before child expansion):")
        for rank, p in enumerate(final_results, 1):
            preview = p["text"].replace("\n", " ")[:120]
            print(f"  #{rank:<3} {p['id']:<16}  score={p.get('score', '?')!s:<8}  "
                  f"[{p.get('title', '')[:50]}]  \"{preview}…\"")

    # ── expand parent → children, keeping text and ID aligned ─────────────
    # Each child gets its own text entry so ctx_texts[i] and ctx_ids[i]
    # always refer to the same chunk. This fixes the old mismatch where
    # parent texts were paired with child IDs.
    parent_to_children = _load_parent_to_children(out)
    child_lookup = _load_children_lookup(out)
    ctx_texts: list = []
    ctx_ids: list = []
    for p in final_results:
        child_ids = parent_to_children.get(p["id"], [])
        if child_ids:
            for cid in child_ids:
                if cid in child_lookup:
                    ctx_texts.append(child_lookup[cid]["text"])
                    ctx_ids.append(cid)
                # else: child ID recorded but text not found — skip silently
        else:
            # No children for this parent — use parent text and parent ID as-is
            ctx_texts.append(p["text"])
            ctx_ids.append(p["id"])

    # ── alignment guard ───────────────────────────────────────────────────
    assert len(ctx_texts) == len(ctx_ids), (
        f"ALIGNMENT BUG: {len(ctx_texts)} texts vs {len(ctx_ids)} IDs "
        f"for question: {question[:60]!r}"
    )

    # ── trace: context/ID alignment after child expansion ─────────────────
    if trace:
        print(f"\nCONTEXT/ID ALIGNMENT (after child expansion, {len(ctx_ids)} child chunks total):")
        for i, (cid, txt) in enumerate(zip(ctx_ids, ctx_texts), 1):
            parent_id = cid.split("#")[0] if "#" in cid else cid
            source = "child" if "#" in cid else "parent"
            preview = txt.replace("\n", " ")[:100]
            print(f"  [{i}] parent={parent_id:<16}  child_id={cid:<24}  source={source}  "
                  f"preview=\"{preview}…\"")

        ref_ids = reference_ids or []
        matched = len(set(ctx_ids) & set(ref_ids))
        print(f"\nREFERENCE IDs:  {ref_ids}")
        print(f"RETRIEVED IDs:  {ctx_ids}")
        print(f"ID MATCH: {matched}/{len(ref_ids)}  "
              f"{'✓' if matched == len(ref_ids) and ref_ids else ('✗' if ref_ids else '—')}")

    return ctx_texts, ctx_ids


# ------------------------------------------------------------------ answer generator

SYSTEM_PROMPT = (
    "You are a traffic-law assistant. "
    "Answer the user's question using ONLY the law excerpts provided. "
    "Be concise and cite the relevant VTL section number."
)


def _generate_answer(question: str, contexts: list, retries: int = 5) -> str:
    """
    Call Groq to generate a grounded answer from retrieved law contexts.
    Groq is fully OpenAI-compatible.
    Retries on 503 / 429 errors with exponential back-off.
    """
    from openai import OpenAI, InternalServerError, RateLimitError

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
    )
    ctx_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"LAW EXCERPTS:\n{ctx_block}\n\nQUESTION: {question}"},
    ]

    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0,
                messages=messages,
            )
            if resp.choices and resp.choices[0].message.content:
                return resp.choices[0].message.content.strip()
            wait = 2 ** attempt
            print(f"\n  ⚠ Empty response (attempt {attempt}/{retries}), retrying in {wait}s …")
            time.sleep(wait)
        except (InternalServerError, RateLimitError) as e:
            wait = 2 ** attempt
            print(f"\n  ⚠ Groq busy (attempt {attempt}/{retries}), retrying in {wait}s … ({e})")
            time.sleep(wait)

    raise RuntimeError(f"Groq failed after {retries} retries for: {question[:60]}")


# ------------------------------------------------------------------ bge-m3 embeddings for RAGAS
def _build_bge_embeddings():
    """
    Wrap BAAI/bge-m3 in a LangChain interface for RAGAS.
    Used by ResponseRelevancy — same vector space as the retriever.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import torch

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        from ragas.embeddings.base import LangchainEmbeddingsWrapper  # ragas 0.4 alternate path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Embeddings device: {device}")
    hf_emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(hf_emb)


# ------------------------------------------------------------------ Groq judge LLM for RAGAS
def _build_judge():
    """
    Build the Groq-backed RAGAS judge LLM.
    Uses ChatOpenAI pointed at Groq's endpoint — no extra packages needed.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    chat = ChatOpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        model=GROQ_JUDGE_MODEL,
        temperature=0,
        max_tokens=2048,   # prevent truncated JSON from causing NaN in LLM-judge metrics
    )
    return LangchainLLMWrapper(chat)


# ------------------------------------------------------------------ evaluation
def run_evaluation(
    testset_path: str,
    out: str,
    retrieve_k: int = 20,
    final_k: int = 5,
    trace: bool = False,
    # Legacy parameter kept for backwards compatibility
    top_k: int = None,
):
    """
    Full evaluation loop:
      1. Load ragas_testset.jsonl
      2. Run the RAG pipeline (retriever + Groq generator) on every question
      3. Score with RAGAS metrics (judge = Groq, embeddings = bge-m3)
      4. Save results to ragas_results.csv

    Metrics:
      LLMContextRecall               – did retrieval cover what the reference says?
      LLMContextPrecisionWithRef     – is retrieved context relevant and well ranked?
      NonLLMContextRecall            – text-overlap: retrieved vs gold reference contexts
      IDBasedContextRecall           – exact chunk-ID match vs gold IDs
      Faithfulness                   – is the answer grounded in context (no hallucination)?
      ResponseRelevancy              – does the answer address the question? (uses bge-m3)
      FactualCorrectness             – does the answer match the reference answer?
    """
    from ragas import EvaluationDataset, evaluate
    try:
        from ragas.metrics.collections import (          # ragas >= 0.4.x new path
            LLMContextRecall,
            LLMContextPrecisionWithReference,
            NonLLMContextRecall,
            IDBasedContextRecall,
            Faithfulness,
            ResponseRelevancy,
            FactualCorrectness,
        )
    except ImportError:
        from ragas.metrics import (                      # ragas < 0.4.x fallback
            LLMContextRecall,
            LLMContextPrecisionWithReference,
            NonLLMContextRecall,
            IDBasedContextRecall,
            Faithfulness,
            ResponseRelevancy,
            FactualCorrectness,
        )
        import warnings
        warnings.warn(
            "ragas.metrics.collections not found — falling back to ragas.metrics. "
            "LLM-judge metrics may produce NaN. Upgrade ragas to >= 0.4.",
            UserWarning, stacklevel=2,
        )

    # Handle legacy top_k parameter
    if top_k is not None:
        retrieve_k = top_k
        final_k = top_k

    # ── guard ──────────────────────────────────────────────────────────────
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        raise ValueError(
            "\n\nGroq API key not set!\n"
            "Edit ragas.py and replace 'YOUR_GROQ_API_KEY_HERE' with your key.\n"
            "Free key (no credit card): https://console.groq.com\n"
        )

    # ── 1) Load test set ───────────────────────────────────────────────────
    rows = [json.loads(line) for line in open(testset_path, encoding="utf-8")
            if line.strip()]
    print(f"Loaded {len(rows)} test cases from {testset_path}")
    print(f"Pipeline: retrieve_k={retrieve_k}, final_k={final_k}, "
          f"rerank={'yes' if final_k < retrieve_k else 'no'}, trace={trace}")

    # ── 2) Run the RAG pipeline on every question ──────────────────────────
    print("\nStep 1/3 — Retrieving contexts and generating answers …")
    samples = []
    for i, r in enumerate(rows, 1):
        print(f"  [{i:>2}/{len(rows)}] {r['user_input'][:70]} …")

        ctx_texts, ctx_ids = _retrieve_contexts(
            r["user_input"],
            out=out,
            retrieve_k=retrieve_k,
            final_k=final_k,
            trace=trace,
            question_index=i - 1,                        # i starts at 1 in the loop
            reference_ids=r.get("reference_context_ids", []),
        )

        response = (
            _generate_answer(r["user_input"], ctx_texts)
            if ctx_texts
            else "No relevant law sections were found."
        )

        samples.append({
            "user_input":            r["user_input"],
            "response":              response,
            "retrieved_contexts":    ctx_texts,
            "reference":             r["reference"],
            "reference_contexts":    r.get("reference_contexts", []),
            "retrieved_context_ids": ctx_ids,
            "reference_context_ids": r.get("reference_context_ids", []),
        })

    # ── 3) Build judge and embeddings ──────────────────────────────────────
    print("\nStep 2/3 — Loading judge (Groq) and embeddings (bge-m3) …")
    judge_llm = _build_judge()
    bge_emb   = _build_bge_embeddings()

    # ── 4) Evaluate ────────────────────────────────────────────────────────
    print("\nStep 3/3 — Running RAGAS evaluation …")
    result = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[
            LLMContextRecall(llm=judge_llm),
            NonLLMContextRecall(),
            IDBasedContextRecall(),
            Faithfulness(llm=judge_llm),
            FactualCorrectness(llm=judge_llm),
        ],
    )

    # ── 5) Print and save ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(result)

    df = result.to_pandas()
    df["query_type"] = [r.get("query_type", "") for r in rows]
    df["difficulty"]  = [r.get("difficulty",  "") for r in rows]
    df.to_csv("ragas_results.csv", index=False)
    print("\nDetailed results saved → ragas_results.csv")

    if df["query_type"].any():
        print("\nScores by query_type:")
        print(df.groupby("query_type").mean(numeric_only=True).to_string())

    if df["difficulty"].any():
        print("\nScores by difficulty:")
        print(df.groupby("difficulty").mean(numeric_only=True).to_string())

    return result


# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Evaluate the Accident Intel RAG pipeline with RAGAS + Groq"
    )
    ap.add_argument("--out",     default=DEFAULT_OUT,
                    help="chunking output dir (parents.json, chroma/, …)")
    ap.add_argument("--testset", default=DEFAULT_TESTSET,
                    help="path to ragas_testset.jsonl")
    ap.add_argument("--top-k",   type=int, default=5,
                    help="legacy: law sections retrieved per question (default: 5). "
                         "Prefer --retrieve-k / --final-k for the full pipeline.")
    ap.add_argument("--retrieve-k", type=int, default=None,
                    help="candidates retrieved before reranking (default: 20, or --top-k if supplied)")
    ap.add_argument("--final-k",   type=int, default=None,
                    help="sections passed to the LLM after reranking (default: 5, or --top-k if supplied)")
    ap.add_argument("--trace",     action="store_true",
                    help="print a per-question retrieval trace showing where evidence is gained/lost")
    a = ap.parse_args()

    # Backwards-compat: --top-k sets both if the new flags are absent
    retrieve_k = a.retrieve_k if a.retrieve_k is not None else (a.top_k if a.top_k != 5 else 20)
    final_k    = a.final_k    if a.final_k    is not None else (a.top_k if a.top_k != 5 else 5)

    run_evaluation(testset_path=a.testset, out=a.out,
                   retrieve_k=retrieve_k, final_k=final_k, trace=a.trace)
