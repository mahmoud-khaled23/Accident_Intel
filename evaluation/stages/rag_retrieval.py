"""RAG retrieval stage evaluation: Recall@k, Precision@k, MRR, nDCG@k."""
from __future__ import annotations

import statistics
import time
from typing import Any

from src.rag.retrieval.index_and_retrieve import retrieve
from ..config import EvaluationConfig
from ..datasets.loader import load_rag_testset
from ..metrics.retrieval import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .base import BaseStageEvaluator, StageResult


def _normalize_id(id_str: str) -> str:
    """Normalize child chunk id (e.g. 'VTL-1111#d.1') or parent id to canonical section id ('VTL-1111')."""
    return id_str.split("#")[0].strip().upper()


class RagRetrievalEvaluator(BaseStageEvaluator):
    stage_name = "rag_retrieval"

    def run(self, config: EvaluationConfig) -> StageResult:
        testset = load_rag_testset(limit=config.sample_limit)
        if not testset:
            return StageResult(
                stage_name=self.stage_name,
                status="SKIPPED",
                reason="RAG testset is empty or could not be loaded",
            )

        k_values = sorted(config.retrieval_k_values)
        max_k = max(k_values) if k_values else 10
        corpus_out = "data/traffic_laws/chunks"

        per_sample: list[dict[str, Any]] = []
        failure_cases: list[dict[str, Any]] = []
        latencies_ms: list[float] = []

        recalls: dict[int, list[float]] = {k: [] for k in k_values}
        precisions: dict[int, list[float]] = {k: [] for k in k_values}
        hit_rates: dict[int, list[float]] = {k: [] for k in k_values}
        ndcgs: dict[int, list[float]] = {k: [] for k in k_values}
        mrrs: list[float] = []

        # Breakdown buckets
        by_difficulty: dict[str, dict[str, list[float]]] = {}
        by_query_type: dict[str, dict[str, list[float]]] = {}

        for idx, item in enumerate(testset):
            question = item["user_input"]
            raw_gold_ids = item.get("reference_context_ids", [])
            gold_ids = list(dict.fromkeys([_normalize_id(g) for g in raw_gold_ids if g]))
            diff = item.get("difficulty", "medium")
            q_type = item.get("query_type", "single_hop")

            # Measure retrieval latency
            t0 = time.perf_counter()
            try:
                retrieved_chunks = retrieve(
                    queries=[question],
                    out=corpus_out,
                    top_k=max_k,
                    mode=config.rag_mode,
                    rerank=config.rag_rerank,
                )
                retrieval_time_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(retrieval_time_ms)
                retrieved_ids = [_normalize_id(c["id"]) for c in retrieved_chunks]
            except Exception as e:
                retrieval_time_ms = (time.perf_counter() - t0) * 1000.0
                retrieved_ids = []
                failure_cases.append({
                    "sample_idx": idx,
                    "question": question,
                    "error": str(e),
                })

            # Unanswerable query handling (negative rejection test)
            if not gold_ids or q_type == "unanswerable":
                per_sample.append({
                    "sample_idx": idx,
                    "question": question,
                    "gold_ids": [],
                    "retrieved_ids": retrieved_ids[:max_k],
                    "query_type": "unanswerable",
                    "latency_ms": round(retrieval_time_ms, 2),
                    "note": "Negative query: no relevant VTL statute exists",
                })
                continue

            # Calculate metrics for answerable queries
            mrr_val = mean_reciprocal_rank(retrieved_ids, gold_ids)
            mrrs.append(mrr_val)

            sample_metrics = {
                "sample_idx": idx,
                "question": question,
                "gold_ids": gold_ids,
                "retrieved_ids": retrieved_ids[:max_k],
                "mrr": round(mrr_val, 4),
                "latency_ms": round(retrieval_time_ms, 2),
                "difficulty": diff,
                "query_type": q_type,
            }

            for k in k_values:
                r_k = recall_at_k(retrieved_ids, gold_ids, k)
                p_k = precision_at_k(retrieved_ids, gold_ids, k)
                h_k = hit_rate_at_k(retrieved_ids, gold_ids, k)
                n_k = ndcg_at_k(retrieved_ids, gold_ids, k)

                recalls[k].append(r_k)
                precisions[k].append(p_k)
                hit_rates[k].append(h_k)
                ndcgs[k].append(n_k)

                sample_metrics[f"recall@{k}"] = round(r_k, 4)
                sample_metrics[f"precision@{k}"] = round(p_k, 4)
                sample_metrics[f"hit_rate@{k}"] = round(h_k, 4)
                sample_metrics[f"ndcg@{k}"] = round(n_k, 4)

            # Record breakdown
            for group_dict, key in [(by_difficulty, diff), (by_query_type, q_type)]:
                if key not in group_dict:
                    group_dict[key] = {"mrr": [], "recall@5": []}
                group_dict[key]["mrr"].append(mrr_val)
                group_dict[key]["recall@5"].append(sample_metrics.get("recall@5", 0.0))

            per_sample.append(sample_metrics)

            # Check for failure (e.g. Recall@5 == 0.0)
            if sample_metrics.get("recall@5", 0.0) == 0.0:
                failure_cases.append({
                    "sample_idx": idx,
                    "question": question,
                    "gold_ids": gold_ids,
                    "retrieved_ids": retrieved_ids[:5],
                    "reason": "Gold document not retrieved in top-5",
                })

        # Aggregate metrics
        num_samples = len(per_sample)
        agg_metrics: dict[str, Any] = {
            "sample_count": num_samples,
            "mrr": round(statistics.mean(mrrs), 4) if mrrs else 0.0,
            "latency_p50_ms": round(statistics.median(latencies_ms), 2) if latencies_ms else 0.0,
            "latency_p95_ms": round(statistics.quantiles(latencies_ms, n=20)[18], 2) if len(latencies_ms) >= 20 else (round(max(latencies_ms), 2) if latencies_ms else 0.0),
            "latency_mean_ms": round(statistics.mean(latencies_ms), 2) if latencies_ms else 0.0,
        }

        for k in k_values:
            agg_metrics[f"recall@{k}"] = round(statistics.mean(recalls[k]), 4) if recalls[k] else 0.0
            agg_metrics[f"precision@{k}"] = round(statistics.mean(precisions[k]), 4) if precisions[k] else 0.0
            agg_metrics[f"hit_rate@{k}"] = round(statistics.mean(hit_rates[k]), 4) if hit_rates[k] else 0.0
            agg_metrics[f"ndcg@{k}"] = round(statistics.mean(ndcgs[k]), 4) if ndcgs[k] else 0.0

        # Sub-group summaries
        breakdowns = {
            "by_difficulty": {
                k: {
                    "count": len(v["mrr"]),
                    "mrr": round(statistics.mean(v["mrr"]), 4),
                    "recall@5": round(statistics.mean(v["recall@5"]), 4),
                }
                for k, v in by_difficulty.items()
            },
            "by_query_type": {
                k: {
                    "count": len(v["mrr"]),
                    "mrr": round(statistics.mean(v["mrr"]), 4),
                    "recall@5": round(statistics.mean(v["recall@5"]), 4),
                }
                for k, v in by_query_type.items()
            },
        }

        return StageResult(
            stage_name=self.stage_name,
            status="SUCCESS",
            sample_count=num_samples,
            metrics=agg_metrics,
            per_sample_results=per_sample,
            failure_cases=failure_cases,
            metadata={
                "rag_mode": config.rag_mode,
                "rag_rerank": config.rag_rerank,
                "k_values": k_values,
                "breakdowns": breakdowns,
            },
        )


def evaluate_rag_retrieval(config: EvaluationConfig) -> StageResult:
    return RagRetrievalEvaluator().execute(config)
