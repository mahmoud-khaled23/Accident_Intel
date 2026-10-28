"""Unified command-line runner for the Accident Intel evaluation framework.

Examples:
    # Run all feasible evaluation stages
    uv run python -m evaluation.runner

    # Run individual stage
    uv run python -m evaluation.runner --stage rag_retrieval
    uv run python -m evaluation.runner --stage vehicle_tracking
    uv run python -m evaluation.runner --stage damage_cost

    # Run with sample limit and specific output directory
    uv run python -m evaluation.runner --limit 10 --output-dir evaluation/results
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure HuggingFace models load directly from local cache without network lookup timeouts
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from .config import ALL_STAGES, EvaluationConfig, StageName
from .manifest import RunManifest
from .reporting.exporter import export_evaluation_artifacts
from .reporting.reporter import generate_markdown_report
from .reporting.visualizer import generate_evaluation_charts
from .stages import (
    StageResult,
    evaluate_accident_reasoning,
    evaluate_comparisons,
    evaluate_damage_cost,
    evaluate_end_to_end,
    evaluate_rag_generation,
    evaluate_rag_retrieval,
    evaluate_vehicle_tracking,
    evaluate_video_understanding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Accident Intel evaluation framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["all"] + ALL_STAGES,
        default="all",
        help="Evaluation stage to execute, or 'all'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Sample limit per stage (for fast debugging/dry runs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/results",
        help="Base directory for saved runs and latest results",
    )
    parser.add_argument(
        "--allow-paid-api-calls",
        action="store_true",
        help="Explicitly enable external paid model API evaluations if keys are configured",
    )
    parser.add_argument(
        "--rag-mode",
        choices=["bm25", "dense", "hybrid"],
        default="hybrid",
        help="Retrieval search mode for RAG evaluation",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable Cross-Encoder reranking for RAG evaluation (slow on CPU)",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Disable chart figure generation",
    )
    return parser.parse_args()


def run_evaluation_suite(config: EvaluationConfig, generate_charts: bool = True) -> tuple[RunManifest, dict[str, StageResult]]:
    """Execute the configured evaluation suite and return manifest and results."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n=======================================================")
    print(f" ACCIDENT INTEL EVALUATION RUN: {config.run_id}")
    print(f" Output directory: {config.run_dir}")
    print(f" Stages to execute: {', '.join(config.stages)}")
    print(f"=======================================================\n")

    manifest = RunManifest(run_id=config.run_id)

    # Register known datasets
    manifest.add_dataset("ragas_testset", "evaluation/datasets/rag/ragas_testset.jsonl", 54)
    manifest.add_dataset("video_benchmark", "evaluation/datasets/video/video_benchmark.json", 8)
    manifest.add_dataset("tracking_benchmark", "evaluation/datasets/tracking/tracking_benchmark.json", 30)
    manifest.add_dataset("reasoning_benchmark", "evaluation/datasets/reasoning/reasoning_benchmark.json", 12)
    manifest.add_dataset("damage_benchmark", "evaluation/datasets/damage/damage_benchmark.json", 10)

    stage_results: dict[str, StageResult] = {}
    stage_runners = {
        "video_understanding": evaluate_video_understanding,
        "vehicle_tracking": evaluate_vehicle_tracking,
        "rag_retrieval": evaluate_rag_retrieval,
        "rag_generation": evaluate_rag_generation,
        "accident_reasoning": evaluate_accident_reasoning,
        "damage_cost": evaluate_damage_cost,
        "end_to_end": evaluate_end_to_end,
        "comparisons": evaluate_comparisons,
    }

    start_suite_time = time.perf_counter()

    for stage_name in config.stages:
        runner_fn = stage_runners.get(stage_name)
        if not runner_fn:
            continue

        print(f"--> Executing stage: [{stage_name}] ...", flush=True)
        t0 = time.perf_counter()
        res = runner_fn(config)
        elapsed = time.perf_counter() - t0

        stage_results[stage_name] = res
        manifest.record_stage(
            stage=stage_name,
            status=res.status,
            sample_count=res.sample_count,
            duration_seconds=elapsed,
            reason=res.reason,
        )

        status_symbol = "PASS" if res.status == "SUCCESS" else (res.status if res.status in ("BLOCKED", "SKIPPED") else "FAIL")
        print(f"    [{status_symbol}] Status: {res.status} | Samples: {res.sample_count} | Duration: {elapsed:.2f}s")
        if res.reason:
            print(f"        Info: {res.reason}")

    total_suite_elapsed = time.perf_counter() - start_suite_time
    print(f"\nAll stages completed in {total_suite_elapsed:.2f}s.")

    # 1. Export JSON, CSV, JSONL artifacts to run_dir and mirror to latest_dir
    print("--> Exporting artifacts (summary.json, metrics.csv, manifest.json, per_sample_results.jsonl)...")
    export_evaluation_artifacts(
        manifest=manifest,
        stage_results=stage_results,
        output_dir=config.run_dir,
        latest_dir=config.latest_dir,
    )

    # 2. Generate charts
    chart_paths = []
    if generate_charts:
        print("--> Generating evaluation charts...")
        res_dict = {k: v.to_dict() for k, v in stage_results.items()}
        chart_paths = generate_evaluation_charts(res_dict, config.run_dir)
        # Mirror charts to latest_dir
        if config.latest_dir:
            config.latest_dir.mkdir(parents=True, exist_ok=True)
            for cp in chart_paths:
                p = Path(cp)
                (config.latest_dir / p.name).write_bytes(p.read_bytes())
        print(f"    Generated {len(chart_paths)} charts.")

    # 3. Generate comprehensive markdown report
    print("--> Generating Markdown report (report.md)...")
    report_file_run = config.run_dir / "report.md"
    generate_markdown_report(manifest, stage_results, chart_paths, report_file_run)
    if config.latest_dir:
        (config.latest_dir / "report.md").write_text(report_file_run.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"\n[OK] Artifacts and report saved to:\n  - Run directory: {config.run_dir}\n  - Latest pointer: {config.latest_dir}\n")
    return manifest, stage_results


def main() -> None:
    args = parse_args()

    stages_to_run: list[StageName] = ALL_STAGES if args.stage == "all" else [args.stage]

    config = EvaluationConfig(
        stages=stages_to_run,
        sample_limit=args.limit,
        seed=args.seed,
        output_dir=args.output_dir,
        rag_mode=args.rag_mode,
        rag_rerank=args.rerank,
        allow_paid_api_calls=args.allow_paid_api_calls,
    )

    try:
        run_evaluation_suite(config, generate_charts=not args.no_charts)
    except Exception as e:
        print(f"\nFATAL: Evaluation runner failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
