# Accident Intel: Evaluation Framework & Results Sharing

Research-quality evaluation pipeline and results sharing framework for the **Accident Intel** multi-modal traffic collision and legal liability analysis system.

---

## 1. Overview & Architecture

The evaluation framework provides rigorous, automated, reproducible measurement of every component across the Accident Intel processing pipeline:

```
[ Accident Video ]
        │
        ├──> [ Video Understanding ] ──────> Events, Timeline, Collision, Evidence Compliance
        │
        ├──> [ Vehicle Tracking ]    ──────> Detections, IoU, Track ID Switches, FPS Throughput
        │
        ├──> [ Traffic-Law RAG ]     ──────> Hybrid Vector+BM25, Recall@k, Precision@k, MRR, nDCG@k
        │
        ├──> [ Accident Reasoning ]  ──────> Fault Level Classification, Statutory Citations, Insufficient Evidence Refusal
        │
        ├──> [ Damage & Repair Cost ]──────> Mask IoU/Dice, MAE, RMSE, MAPE, Spread Capture Rate
        │
        └──> [ End-to-End API ]      ──────> Integration Latencies (p50/p95), Contract Validation, Error Propagation
```

### Key Principles

1. **Empirical Honesty**: Metrics originate strictly from executed runs against ground-truth benchmarks. No fabricated baselines or synthetic numbers.
2. **Transparent Status Tracking**:
   - `SUCCESS`: Complete, verified execution against benchmark dataset.
   - `BLOCKED`: Explicitly identifies missing external dependencies (e.g., hosted LLM provider API keys such as `GROQ_API_KEY` or `OPENAI_API_KEY`) with clear reproduction steps.
   - `SKIPPED`: Configured out or empty dataset.
   - `FAILED`: Execution exception with preserved stack trace.
3. **Decoupled RAG Evaluation**: Retrieval evaluation (Recall@k, MRR, nDCG) executes locally and deterministically using dense embeddings (`BAAI/bge-m3`) and BM25 without requiring paid API credits. End-to-end generation is independently isolated.
4. **Reproducible Artifacts**: Every execution generates a timestamped run directory containing `manifest.json`, `summary.json`, `metrics.csv`, `per_sample_results.jsonl`, `report.md`, and high-resolution visualization charts.

---

## 2. Quickstart

### Run Complete Evaluation Suite

To run all feasible stages with default parameters and generate reports and charts:

```bash
uv run python -m evaluation.runner
```

### Run Specific Stages

```bash
# Evaluate Traffic-Law Retrieval only
uv run python -m evaluation.runner --stage rag_retrieval

# Evaluate Vehicle Tracking only
uv run python -m evaluation.runner --stage vehicle_tracking

# Evaluate Vehicle Damage & Repair Cost only
uv run python -m evaluation.runner --stage damage_cost

# Evaluate Accident Legal Reasoning only
uv run python -m evaluation.runner --stage accident_reasoning

# Run Fair Model & Configuration Comparisons
uv run python -m evaluation.runner --stage comparisons
```

### Command-Line Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--stage` | string | `all` | Specific stage to execute (`video_understanding`, `vehicle_tracking`, `rag_retrieval`, `rag_generation`, `accident_reasoning`, `damage_cost`, `end_to_end`, `comparisons`, or `all`). |
| `--limit` | int | `None` | Max samples to evaluate per stage (useful for fast smoke tests). |
| `--seed` | int | `42` | Random seed for reproducibility. |
| `--output-dir` | path | `evaluation/results` | Root output directory for run records and artifacts. |
| `--rag-mode` | choice | `hybrid` | Retrieval mode (`bm25`, `dense`, or `hybrid`). |
| `--rerank` | flag | `False` | Opt into Cross-Encoder precision reranker (`BAAI/bge-reranker-v2-m3`). |
| `--no-charts` | flag | `False` | Disable generation of matplotlib PNG figures. |
| `--allow-paid-api-calls` | flag | `False` | Explicitly permit external paid API calls if credentials exist. |

---

## 3. Benchmarks & Labeled Datasets

All evaluation datasets live under `evaluation/datasets/` and are validated by `evaluation/validators.py` before execution:

| Dataset | Path | Format | Samples | Description |
|---|---|---|---|---|
| **Traffic-Law RAG** | `evaluation/datasets/rag/ragas_testset.jsonl` | JSONL | 54 | Real NYS VTL legal queries, gold parent/child context IDs, reference answers, and unanswerable negative trap queries. |
| **Video Understanding** | `evaluation/datasets/video/video_benchmark.json` | JSON | 8 | Multi-vehicle accident scenarios with ground-truth timelines, collision timestamps, visible object rosters, and negative hallucination traps. |
| **Vehicle Tracking** | `evaluation/datasets/tracking/tracking_benchmark.json` | JSON | 30 frames | Sequential video sequence with bounding box coordinates, track IDs, vehicle classes, and identity continuity. |
| **Accident Reasoning** | `evaluation/datasets/reasoning/reasoning_benchmark.json` | JSON | 12 | Complex collision scenarios covering sole fault, shared responsibility, right-of-way disputes, and insufficient-evidence cases. |
| **Damage & Cost** | `evaluation/datasets/damage/damage_benchmark.json` | JSON | 10 | Damaged vehicles across economy, mid, and luxury tiers with ground truth repair actions and reference repair costs in Egyptian Pounds (EGP). |

---

## 4. Evaluated Pipeline Stages & Metrics

### Stage 1: Video Understanding & Visual Grounding
- **Mean Event F1**: Harmonic mean of detected vs. ground-truth visual event precision and recall.
- **Temporal Ordering Accuracy**: Pairwise concordance of event sequence timing.
- **Collision Detection Accuracy**: Binary collision detection and timing tolerance error ($\le 2.0\text{s}$).
- **Hallucination Rate**: Fraction of negative trap assertions falsely claimed by the model.
- **Evidence Compliance Score**: Composite adherence to evidence-only constraints.

### Stage 2: Vehicle Detection & Tracking
- **Detection Precision, Recall, F1**: Object-level vehicle detection overlap at $\text{IoU} \ge 0.5$.
- **Mean IoU**: Average bounding-box intersection over union on matched vehicle detections.
- **Identity Switches (IDSW)**: Frequency of track ID re-assignment on continuous trajectories.
- **Throughput (FPS) & Latency**: Real inference speed on hardware platform.

### Stage 3: Traffic-Law RAG Retrieval
- **Recall@k ($k \in \{1, 3, 5, 10\}$)**: Percentage of gold legal contexts retrieved in top-$k$ results.
- **Precision@k**: Density of relevant legal authorities among the top-$k$ candidates.
- **Hit Rate@k**: Probability that at least one relevant statutory section appears in top-$k$.
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of the first relevant legal section:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
- **nDCG@k**: Normalized Discounted Cumulative Gain accounting for graded relevance positioning.
- **Negative Rejection Accuracy**: Handling of unanswerable queries outside New York State traffic jurisdiction.

### Stage 4: RAG Generation (LLM Synthesis)
- **Status Isolation**: When `GROQ_API_KEY` or `OPENAI_API_KEY` is not exported, cleanly reported as `BLOCKED` with reproduction details. Retrieval remains fully evaluable.

### Stage 5: Accident Reasoning & Statutory Assessment
- **Schema Validity**: Adherence to the structured JSON decision contract.
- **Legal Responsibility Classification**: Agreement with expert ground truth across `potential_responsibility`, `shared_responsibility`, and `insufficient_evidence`.
- **Citation Precision, Recall, F1**: Accuracy of cited NYS VTL statutory sections.
- **Insufficient Evidence Refusal Compliance**: Strict refusal to assign liability when camera angles or evidence are ambiguous.

### Stage 6: Vehicle Damage & Repair-Cost Estimation
- **Segmentation IoU & Dice**: Pixel-level mask overlap against vehicle part boundaries.
- **Cost Estimation MAE & RMSE**: Mean Absolute Error and Root Mean Squared Error against market repair prices in EGP.
- **Percentage Error (MAPE)**: Relative error normalized by vehicle repair baseline.
- **Spread Capture Rate**: Percentage of reference costs falling within the predicted `[min_cost, max_cost]` interval.
- **Repair Action Accuracy**: Classification agreement (`repair`, `paint`, `replace`).

### Stage 7: End-to-End Pipeline & API Service
- **Service Latency (p50 & p95)**: Real HTTP transaction latency for `/health`, `/api/laws`, and `/api/cost/analysis`.
- **Validation Pass Rate**: HTTP 400/415/422 rejection of corrupt video streams, empty files, and invalid formats.
- **Contract Integrity**: Response schema conformance for all public endpoints.

### Stage 8: Fair Model & Configuration Comparisons
- **BM25 vs. Hybrid (BM25 + Dense BGE-M3)**: Direct head-to-head on identical test queries.
- **Tracking Confidence Sweep**: Detection Precision/Recall/F1 tradeoff across confidence thresholds ($0.10$, $0.25$, $0.50$, $0.75$).
- **Deployment Architecture Comparison**: On-premise local inference vs. hosted API serving tradeoffs (latency, cost, throughput).

---

## 5. Generated Artifacts & Structure

Each evaluation run records all telemetry to `evaluation/results/runs/<run_id>/` and mirrors the latest state to `evaluation/results/latest/`:

```
evaluation/results/latest/
├── manifest.json              # Git hash, dirty flag, Python/OS env, dataset SHA-256 hashes
├── summary.json               # Structured stage results, aggregate metrics, failure cases
├── metrics.csv                # Machine-readable tabular dump of all stage metrics
├── per_sample_results.jsonl   # Per-sample predictions, gold targets, latencies, and metadata
├── report.md                  # Comprehensive 14-section research evaluation report
├── chart_rag_metrics.png      # High-res curve of Recall@k, Precision@k, HitRate@k
├── chart_retrieval_comparison.png # Bar comparison: BM25 vs. Hybrid (MRR, Recall@5, Latency)
├── chart_tracking_prf.png     # Tracking Precision, Recall, F1 sweep across confidence thresholds
└── chart_stage_latency.png    # End-to-end latency waterfall across pipeline stages
```

---

## 6. Results Sharing & Web Dashboard

### Frontend Interactive Dashboard

Navigate to `/evaluation` in the React frontend:

- **Executive Summary Cards**: Instant view of RAG Recall@5, Tracking F1, Mean Mask IoU, and API p95 Latency.
- **Stage Audit Table**: Pass/Blocked badges, sample counts, stage runtimes, and key findings.
- **Comparative Analysis**: Visual breakdown of BM25 vs. Hybrid RRF retrieval and tracking sweeps.
- **Artifact Downloads**: One-click download of `summary.json`, `metrics.csv`, `manifest.json`, and `report.md`.

### Backend API Endpoints

The FastAPI backend exposes dedicated evaluation endpoints under `/api/evaluation`:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/evaluation/latest` | Returns `summary.json` and `manifest.json` for the most recent run. |
| `GET` | `/api/evaluation/history` | Lists previous evaluation runs with timestamps and stage statuses. |
| `GET` | `/api/evaluation/artifacts/{filename}` | Serves artifacts (`report.md`, `metrics.csv`, PNG charts). |

---

## 7. Automated Testing & Verification

Run the test suite to verify all metrics, dataset validators, schemas, and API routers:

```bash
uv run python -m pytest tests/test_evaluation_framework.py -v
```

