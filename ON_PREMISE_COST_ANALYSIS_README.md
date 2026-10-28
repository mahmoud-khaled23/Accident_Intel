# Accident Intel: fully on-premise production cost analysis

**Prepared 25 October 2026.** Currency conversion used throughout: **US$1 = EGP 55.29**, the latest observed market close available at research time (23 October; 25 October was a Sunday). This is a comparison rate, not a bank settlement quote. Hardware and operating-cost figures are estimates unless marked as a sourced listing.

## 1. Executive summary and recommendation

For **1,000 one-minute videos per month**, one modern GPU should be sufficient for asynchronous processing, provided the service limits concurrent visual jobs and validates performance with representative footage. The larger risks are not raw GPU capacity: the repository currently has hosted AI paths, damage-model weights and licenses are unverified, its retrieval corpus appears to be New York traffic law, and its background processing is not yet a durable production queue.

I recommend **Option B: one RTX 5090 32 GB inference node plus separate NAS/backup**. Its planning CAPEX is **EGP 1.10 million (US$19,931)**. It can run a 7B/8B VLM and a quantized 14B reasoning LLM locally, sequentially. It is a **single-node service with backups, not high availability**.

At the baseline workload, estimated Option B TCO is:

- **3 years:** EGP 1.68m (US$30,349) cash; EGP 2.58m (US$46,629) including part-time operations labor.
- **5 years:** EGP 2.06m (US$37,294) cash; EGP 3.56m (US$64,451) including part-time operations labor.

These are planning totals, not supplier quotes. They assume no external AI inference, so API cost is **$0**. They exclude model training, commercial licensing quotes, legal validation, and a redundant second site.

The existing application is **not compliant with the no-API requirement as configured**. The video path can select a hosted VLM, legal reasoning calls an OpenAI-compatible API, and repair pricing can query DuckDuckGo/Gemini. A damage endpoint can also return sample fallback results. Those paths must be removed or disabled and replaced with explicit local failure states.

## 2. Assumptions and exclusions

The baseline is 1,000 videos/month, one minute each, 8 sampled frames/video, one principal analysis per video, damage estimation included, asynchronous jobs, 30-day rolling video retention, Egypt, and 100% local inference. This yields 8,000 sampled frames/month, 36,000 videos over three years, and 60,000 over five years. Eight frames is a **sizing assumption**, not a proven accuracy threshold.

Storage planning assumes 1080p H.264 input around 4 Mbps: approximately 30 MB/video, or 30 GB/month of original footage. Allow 1–2 TB usable for videos, derivatives, uploads, logs, database, staging, and growth. A separate 4-disk NAS in RAID6/SHR-2 gives roughly 24 TB usable with 12 TB disks. Backups should be sized for the actual protected dataset and retention policy.

Costs exclude dataset creation, damage-model training, court/legal validation, office rent, a dedicated engineer, and high-availability infrastructure. A 20% procurement shock is included in the conservative scenario. USD/EGP conversion is held constant only to make totals comparable; future Egyptian hardware purchases remain exposed to exchange-rate and import-price changes.

## 3. Existing architecture and model inventory

I inspected the repository’s README, `.env.example`, API settings/services, video analyzer, tracker, RAG retrieval, damage pipeline, and pricing code.

| Component | What the repository contains | Production implications |
|---|---|---|
| Ingestion, API, UI | FastAPI, React, SQLite, filesystem storage; analysis runs through in-process background work | Prototype structure. Production needs a durable queue, object storage, auth, database migrations, and separate workers. |
| Video decode | OpenCV and FFmpeg-compatible processing | CPU is adequate at this volume; NVDEC is optional. Pin and stage the decoder build locally. |
| Detection/tracking | `yolo11x.pt`, Ultralytics tracking with BoT-SORT and accident heuristics | Implemented, but weights are not in the repository. Verify the exact classes, weights, and commercial license. GPU is preferred for throughput; CPU is possible but slower. |
| Video VLM | `Qwen/Qwen2-VL-7B-Instruct`, Transformers, FP16, CUDA required; local loader exists | Local inference is implemented, but `VIDEO_VLM_API=auto` can select hosted inference. Force local mode and remove the hosted adapter/fallback. API env defaults to 4 frames; the CLI defaults to 32. |
| RAG | Local BAAI/bge-m3 dense retrieval, BM25/RRF, optional BGE reranker; embedding/reranker configured on CPU | Local retrieval is implemented. README describes New York/VTL laws; Egyptian law corpus and legal validity are unverified. |
| Liability reasoning | `src/reasoning/reasoning_accident.py` uses an OpenAI-compatible chat API | No local reasoning backend is implemented. Add one and require citations to retrieved law IDs. |
| Damage/parts | Two YOLO segmentation weight paths, damage-to-part mapping, fusion, heuristic severity | Pipeline code exists, but actual weights, class definitions, training data, performance, and rights are absent or unverified. |
| Repair pricing | Rule tables plus optional DuckDuckGo/Gemini price agent | The live price path uses the internet. The tables are described as placeholder prices. Replace with a locally maintained, dated Egypt-specific catalog. |

The current code therefore violates the no-API constraint in multiple places. The production version should have **no API fallback**: if a local model cannot load or infer, mark the job failed or “needs review.” Do not return fabricated sample results.

## 4. Workload sizing

No end-to-end benchmark for this repository was available. The following are **engineering estimates**, not measured benchmarks:

| Stage | Estimated GPU time per video | Notes |
|---|---:|---|
| YOLO tracking | 10–60 seconds | Assumes 5–10 FPS processing; actual source frame rate and resolution matter. |
| VLM, 8 sampled frames | 15–90 seconds | Sensitive to frame resolution, visual tokens, prompt length, runtime, and model. |
| Local text reasoning | 2–15 seconds | Assumes a 4-bit 14B model or smaller local model and modest context. |
| Damage segmentation | 1–10 seconds/photo | Two models; excludes human review. The repo does not establish automated damage analysis for every video. |
| **Planning total** | **0.5–3 GPU-minutes/video** | Broad sizing range pending benchmark. |

That planning range equals about **8–50 GPU-hours/month** at 1,000 videos/month. A burst of 100 videos in four hours would require 25 videos/hour. One GPU should handle the baseline with a queue and one visual job at a time, though the high end of the range could consume most of the burst window.

Use one inference worker per GPU initially, queue capacity around 500–1,000 jobs, bounded retries for transient failures, and a dead-letter state after retries. At 2,000 videos/month the system likely remains viable with scheduling; at 5,000/month, test the required completion window and consider a second GPU.

A practical initial node has 16–24 CPU cores, 128 GB RAM (64 GB minimum), 2–4 TB NVMe, separate NAS, and 10GbE preferred. Measure wall power during pilot; do not estimate electricity by multiplying GPU TDP by every hour in the month.

**VRAM planning estimates:** Qwen2-VL-7B FP16/BF16 weights use roughly 14–16 GB before runtime; budget 20–28 GB total for one controlled-resolution request. Qwen3-VL-8B should be budgeted similarly, around 22–30 GB at the proposed input envelope. Qwen3-14B BF16 weights alone are about 30 GB, so a 32 GB card does not provide safe room for KV cache and runtime. Use 4-bit quantization or a larger-memory GPU, and validate output quality. Do not assume the VLM, reasoning model, and damage networks can all remain resident concurrently; load them sequentially on the recommended single-GPU system.

More frames and higher image resolution increase visual input memory and latency. Moving from 8 to 16 or 30 frames increases workload and can force smaller resolution, chunked processing, or more VRAM. Concurrent jobs multiply activation and cache use. These are reasons to benchmark the frame policy, not evidence that eight frames is accurate enough.

## 5. Model selection and local compatibility

| Task | Proposed local model | License and runtime | Memory / trade-off |
|---|---|---|---|
| Video interpretation baseline | **Qwen2-VL-7B-Instruct**, exact Hugging Face repository `Qwen/Qwen2-VL-7B-Instruct`; model files listed around 16.6 GB | Apache-2.0; Transformers local inference; Linux + NVIDIA CUDA. Offline after weights and dependencies are staged. | Plan 20–28 GB VRAM for the model, runtime, visual input and cache at controlled resolution; 32–48 GB system RAM. Retains the repo’s existing implementation. |
| VLM alternative to evaluate | **Qwen3-VL-8B-Instruct**, exact repo `Qwen/Qwen3-VL-8B-Instruct`; repository files around 17.5 GB | Apache-2.0; local Transformers path, with serving runtime compatibility to be verified | Similar VRAM envelope to the 7B baseline. Newer model does not prove better performance on CCTV accident evidence. Compare on a labeled set first. |
| Liability reasoning | **Qwen3-14B**, 14.8B parameters, Apache-2.0 | Transformers, vLLM or a validated local quantized runtime; Linux + CUDA | BF16 is too tight for a 32 GB card once runtime and KV cache are counted. 4-bit is more practical on 32 GB, with possible quality loss. Compare with Qwen3-8B if minimizing memory. |
| Vehicle detection | Existing **YOLO11x**, 56.9M parameters | Ultralytics code/weights under AGPL-3.0 or Enterprise licensing; PyTorch/Ultralytics on CPU or CUDA | Under 0.2 GB FP16 weights; reserve 1–3 GB VRAM, 4–8 GB system RAM. BoT-SORT is the tracking method, not another learned model. |
| Damage and parts | Existing `damage_segmentation.pt` and `parts_segmentation.pt` paths | Exact model versions, class counts and licensing are not available in the repo | Budget 2–6 GB VRAM for both, but accuracy cannot be claimed without weights, data provenance, per-class metrics and evaluation. |
| RAG | **BAAI/bge-m3** embedding model (~568M parameters, MIT) and optional **BAAI/bge-reranker-v2-m3** (~0.6B, Apache-2.0) | SentenceTransformers on CPU; offline when staged | Repository config already loads these on CPU. Budget 4–8 GB RAM per model while loaded. |

Ultralytics documents AGPL-3.0 and Enterprise licensing. Obtain legal advice or a commercial license quote before using it in a closed commercial service. Keep model cards and license files with checksummed, immutable local artifacts. Stage packages and weights in a local registry; test inference with internet access disabled.

**Damage estimation requires more than segmentation.** Detecting a damaged region, assigning it to a vehicle part, classifying severity, and estimating EGP repair cost are separate tasks. The repository’s severity is partly heuristic, and its cost engine uses rules/tables. Production needs labeled Egyptian vehicle images and masks, part/damage/severity annotations, repair quotes or invoices, and dated part/labor/paint prices. Calibrate against held-out cases and require human review.

For legal reasoning, require the model to separate **observations, cited legal text, assumptions, and uncertain conclusions**. Validate each citation against retrieved chunk IDs. The README’s New York/VTL references must not be treated as Egyptian law.

## 6. Hardware configuration comparison

Prices below are planning budgets. Egypt listings were available for GPUs, NAS, drives and UPS units; there was no directly comparable complete production server quote.

| | A — minimum viable production | B — recommended | C — higher headroom |
|---|---|---|---|
| GPU | 1× RTX 5080, 16 GB | 1× RTX 5090, 32 GB | 2× RTX 5090, 32 GB each |
| Host | Ryzen 9/Core Ultra class, 16 cores, 128 GB RAM | Threadripper/EPYC workstation, 24 cores, 128–256 GB ECC | EPYC/Threadripper Pro, 32+ cores, 256 GB ECC |
| Storage | 2 TB NVMe; separate 4-bay NAS, 4×12 TB RAID6 | 4 TB NVMe; separate 4-bay NAS, 4×12 TB RAID6 | 4–8 TB NVMe; larger backup arrangement |
| Power | 1.2–1.5 kW PSU, airflow tower, 3 kVA online UPS | 1.6 kW PSU, workstation chassis, 3 kVA online UPS | 2.4 kW+ PSU, rack airflow, 6 kVA online UPS |
| CAPEX estimate | **EGP 663k / $11,990** | **EGP 1,102k / $19,931** | **EGP 2,204k / $39,862** |
| Workload | 1,000/month at 8 frames; 16 GB VRAM restricts resolution/concurrency | 1,000–2,000/month baseline; one visual job at a time | Greater concurrency and frame-resolution headroom |
| Main limitation | 16 GB may require quantization or tighter visual settings | Single host is a failure domain; 14B reasoning should be quantized | Higher power and complexity; two GPUs in one host still do not provide HA |

A single node with a separate NAS is recoverable, but it is not high availability. A genuinely redundant service needs two independent inference nodes, replicated database/queue, redundant networking and power, failover tests, and separate backups. Budget an additional **EGP 0.8–1.4m** over Option B as an engineering estimate.

## 7. Current sourced prices

Accessed 25 October 2026. Listings are not supplier quotations and do not consistently disclose all VAT, shipping, import, or warranty terms.

| Item | Observed source | Price and treatment |
|---|---|---|
| RTX 5080, 16 GB | [NileShopper Egypt listings](https://nileshopper.com/ar/category/graphics-card/geforce-rtx-5080) | From EGP 75,000; budget EGP 85,000. Listing aggregator; confirm stock, warranty and VAT. |
| RTX 5090, 32 GB | [MSI listing via NileShopper](https://nileshopper.com/products/65175-msi-rtx-5090-vanguard-soc-gddr7-graphic-cards) | EGP 335,000; listing says in stock and 4-year warranty. Confirm directly with retailer. |
| NAS enclosure | [Contra Synology listings](https://contratechnology.com.contratech.net/en/products/synology) | Synology DS425+ listed at EGP 30,200, disks excluded. |
| 12 TB NAS drive | [EG-Tech NAS drive listings](https://shop.itegy.com.eg/product-category/networking-servers/network-storage/nas-hdd-and-ssd/) | Synology HAT3310-12T listed at EGP 42,534, explicitly including 14% VAT. Four drives: EGP 170,136. |
| 3 kVA online UPS | [Do Link Delta listing](https://dolinkegypt.com/en/shop/category/ups-security-products-egypt-171); [EG-Tech APC listing](https://shop.itegy.com.eg/product-category/power-ups/ups/) | Delta CL3000 EGP 20,700; APC SRV3KI EGP 42,500. Allow EGP 25–43k depending on runtime and warranty. |
| Commercial electricity tariff | [EgyptERA April 2026 tariff schedule](https://egyptera.org/en/TarrifApril2026.aspx) | EGP 2.79/kWh at the highest commercial tier. Confirm bill treatment and service charges. |
| USD/EGP | [2026 exchange-rate history](https://www.poundsterlinglive.com/history/USD-EGP-2026) | EGP 55.29 per USD at the 23 October close. |

Host components, RAM, NVMe, network/security gear, rack and integration are **engineering allowances**. Obtain three itemized Egyptian quotes and confirm whether VAT, import charges and warranty are included.

## 8. CAPEX breakdown

EGP thousands; the table keeps GPU, internal NVMe, NAS, UPS and networking separate to avoid double-counting. NAS includes four 12 TB disks. Contingency covers local availability and import uncertainty.

| CAPEX | A | B | C |
|---|---:|---:|---:|
| GPU(s) | 85 | 335 | 670 |
| CPU, motherboard, chassis, PSU, cooling, RAM | 190 | 270 | 500 |
| NVMe/boot and internal data | 25 | 35 | 65 |
| NAS and disks | 210 | 210 | 420 |
| UPS and power protection | 30 | 35 | 65 |
| Firewall, switch, 10GbE, rack/installation | 45 | 55 | 110 |
| Deployment/integration | 40 | 80 | 120 |
| Initial contingency | 38 | 82 | 254 |
| **Initial CAPEX** | **663** | **1,102** | **2,204** |
| **USD at 55.29** | **$11,990** | **$19,931** | **$39,862** |

Software/development costs are separate. No Ultralytics commercial license quote, damage-model development, or dataset creation is included.

## 9. Monthly and annual OPEX

Electricity formula: **average wall power (kW) × 730 hours/month × EGP 2.79/kWh**. Estimated average power, including storage/networking and a cooling allowance, is 0.48 kW for A, 0.58 kW for B, and 0.90 kW for C. This yields about EGP 978, 1,181 and 1,831/month respectively. These are estimates; measure actual wall power during the pilot.

| OPEX item | Planning amount | Basis |
|---|---:|---|
| Electricity | A EGP 1,000; B EGP 1,200; C EGP 1,850/month | Formula above; not GPU TDP × all hours |
| Business internet | EGP 2,000/month | Allowance; inference can work offline |
| Maintenance/replacement reserve | 0.75% of compute CAPEX/month | Approx. EGP 1.5k / 3k / 6k for A/B/C |
| Backup expansion, security updates, monitoring | EGP 4,000/month | Self-hosted tools assumed; periodic replacement allowance |
| Incremental admin/security services | EGP 6,000/month | Contract support allowance |
| **Cash OPEX** | **A EGP 15k; B EGP 16k; C EGP 18k/month** | EGP 180k / 192k / 216k annually |
| Part-time operations labor | EGP 25,000/month | 0.25 FTE planning estimate |
| **Fully loaded OPEX** | **A EGP 40k; B EGP 41k; C EGP 43k/month** | EGP 480k / 492k / 516k annually |

A dedicated ML/MLOps engineer is estimated at **EGP 80k–160k/month loaded**, depending on experience and benefits. Compared with the part-time allowance, that adds EGP 55k–135k/month. This is an allocated labor estimate, not a salary survey. External AI inference and API spend remain zero.

## 10–11. Three-year and five-year TCO

Base TCO keeps 2026 EGP nominally fixed: CAPEX plus monthly cash OPEX; fully loaded TCO adds EGP 25k/month for part-time operations labor. It does not include a GPU replacement in the base case.

| Option / horizon | Cash TCO EGP | Cash USD | Fully loaded EGP | Fully loaded USD |
|---|---:|---:|---:|---:|
| A, 3 years | 1,203,000 | 21,761 | 2,103,000 | 38,037 |
| B, 3 years | 1,678,000 | 30,349 | 2,578,000 | 46,629 |
| C, 3 years | 2,852,000 | 51,582 | 3,752,000 | 67,842 |
| A, 5 years | 1,563,000 | 28,273 | 3,063,000 | 55,390 |
| B, 5 years | 2,062,000 | 37,294 | 3,562,000 | 64,451 |
| C, 5 years | 3,284,000 | 59,396 | 4,784,000 | 86,526 |

**Conservative case:** 20% increase to initial procurement, 25% uplift to recurring costs across the period, and a year-four GPU replacement reserve equal to 60% of current GPU cost. This is a stress scenario, not a price forecast.

| Option / horizon | Cash TCO EGP / USD | Fully loaded EGP / USD |
|---|---:|---:|
| A, 3 years | 1.47m / $26.6k | 2.37m / $42.9k |
| B, 3 years | 2.01m / $36.4k | 2.91m / $52.6k |
| C, 3 years | 3.20m / $57.9k | 4.10m / $74.2k |
| A, 5 years | 2.06m / $37.3k | 3.86m / $69.9k |
| B, 5 years | 2.61m / $47.2k | 4.11m / $74.4k |
| C, 5 years | 4.58m / $82.8k | 6.08m / $110.0k |

Unit costs use 36,000 videos in three years and 60,000 in five years. Since each video is one minute, cost/video equals cost/video-minute.

| Option | 3y cash/video | 3y loaded/video | 5y cash/video | 5y loaded/video |
|---|---:|---:|---:|---:|
| A | EGP 33.42 ($0.60) | EGP 58.42 ($1.06) | EGP 26.05 ($0.47) | EGP 51.05 ($0.92) |
| B | EGP 46.61 ($0.84) | EGP 71.61 ($1.30) | EGP 34.37 ($0.62) | EGP 59.37 ($1.07) |
| C | EGP 79.22 ($1.43) | EGP 104.22 ($1.88) | EGP 54.73 ($0.99) | EGP 79.73 ($1.44) |

Cost per 1,000 videos is the per-video figure × 1,000. For Option B, amortized monthly TCO is EGP 46,611 cash or EGP 71,611 loaded over three years; EGP 34,367 cash or EGP 59,367 loaded over five years. This amortization includes CAPEX and is not the monthly cash bill.

## 12. Sensitivity analysis

- **500 / 1,000 / 2,000 / 5,000 videos/month:** approximately 4–25 / 8–50 / 17–100 / 42–250 GPU-hours/month under the broad planning range. The higher volumes may require a second GPU if completion windows are short.
- **8 / 16 / 30 frames:** as a first approximation, frame-processing load grows about 1× / 2× / 3.75×. This does not establish an accuracy benefit. Thirty frames may require lower resolution or chunked processing.
- **Lower vs higher resolution:** lower resolution reduces memory and latency but may lose distant or small evidence. Higher resolution raises visual-token memory and compute.
- **4-bit vs FP16/BF16:** 4-bit makes a 14B reasoning model more practical on a 32 GB card, but may affect nuanced legal reasoning and structured output. Validate against human-reviewed cases. Higher precision may require 48 GB+ practical VRAM for 14B with useful context.
- **Smaller vs larger VLM:** 7B/8B fits the recommended budget. Models above 30B typically need larger-memory or multiple GPUs and higher CAPEX. Buy that capacity only if a blinded CCTV evaluation shows meaningful quality improvement.
- **One vs multiple GPUs:** a second GPU increases concurrency if workloads are separated, but VRAM is not automatically pooled. A second GPU in one host is not HA.
- **30 vs 90-day retention:** source footage alone is roughly 30 vs 90 GB/month at 4 Mbps. Allow 3× hot storage and backups; capacity is ample in the proposed NAS, but privacy exposure increases.
- **Electricity +25% / +50%:** Option B estimated electricity rises from EGP 1,200/month to about EGP 1,500 / 1,800. Staffing and hardware dominate total cost.
- **Procurement +20%:** adds about EGP 133k / 220k / 441k to A/B/C CAPEX before any further contingency.
- **GPU replacement in year 3 or 4:** reserve roughly EGP 51k / 201k / 402k for A/B/C using the conservative 60% replacement-cost assumption. Year 3 affects 3-year TCO; year 4 affects 5-year TCO.
- **Part-time vs dedicated support:** dedicated staffing adds an estimated EGP 55k–135k/month over the baseline 0.25 FTE, or EGP 1.98m–4.86m over three years before inflation.
- **Redundant deployment:** add approximately EGP 0.8m–1.4m over Option B CAPEX plus EGP 10k–30k/month operational allowance, pending an actual design and quote.

## 13. Reliability, security and production risks

Use a local API/UI tier, durable queue, separate GPU worker, PostgreSQL or a tested migration from SQLite, and local object storage/NAS. Make jobs idempotent, set timeouts, bound retries, and dead-letter repeated failures. Initially allow one visual job per GPU. Warm models at worker start, monitor VRAM, and load large models sequentially on single-GPU systems.

Maintain an on-prem model registry with immutable versions, SHA-256 checksums, license copies, and dependency manifests. Stage all assets before deployment and verify offline startup with external network access blocked. Version the traffic-law corpus by jurisdiction, effective date, source, document checksum, index build, and retrieved chunk IDs.

Require outputs to distinguish observations with timestamps/track IDs, retrieved law text and citation IDs, assumptions, interpretations, and uncertainty. Validate citations and require human signoff. The repository’s current New York/VTL references are not a substitute for an authoritative Egyptian corpus.

Use private-network or VPN access, MFA, role-based access, encryption at rest and in transit, secrets outside `.env`, and auditable records of uploads, model versions, edits, and downloads. Validate and sandbox video decoding. Purge source video after 30 days unless a documented legal hold applies. Back up the database and metadata daily, protect video/object data incrementally, keep an encrypted offline or immutable copy, and test restores quarterly.

Monitor queue age, job latency, failures, GPU/CPU/RAM, storage health, backup freshness, and model hashes. Stage security and model updates, benchmark them locally, and retain rollback capability. External SOC services, penetration tests, legal advice, and a second site are not included in the cost model.

Material risks are the unverified damage weights and rights, Ultralytics licensing obligations, the non-Egypt law corpus, lack of local legal reasoning implementation, sample fallback behavior, lack of a durable queue, GPU availability/warranty, and Option A’s 16 GB VRAM limit.

## 14. Procurement plan and validation checklist

1. Remove hosted API paths and secrets; ensure inference workers fail closed and have no external AI network route.
2. Inventory damage and tracking weights, hashes, labels, training data rights, and performance. Resolve Ultralytics licensing.
3. Build a pinned local software image and stage every weight. Test startup and inference with WAN disabled.
4. Create a representative, legally reviewed CCTV benchmark across lighting, camera angles, weather, occlusion, night conditions, motorcycles, and ambiguous events.
5. Compare Qwen2-VL-7B and Qwen3-VL-8B at 8/16 frames and several resolutions. Compare local 14B 4-bit reasoning with an 8B option. Measure p50/p95 latency, peak VRAM, failure rate, event recall, citation validity, human agreement, and wall kWh/job.
6. Run a burst test of 100 videos in four hours with tracking, VLM, RAG, reasoning, damage assessment, and storage active. Verify retries, queue behavior, retention purge, and restore.
7. Obtain three Egyptian itemized system quotes. Confirm exact GPU and VRAM, VAT/import costs, invoice, delivery, warranty, replacement lead time, UPS battery/runtime, and network compatibility.
8. Buy Option B only after the benchmark confirms the chosen VLM fits and meets the agreed evidence-quality and completion targets. If availability targets require HA, procure two independent nodes and test failover before launch.

## 15. Sources and evidence labels

Primary model and specification sources:

- [Qwen2-VL-7B repository and files](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/tree/main) — Apache-2.0, Transformers and local serving options.
- [Qwen3-VL-8B repository](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/main) — Apache-2.0.
- [Qwen3-14B model card](https://huggingface.co/Qwen/Qwen3-14B) — Apache-2.0, 14.8B parameters.
- [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11) — YOLO11x parameter count and vendor-reported benchmarks; those vendor figures are not used as measured Accident Intel throughput.
- [Ultralytics licensing](https://docs.ultralytics.com/help/contributing) — AGPL-3.0 or Enterprise options.
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3) and [BGE reranker card](https://huggingface.co/BAAI/bge-reranker-v2-m3).
- [NVIDIA RTX 50-series specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/) — vendor specifications, not inference benchmarks.
- Egypt price, tariff, and FX sources are linked in section 7.

**Evidence status:** Sourced prices are observed listings, not binding quotes. Model cards and manufacturer pages provide vendor specifications, not this project’s performance. System costs, throughput, power, staffing and TCO are **engineering estimates or explicit assumptions**. There are **no measured end-to-end benchmarks** for this repository. No external AI inference is included.
