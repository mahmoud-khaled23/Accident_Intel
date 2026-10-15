"""Command-line entry point for the Car Damage Cost pipeline.

Runs the full damage-analysis and optional cost-estimation workflow from the
command line. Accepts one or more photos of a damaged car, runs YOLO
segmentation and optional VLM analysis, fuses per-photo results, and writes
a JSON case report. When vehicle information (make, model, year, tier) is
supplied, a cost estimate is also produced.

Usage::

    python main.py photo1.jpg photo2.jpg --make Toyota --model Corolla --year 2018 --tier mid
    python main.py photos/*.jpg --no-vlm --no-viz            # perception only
    python main.py a.jpg --make Kia --model Cerato --year 2020 --tier mid --no-price-agent
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config import Config
from cost_engine import estimate_cost, load_tables
from pipeline import analyze_case
from schemas import CarInfo
from vlm_client import VlmClient
from yolo_wrapper import YoloSegWrapper


def main() -> None:
    """Parse CLI arguments, run the damage pipeline, and print a summary.

    Reads one or more damaged-car images, runs segmentation (and optionally
    VLM analysis), writes a ``report.json``, and — when vehicle tier
    information is provided — a ``cost.json`` with repair-cost estimates in EGP.

    Returns:
        None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="damaged-car photos")
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--no-vlm", action="store_true", help="skip all VLM calls")
    ap.add_argument("--viz-dir", default="debug_out", help="where to save mask visualisations")
    ap.add_argument("--no-viz", action="store_true", help="do not save visualisation images")
    # cost
    ap.add_argument("--make")
    ap.add_argument("--model")
    ap.add_argument("--year", type=int)
    ap.add_argument("--tier", choices=["economy", "mid", "luxury"])
    ap.add_argument("--cost-out", default="cost.json")
    ap.add_argument("--no-price-agent", action="store_true", help="use only the placeholder price table")
    args = ap.parse_args()

    cfg = Config.from_env()

    damage_model = YoloSegWrapper(cfg.damage_weights, conf=cfg.damage_conf, class_conf=cfg.class_conf)
    parts_model = YoloSegWrapper(cfg.parts_weights, conf=cfg.parts_conf, imgsz=cfg.parts_imgsz, iou=cfg.parts_iou)
    vlm = None if args.no_vlm else VlmClient(cfg.vlm_base_url, cfg.vlm_model, cfg.vlm_timeout, api=cfg.vlm_api, api_key=cfg.vlm_api_key, extra=cfg.vlm_extra)

    report = analyze_case(args.images, damage_model, parts_model, vlm, cfg,
                          debug_dir=None if args.no_viz else args.viz_dir)

    Path(args.out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"case {report.case_id}: {len(report.fused)} fused damage groups -> {args.out}")
    for f in report.fused:
        flag = "  [REVIEW]" if f.needs_review else ""
        print(f"  {f.part:<22} {f.damage_type:<14} {f.severity.value:<9} x{f.count}  conf={f.max_confidence}{flag}")
    if not args.no_viz:
        print(f"  debug images -> {args.viz_dir}/")
    for w in report.warnings:
        print("  warning:", w)

    # ---------------- cost ----------------
    if not (args.make and args.model and args.year and args.tier):
        print("\n(no --make/--model/--year/--tier given: skipping cost estimate)")
        return

    car = CarInfo(make=args.make, model=args.model, year=args.year, tier=args.tier)
    tables = load_tables(cfg.prices_path)

    agent = None
    if not args.no_price_agent:
        if cfg.gemini_api_key:
            from price_agent import PriceAgent
            agent = PriceAgent(cfg.gemini_api_key, cfg.gemini_model, cfg.price_cache_path)
        else:
            print("\nGEMINI_API_KEY not set -> using the placeholder price table")

    cost = estimate_cost(report.fused, car, tables, agent)
    Path(args.cost_out).write_text(cost.model_dump_json(indent=2), encoding="utf-8")

    print(f"\nCost estimate for {car.year} {car.make} {car.model} ({car.tier.value}) -> {args.cost_out}")
    for l in cost.lines:
        flag = "  [REVIEW]" if l.needs_review else ""
        print(f"  {l.part:<22} {l.action:<17} {l.total_min:>9,.0f} - {l.total_max:>9,.0f} EGP  ({l.price_source[:45]}){flag}")
    print(f"  {'TOTAL':<40} {cost.total_min:>9,.0f} - {cost.total_max:>9,.0f} EGP")
    if cost.unconfirmed_max:
        print(f"  of which unconfirmed (needs review): {cost.unconfirmed_min:,.0f} - {cost.unconfirmed_max:,.0f} EGP")
    for u in cost.unpriced:
        print("  not priced:", u)
    for n in cost.notes:
        print("  note:", n)


if __name__ == "__main__":
    main()
