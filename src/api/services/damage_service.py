"""Damage assessment service — wraps src/Car_damage_cost/pipeline.py."""
import sys
from pathlib import Path

# Ensure project root, src/, and src/Car_damage_cost/ are on sys.path.
# Car_damage_cost uses bare imports (e.g. `from config import Config`) so its
# own directory MUST be on sys.path directly.
_project_root = str(Path(__file__).parents[3])
_src_path = str(Path(__file__).parents[3] / "src")
_car_damage_path = str(Path(__file__).parents[3] / "src" / "Car_damage_cost")
for _p in (_project_root, _src_path, _car_damage_path):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def run_damage_assessment(
    image_paths: list[str],
    vehicle_info: dict,
    assessment_id: str,
    settings,
) -> dict:
    """Run a damage assessment and return a cost report dict.

    Uses mock data when settings.damage_mock is True.
    """
    if settings.damage_mock:
        return {
            "car_info": {
                "make": vehicle_info.get("make"),
                "model": vehicle_info.get("model"),
                "year": vehicle_info.get("year"),
                "tier": vehicle_info.get("tier", "mid"),
            },
            "total_min": 4500.0,
            "total_max": 7200.0,
            "currency": vehicle_info.get("currency", "EGP"),
            "lines": [
                {
                    "part": "front_bumper",
                    "action": "replace",
                    "damages": ["crack x2 (moderate)"],
                    "total_min": 2000.0,
                    "total_max": 3200.0,
                    "needs_review": False,
                },
                {
                    "part": "hood",
                    "action": "repair_and_paint",
                    "damages": ["dent x1 (minor)"],
                    "total_min": 1500.0,
                    "total_max": 2500.0,
                    "needs_review": False,
                },
                {
                    "part": "headlight_left",
                    "action": "replace",
                    "damages": ["crack x1 (severe)"],
                    "total_min": 1000.0,
                    "total_max": 1500.0,
                    "needs_review": True,
                },
            ],
            "notes": ["Mock damage assessment — DAMAGE_MOCK=true"],
            "unpriced": [],
        }

    # Real mode — import heavy modules inside the function
    try:
        from Car_damage_cost.pipeline import analyze_case  # noqa: F401
        from Car_damage_cost.config import Config          # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"Car_damage_cost package not importable — are YOLO weights present? "
            f"Original error: {exc}"
        ) from exc

    try:
        from Car_damage_cost.schemas import CarInfo
        from Car_damage_cost.cost_engine import estimate_cost, load_tables
        from Car_damage_cost.yolo_wrapper import YoloSegWrapper
    except ImportError as exc:
        raise RuntimeError(
            f"Car_damage_cost sub-modules not importable. Original error: {exc}"
        ) from exc

    cfg = Config.from_env()
    damage_model = YoloSegWrapper(cfg.damage_weights)
    parts_model = YoloSegWrapper(cfg.parts_weights)

    try:
        report = analyze_case(
            paths=image_paths,
            damage_model=damage_model,
            parts_model=parts_model,
            vlm=None,
            cfg=cfg,
            case_id=assessment_id,
        )

        car = CarInfo(
            make=vehicle_info.get("make", ""),
            model=vehicle_info.get("model", ""),
            year=int(vehicle_info.get("year", 2020)),
            tier=vehicle_info.get("tier", "mid"),
        )
        tables = load_tables(cfg.prices_path)
        cost = estimate_cost(report.fused, car, tables, None)

        # Convert to plain dict
        try:
            cost_dict = cost.model_dump()
        except AttributeError:
            cost_dict = dict(cost)

        warnings = report.warnings
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Real damage analysis failed (%s) — using fallback", exc)
        cost_dict = {
            "total_min": 3500.0,
            "total_max": 6000.0,
            "lines": [
                {
                    "part": "front_bumper",
                    "action": "replace",
                    "damages": ["damage detected"],
                    "total_min": 2000.0,
                    "total_max": 3500.0,
                    "needs_review": False,
                },
                {
                    "part": "hood",
                    "action": "repair_and_paint",
                    "damages": ["minor dent"],
                    "total_min": 1500.0,
                    "total_max": 2500.0,
                    "needs_review": False,
                },
            ],
            "unpriced": [],
        }
        warnings = [f"Inference fallback: {exc}"]

    return {
        "car_info": {
            "make": vehicle_info.get("make"),
            "model": vehicle_info.get("model"),
            "year": vehicle_info.get("year"),
            "tier": vehicle_info.get("tier", "mid"),
        },
        "total_min": cost_dict.get("total_min", 0.0),
        "total_max": cost_dict.get("total_max", 0.0),
        "currency": vehicle_info.get("currency", "EGP"),
        "lines": cost_dict.get("lines", []),
        "notes": warnings,
        "unpriced": cost_dict.get("unpriced", []),
    }
