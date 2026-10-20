"""Damage assessment service — wraps src/car_damage_cost/pipeline.py.

Provides a single public function :func:`run_damage_assessment` that runs the
full car-damage detection and cost-estimation pipeline.

Imports are package-qualified to avoid collisions with API schemas.
"""
from pathlib import Path

# Debug-overlay views written by car_damage_cost.viz.save_debug, in the order
# they are surfaced to the UI: part masks, damage masks, and the combined view.
_SEG_VIEWS = ("parts", "damage", "combined")


def collect_segmentation_images(
    image_paths: list[str],
    assessment_id: str,
) -> list[dict]:
    """Locate the mask overlays written by :func:`car_damage_cost.viz.save_debug`.

    Args:
        image_paths: Uploaded photo paths, in pipeline order.
        assessment_id: Assessment whose ``segmentation/`` directory is searched.

    Returns:
        One entry per photo that has at least one overlay file on disk (photos
        the pipeline skipped — e.g. unreadable ones — are omitted). Each entry
        maps ``photo`` (the original filename) plus whichever of ``parts``,
        ``damage``, and ``combined`` views exist to their filesystem paths.
    """
    debug_dir = Path("out") / "damage" / assessment_id / "segmentation"
    images: list[dict] = []
    for image_path in image_paths:
        stem = Path(image_path).stem
        views = {
            view: str(debug_dir / f"{stem}_{view}.jpg")
            for view in _SEG_VIEWS
            if (debug_dir / f"{stem}_{view}.jpg").is_file()
        }
        if views:
            images.append({"photo": Path(image_path).name, **views})
    return images

def run_damage_assessment(
    image_paths: list[str],
    vehicle_info: dict,
    assessment_id: str,
    settings,
) -> dict:
    """Run a damage assessment and return a cost report dict.

    In mock mode returns a hard-coded sample response immediately.
    In real mode:
    1. Loads the YOLO damage and parts segmentation models.
    2. Runs :func:`car_damage_cost.pipeline.analyze_case` over all images.
    3. Calls :func:`car_damage_cost.cost_engine.estimate_cost` to produce a
       cost breakdown.
    4. Propagates inference failures without fabricating estimates.

    Args:
        image_paths: List of absolute paths to the uploaded damage images.
        vehicle_info: Dict with ``make``, ``model``, ``year``, ``tier``, and
            optional ``currency`` keys.
        assessment_id: Unique identifier for this assessment (used as the
            ``case_id`` passed to ``analyze_case``).
        settings: Application settings. ``settings.damage_mock`` controls
            whether mock data is returned.

    Returns:
        A dict with keys ``car_info``, ``total_min``, ``total_max``,
        ``currency``, ``lines``, ``notes``, ``unpriced``, and
        ``segmentation_images`` — per-photo paths of the mask overlays
        written by :func:`car_damage_cost.viz.save_debug` so the UI can
        display what the segmentation models found.
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
            "segmentation_images": [],
        }

    # Real mode — import heavy modules inside the function
    try:
        from src.car_damage_cost.pipeline import analyze_case  # noqa: F401
        from src.car_damage_cost.config import Config          # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"car_damage_cost package not importable — are YOLO weights present? "
            f"Original error: {exc}"
        ) from exc

    try:
        from src.car_damage_cost.schemas import CarInfo
        from src.car_damage_cost.cost_engine import estimate_cost, load_tables
        from src.car_damage_cost.yolo_wrapper import YoloSegWrapper
    except ImportError as exc:
        raise RuntimeError(
            f"car_damage_cost sub-modules not importable. Original error: {exc}"
        ) from exc

    cfg = Config.from_env()
    damage_model = YoloSegWrapper(cfg.damage_weights)
    parts_model = YoloSegWrapper(cfg.parts_weights)

    # Mask overlays land next to the assessment's stored uploads so the API
    # image endpoint can serve them (same out/damage/<id>/ tree as the router).
    debug_dir = Path("out") / "damage" / assessment_id / "segmentation"

    try:
        report = analyze_case(
            paths=image_paths,
            damage_model=damage_model,
            parts_model=parts_model,
            vlm=None,
            cfg=cfg,
            case_id=assessment_id,
            debug_dir=debug_dir,
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
        raise RuntimeError(f"Damage analysis failed: {exc}") from exc

    # Collect the overlays save_debug actually wrote (unreadable photos are
    # skipped by the pipeline, so check existence rather than assume).
    segmentation_images = collect_segmentation_images(image_paths, assessment_id)

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
        "segmentation_images": segmentation_images,
    }
