"""Accident analysis orchestration pipeline."""
import logging
from pathlib import Path

from . import tracking_service, vlm_service, law_service, reasoning_service

logger = logging.getLogger(__name__)


def run_analysis_pipeline(
    accident_id: str,
    video_path: str,
    db_path: str,
    settings,
) -> None:
    """Run the full analysis pipeline as a background task.

    Stages: tracking → VLM analysis → law retrieval → reasoning.
    DB status is updated at each stage.
    """
    from ..models import update_accident, get_accident

    out_dir = str(Path(db_path).parent / "accidents" / accident_id)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    try:
        # 1. Vehicle tracking
        update_accident(db_path, accident_id, {"status": "tracking"})
        tracking_data = tracking_service.run_tracking(
            video_path, accident_id, out_dir, mock=settings.tracking_mock
        )
        record = get_accident(db_path, accident_id) or {}
        record["tracking_data"] = tracking_data
        update_accident(db_path, accident_id, {"tracking_data": tracking_data})

        # 2. VLM video analysis
        update_accident(db_path, accident_id, {"status": "vlm_analysis"})
        vlm_description = vlm_service.analyze_video(video_path, accident_id, settings)
        update_accident(db_path, accident_id, {"vlm_description": vlm_description})

        # 3. Law retrieval
        update_accident(db_path, accident_id, {"status": "law_retrieval"})
        laws = law_service.retrieve_laws(vlm_description, str(settings.rag_out_dir), settings)
        update_accident(db_path, accident_id, {"laws": laws})

        # 4. Reasoning / liability assessment
        update_accident(db_path, accident_id, {"status": "reasoning"})
        report = reasoning_service.analyze_accident(
            vlm_description, tracking_data, laws, settings
        )
        update_accident(
            db_path,
            accident_id,
            {"report": report, "status": "completed"},
        )

        logger.info("Analysis pipeline completed for accident %s", accident_id)

    except Exception as exc:
        logger.exception("Analysis pipeline failed for accident %s", accident_id)
        try:
            update_accident(
                db_path,
                accident_id,
                {"status": "failed", "error": str(exc)},
            )
        except Exception:
            pass  # Don't mask the original exception
