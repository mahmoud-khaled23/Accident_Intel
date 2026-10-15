"""Sequential, persistent upload-to-report pipeline (one inference job at a time).

Runs the full analysis pipeline for a single accident in order:
1. Vehicle tracking
2. VLM scene description
3. Traffic-law retrieval (RAG)
4. Liability reasoning

GPU access and local model caches are protected by a module-level lock so
only one pipeline runs at a time.  Each step writes its results to the
database immediately so partial results are available after a crash.
"""
import logging
from threading import Lock

from . import tracking_service, vlm_service, law_service, reasoning_service

logger = logging.getLogger(__name__)
_inference_lock = Lock()


def run_analysis_pipeline(accident_id: str, video_path: str, db_path: str, settings) -> None:
    """Execute the full analysis pipeline for one accident, writing each step's results to the DB.

    The function acquires :data:`_inference_lock` before running to serialise
    GPU/model access.  The accident's ``status`` column is updated before and
    after each step so the frontend can show fine-grained progress.

    On any exception the accident is marked ``"failed"`` with the failing
    stage and error message stored in the database.

    Args:
        accident_id: Identifier of the accident to analyse. Must already
            exist in the database with status ``"processing"``.
        video_path: Absolute path to the uploaded video file.
        db_path: Path to the SQLite database file.
        settings: Application :class:`~config.Settings` passed to each
            service.
    """
    from ..models import update_accident, get_accident

    # Keep queued jobs at processing; serialize GPU access and local model caches.
    with _inference_lock:
        stage = "tracking"
        try:
            record = get_accident(db_path, accident_id)
            if record is None:
                return
            out_dir = settings.data_dir / "accidents" / accident_id
            out_dir.mkdir(parents=True, exist_ok=True)
            update_accident(db_path, accident_id, {"status": stage, "error": None})
            tracking = tracking_service.run_tracking(
                video_path, accident_id, str(out_dir), mock=settings.tracking_mock,
                model_path=settings.tracking_model_path,
            )
            update_accident(db_path, accident_id, {"tracking_data": tracking})

            stage = "vlm_analysis"
            update_accident(db_path, accident_id, {"status": stage})
            description = vlm_service.analyze_video(video_path, accident_id, settings, tracking)
            if not description.strip():
                raise RuntimeError("The visual model returned an empty description.")
            update_accident(db_path, accident_id, {"vlm_description": description})

            stage = "law_retrieval"
            update_accident(db_path, accident_id, {"status": stage})
            laws = law_service.retrieve_laws(description, str(settings.rag_out_dir), settings)
            update_accident(db_path, accident_id, {"laws": laws})

            stage = "reasoning"
            update_accident(db_path, accident_id, {"status": stage})
            context = {key: record.get(key, "") for key in ("location", "date_time", "description", "notes")}
            report = reasoning_service.analyze_accident(description, tracking, laws, settings, context)
            update_accident(db_path, accident_id, {"report": report, "status": "completed", "error": None})
            logger.info("Analysis completed for %s", accident_id)
        except Exception as exc:
            logger.exception("Analysis %s failed during %s", accident_id, stage)
            update_accident(db_path, accident_id, {"status": "failed", "failed_stage": stage,
                                                  "error": f"{stage}: {exc}"})
