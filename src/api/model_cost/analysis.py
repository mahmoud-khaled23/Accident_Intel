"""Aggregation of model usage into a per-model cost breakdown."""
import logging

from src.api.dependencies import get_settings
from src.api.models import get_conn

logger = logging.getLogger(__name__)

_NOTE = (
    "Tracks model calls made after cost tracking was deployed. "
    "Historical runs are not backfilled; only newly recorded calls appear."
)


def get_cost_analysis() -> dict:
    """Aggregate ``model_usage`` into a per-model breakdown plus grand total.

    Returns:
        A dict with keys:

        - ``"note"`` — informational string about backfill limitations.
        - ``"models"`` — list of per-model dicts sorted by ``cost_usd`` desc
          then ``total_tokens`` desc.
        - ``"grand_total"`` — dict with summed totals across all models.
    """
    db_path = str(get_settings().db_path)
    conn = get_conn(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                model_id,
                source,
                COUNT(*)            AS calls,
                SUM(prompt_tokens)     AS prompt_tokens,
                SUM(completion_tokens) AS completion_tokens,
                SUM(total_tokens)      AS total_tokens,
                SUM(cost_usd)          AS cost_usd
            FROM model_usage
            GROUP BY model_id, source
            ORDER BY cost_usd DESC, total_tokens DESC
            """
        ).fetchall()

        total_row = conn.execute(
            """
            SELECT
                COUNT(*)               AS calls,
                SUM(prompt_tokens)     AS prompt_tokens,
                SUM(completion_tokens) AS completion_tokens,
                SUM(total_tokens)      AS total_tokens,
                SUM(cost_usd)          AS cost_usd
            FROM model_usage
            """
        ).fetchone()
    finally:
        conn.close()

    models = [
        {
            "model_id": row["model_id"],
            "source": row["source"],
            "calls": row["calls"],
            "prompt_tokens": row["prompt_tokens"] or 0,
            "completion_tokens": row["completion_tokens"] or 0,
            "total_tokens": row["total_tokens"] or 0,
            "cost_usd": row["cost_usd"] or 0.0,
        }
        for row in rows
    ]

    if total_row and total_row["calls"]:
        grand_total = {
            "calls": total_row["calls"],
            "prompt_tokens": total_row["prompt_tokens"] or 0,
            "completion_tokens": total_row["completion_tokens"] or 0,
            "total_tokens": total_row["total_tokens"] or 0,
            "cost_usd": total_row["cost_usd"] or 0.0,
        }
    else:
        grand_total = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }

    return {
        "note": _NOTE,
        "models": models,
        "grand_total": grand_total,
    }
