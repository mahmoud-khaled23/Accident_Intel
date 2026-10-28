"""SQLite persistence for per-model usage records."""
import json
import logging

from src.api.dependencies import get_settings
from src.api.models import get_conn, _now_iso

logger = logging.getLogger(__name__)


def init_model_usage(db_path: str) -> None:
    """Create the ``model_usage`` table and indexes if they do not already exist.

    Uses the same ``executescript`` style as ``src.api.models.init_db``.

    Args:
        db_path: Absolute or relative path to the SQLite database file.
    """
    from pathlib import Path  # noqa: PLC0415

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = get_conn(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS model_usage (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id          TEXT    NOT NULL,
                stage             TEXT    NOT NULL,
                source            TEXT    NOT NULL,
                prompt_tokens     INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens      INTEGER NOT NULL DEFAULT 0,
                cost_usd          REAL    NOT NULL DEFAULT 0.0,
                job_id            TEXT,
                extra             TEXT,
                created_at        TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_model_usage_model   ON model_usage(model_id);
            CREATE INDEX IF NOT EXISTS idx_model_usage_created ON model_usage(created_at);
            """
        )
        conn.commit()
    finally:
        conn.close()


def record_usage(
    model_id: str,
    stage: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int | None = None,
    cost_usd: float | None = None,
    job_id: str | None = None,
    extra: dict | None = None,
) -> None:
    """Record one model call into the ``model_usage`` table.

    All errors are caught and logged at WARNING level — this function never
    raises into the instrumented call site.

    ``total_tokens`` defaults to ``prompt_tokens + completion_tokens``.
    ``cost_usd`` defaults to the computed cost from the pricing table.
    ``source`` is derived internally from the pricing table — callers must NOT
    pass it.

    Args:
        model_id: The model identifier as used by the call site.
        stage: Pipeline stage label, e.g. ``"reasoning"``.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.
        total_tokens: Total tokens (defaults to sum of prompt + completion).
        cost_usd: USD cost override; computed from pricing table if omitted.
        job_id: Optional accident / assessment identifier for traceability.
        extra: Optional dict serialised as JSON into the ``extra`` column.
    """
    try:
        from src.api.model_cost.pricing import _lookup_pricing, compute_cost  # noqa: PLC0415

        db_path = str(get_settings().db_path)

        # Safety net: ensure table exists (idempotent)
        init_model_usage(db_path)

        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        if cost_usd is None:
            cost_usd = compute_cost(model_id, prompt_tokens, completion_tokens)

        source = _lookup_pricing(model_id)["source"]
        extra_str = json.dumps(extra) if extra is not None else None

        conn = get_conn(db_path)
        try:
            conn.execute(
                """
                INSERT INTO model_usage
                    (model_id, stage, source, prompt_tokens, completion_tokens,
                     total_tokens, cost_usd, job_id, extra, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    stage,
                    source,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    cost_usd,
                    job_id,
                    extra_str,
                    _now_iso(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("model_cost: failed to record usage: %s", exc)
