"""SQLite database setup and CRUD helpers.

Provides functions to initialise the SQLite schema and perform create, read,
update, and delete operations on both the ``accidents`` and
``damage_assessments`` tables.  All functions open and close their own
connections so they are safe to call from multiple threads.
"""
import json
import sqlite3
from datetime import datetime, timezone


def init_db(db_path: str) -> None:
    """Create the ``accidents`` and ``damage_assessments`` tables if they don't already exist.

    Also creates any missing parent directories for the database file.

    Args:
        db_path: Absolute or relative path to the SQLite database file.
    """
    from pathlib import Path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS accidents (
                id         TEXT PRIMARY KEY,
                status     TEXT NOT NULL DEFAULT 'created',
                data       TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS damage_assessments (
                id         TEXT PRIMARY KEY,
                status     TEXT NOT NULL DEFAULT 'created',
                data       TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def get_conn(db_path: str) -> sqlite3.Connection:
    """Return a sqlite3 connection with ``row_factory`` set to :class:`sqlite3.Row`.

    Using ``sqlite3.Row`` allows columns to be accessed by name.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        An open :class:`sqlite3.Connection` that must be closed by the caller.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Returns:
        A string such as ``"2024-01-15T10:30:00+00:00"``.
    """
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Accidents
# ---------------------------------------------------------------------------

def create_accident(db_path: str, accident_id: str, data: dict) -> dict:
    """Insert a new accident record into the database.

    Args:
        db_path: Path to the SQLite database.
        accident_id: Unique identifier for the new accident.
        data: Dict of accident fields to store as JSON. The ``status`` key
            is extracted to populate the ``status`` column; it defaults to
            ``"created"`` if not present.

    Returns:
        The stored record dict with ``id`` and ``created_at`` added.
    """
    conn = get_conn(db_path)
    try:
        now = _now_iso()
        conn.execute(
            "INSERT INTO accidents (id, status, data, created_at) VALUES (?, ?, ?, ?)",
            (accident_id, data.get("status", "created"), json.dumps(data), now),
        )
        conn.commit()
        return {**data, "id": accident_id, "created_at": now}
    finally:
        conn.close()


def get_accident(db_path: str, accident_id: str) -> dict | None:
    """Fetch a single accident record by ID.

    Args:
        db_path: Path to the SQLite database.
        accident_id: The accident's unique identifier.

    Returns:
        The accident record dict with ``id``, ``status``, and ``created_at``
        merged in, or ``None`` if no record was found.
    """
    conn = get_conn(db_path)
    try:
        row = conn.execute(
            "SELECT id, status, data, created_at FROM accidents WHERE id = ?",
            (accident_id,),
        ).fetchone()
        if row is None:
            return None
        record = json.loads(row["data"])
        record["id"] = row["id"]
        record["status"] = row["status"]
        record["created_at"] = row["created_at"]
        return record
    finally:
        conn.close()


def update_accident(db_path: str, accident_id: str, updates: dict, *, only_if_idle: bool = False) -> dict | None:
    """Merge ``updates`` into an existing accident record.

    Reads the current JSON blob, merges ``updates`` into it, and writes it
    back. The ``status`` column is also updated if ``updates`` contains a
    ``"status"`` key.

    Args:
        db_path: Path to the SQLite database.
        accident_id: The accident's unique identifier.
        updates: Dict of fields to merge. ``"status"`` is handled specially.
        only_if_idle: When ``True``, the update is skipped (returns ``None``)
            if the record's current status is in :data:`ACTIVE_STATUSES`.

    Returns:
        The refreshed accident record dict, or ``None`` if the record does not
        exist or the update was blocked by ``only_if_idle``.
    """
    conn = get_conn(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status, data FROM accidents WHERE id = ?", (accident_id,)
        ).fetchone()
        if row is None:
            return None
        if only_if_idle and row["status"] in ACTIVE_STATUSES:
            return None
        current = json.loads(row["data"])
        current.update(updates)
        new_status = updates.get("status", row["status"])
        conn.execute(
            "UPDATE accidents SET status = ?, data = ? WHERE id = ?",
            (new_status, json.dumps(current), accident_id),
        )
        conn.commit()
        return get_accident(db_path, accident_id)
    finally:
        conn.close()


ACTIVE_STATUSES = {"processing", "tracking", "vlm_analysis", "law_retrieval", "reasoning"}
"""Status values that indicate a background analysis job is currently running."""


def claim_analysis(db_path: str, accident_id: str, is_demo: bool, *, video_vlm_api: str | None = None) -> bool:
    """Atomically reserve a job for analysis and clear any stale previous results.

    Uses ``BEGIN IMMEDIATE`` to avoid a race where two callers simultaneously
    try to start analysis on the same record.

    Args:
        db_path: Path to the SQLite database.
        accident_id: The accident to claim.
        is_demo: Whether this analysis uses mock (demo) data. Stored in the
            record for display purposes.

    Returns:
        ``True`` if the job was successfully claimed.  ``False`` if the record
        does not exist or is already in an :data:`ACTIVE_STATUSES` state.
    """
    conn = get_conn(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status, data FROM accidents WHERE id = ?", (accident_id,)).fetchone()
        if row is None or row["status"] in ACTIVE_STATUSES:
            return False
        data = json.loads(row["data"])
        data.update(status="processing", error=None, failed_stage=None,
                    tracking_data=None, vlm_description=None, laws=None, report=None,
                    is_demo_data=is_demo, video_vlm=None)
        if video_vlm_api is not None:
            data["video_vlm_api"] = video_vlm_api
        conn.execute("UPDATE accidents SET status = ?, data = ? WHERE id = ?",
                     ("processing", json.dumps(data), accident_id))
        conn.commit()
        return True
    finally:
        conn.close()


def recover_interrupted_analyses(db_path: str) -> None:
    """Mark any in-progress analyses as failed after a server restart.

    Background tasks cannot survive a process restart, so records left in an
    active state would be permanently stuck. This function transitions them to
    ``"failed"`` with a human-readable error message so the user can retry.

    Args:
        db_path: Path to the SQLite database.
    """
    conn = get_conn(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        for row in conn.execute("SELECT id, status, data FROM accidents").fetchall():
            if row["status"] in ACTIVE_STATUSES:
                data = json.loads(row["data"])
                data.update(status="failed", failed_stage=row["status"],
                            error="Analysis interrupted by a server restart. Retry this analysis.")
                conn.execute("UPDATE accidents SET status = 'failed', data = ? WHERE id = ?",
                             (json.dumps(data), row["id"]))
        conn.commit()
    finally:
        conn.close()


def delete_accident(db_path: str, accident_id: str) -> bool:
    """Delete an accident record from the database.

    Args:
        db_path: Path to the SQLite database.
        accident_id: The accident's unique identifier.

    Returns:
        ``True`` if the record existed and was deleted, ``False`` if it was
        not found.

    Raises:
        ValueError: If the accident's current status is in
            :data:`ACTIVE_STATUSES` (analysis is running).
    """
    conn = get_conn(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status FROM accidents WHERE id = ?", (accident_id,)).fetchone()
        if row and row["status"] in ACTIVE_STATUSES:
            raise ValueError("Cannot delete an analysis while it is running.")
        cursor = conn.execute("DELETE FROM accidents WHERE id = ?", (accident_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def list_accidents(
    db_path: str,
    page: int = 1,
    limit: int = 20,
    status: str | None = None,
) -> tuple[list[dict], int]:
    """Return a paginated list of accident records.

    Args:
        db_path: Path to the SQLite database.
        page: 1-based page number.
        limit: Maximum number of records per page.
        status: Optional status filter. When given, only records with this
            status are returned.

    Returns:
        A tuple ``(items, total)`` where ``items`` is the list of accident
        record dicts for the requested page and ``total`` is the count of
        matching records across all pages.
    """
    conn = get_conn(db_path)
    try:
        offset = (page - 1) * limit
        if status:
            total = conn.execute(
                "SELECT COUNT(*) FROM accidents WHERE status = ?", (status,)
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT id, status, data, created_at FROM accidents WHERE status = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            ).fetchall()
        else:
            total = conn.execute("SELECT COUNT(*) FROM accidents").fetchone()[0]
            rows = conn.execute(
                "SELECT id, status, data, created_at FROM accidents "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        items = []
        for row in rows:
            record = json.loads(row["data"])
            record["id"] = row["id"]
            record["status"] = row["status"]
            record["created_at"] = row["created_at"]
            items.append(record)
        return items, total
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Damage assessments
# ---------------------------------------------------------------------------

def create_damage(db_path: str, assessment_id: str, data: dict) -> dict:
    """Insert a new damage assessment record into the database.

    Args:
        db_path: Path to the SQLite database.
        assessment_id: Unique identifier for the new assessment.
        data: Dict of assessment fields stored as JSON.

    Returns:
        The stored record dict with ``id`` and ``created_at`` added.
    """
    conn = get_conn(db_path)
    try:
        now = _now_iso()
        conn.execute(
            "INSERT INTO damage_assessments (id, status, data, created_at) VALUES (?, ?, ?, ?)",
            (assessment_id, data.get("status", "created"), json.dumps(data), now),
        )
        conn.commit()
        return {**data, "id": assessment_id, "created_at": now}
    finally:
        conn.close()


def get_damage(db_path: str, assessment_id: str) -> dict | None:
    """Fetch a single damage assessment record by ID.

    Args:
        db_path: Path to the SQLite database.
        assessment_id: The assessment's unique identifier.

    Returns:
        The assessment record dict, or ``None`` if not found.
    """
    conn = get_conn(db_path)
    try:
        row = conn.execute(
            "SELECT id, status, data, created_at FROM damage_assessments WHERE id = ?",
            (assessment_id,),
        ).fetchone()
        if row is None:
            return None
        record = json.loads(row["data"])
        record["id"] = row["id"]
        record["status"] = row["status"]
        record["created_at"] = row["created_at"]
        return record
    finally:
        conn.close()


def update_damage(db_path: str, assessment_id: str, updates: dict) -> dict | None:
    """Merge ``updates`` into an existing damage assessment record.

    Args:
        db_path: Path to the SQLite database.
        assessment_id: The assessment's unique identifier.
        updates: Dict of fields to merge into the stored data blob.

    Returns:
        The refreshed assessment record dict, or ``None`` if not found.
    """
    conn = get_conn(db_path)
    try:
        row = conn.execute(
            "SELECT status, data FROM damage_assessments WHERE id = ?", (assessment_id,)
        ).fetchone()
        if row is None:
            return None
        current = json.loads(row["data"])
        current.update(updates)
        new_status = updates.get("status", row["status"])
        conn.execute(
            "UPDATE damage_assessments SET status = ?, data = ? WHERE id = ?",
            (new_status, json.dumps(current), assessment_id),
        )
        conn.commit()
        return get_damage(db_path, assessment_id)
    finally:
        conn.close()


def list_damage(db_path: str) -> list[dict]:
    """Return all damage assessment records ordered by creation date (newest first).

    Args:
        db_path: Path to the SQLite database.

    Returns:
        A list of assessment record dicts.
    """
    conn = get_conn(db_path)
    try:
        rows = conn.execute(
            "SELECT id, status, data, created_at FROM damage_assessments ORDER BY created_at DESC"
        ).fetchall()
        result = []
        for r in rows:
            record = json.loads(r["data"])
            record["id"] = r["id"]
            record["status"] = r["status"]
            record["created_at"] = r["created_at"]
            result.append(record)
        return result
    finally:
        conn.close()


def delete_damage(db_path: str, assessment_id: str) -> bool:
    """Delete a damage assessment record from the database.

    Args:
        db_path: Path to the SQLite database.
        assessment_id: The assessment's unique identifier.

    Returns:
        ``True`` if the record existed and was deleted, ``False`` otherwise.
    """
    conn = get_conn(db_path)
    try:
        cur = conn.execute("DELETE FROM damage_assessments WHERE id = ?", (assessment_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
