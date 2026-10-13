"""SQLite database setup and CRUD helpers."""
import json
import sqlite3
from datetime import datetime, timezone


def init_db(db_path: str) -> None:
    """Create tables if they don't already exist."""
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
    """Return a sqlite3 connection with row_factory set."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Accidents
# ---------------------------------------------------------------------------

def create_accident(db_path: str, accident_id: str, data: dict) -> dict:
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


def update_accident(db_path: str, accident_id: str, updates: dict) -> dict | None:
    conn = get_conn(db_path)
    try:
        row = conn.execute(
            "SELECT status, data FROM accidents WHERE id = ?", (accident_id,)
        ).fetchone()
        if row is None:
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


def delete_accident(db_path: str, accident_id: str) -> bool:
    """Delete an accident record. Returns True if it existed, False otherwise."""
    conn = get_conn(db_path)
    try:
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
