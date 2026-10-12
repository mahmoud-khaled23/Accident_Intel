"""FastAPI dependency providers."""
import functools
import sqlite3
from typing import Generator

from .config import Settings


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()


def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Yield a sqlite3 connection, then close it."""
    settings = get_settings()
    from .models import get_conn
    conn = get_conn(str(settings.db_path))
    try:
        yield conn
    finally:
        conn.close()
