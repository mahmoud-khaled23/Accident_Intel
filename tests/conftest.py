"""Shared pytest fixtures for Accident-Intel API tests."""
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def temp_db_path(tmp_path):
    """Return a path to a temporary SQLite DB file."""
    return tmp_path / "test.db"


@pytest.fixture
def test_client(tmp_path):
    """Create a TestClient with all mock flags enabled and isolated storage."""
    db_path = str(tmp_path / "test.db")
    data_dir = str(tmp_path)

    # Override environment before importing the app
    env_overrides = {
        "TRACKING_MOCK": "true",
        "VLM_MOCK": "true",
        "REASONING_MOCK": "true",
        "RAG_MOCK": "false",
        "DATA_DIR": data_dir,
        "DB_PATH": db_path,
    }
    original = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update(env_overrides)

    try:
        # Re-import with patched env to get a fresh Settings instance
        from src.api.main import create_app
        from src.api import dependencies

        # Clear the lru_cache so Settings re-reads env
        dependencies.get_settings.cache_clear()

        app = create_app()
        with TestClient(app) as client:
            yield client
    finally:
        # Restore original env and clear cache
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        try:
            from src.api import dependencies as _dep
            _dep.get_settings.cache_clear()
        except Exception:
            pass
