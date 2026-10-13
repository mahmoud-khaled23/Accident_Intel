"""Isolated API tests: real video decoding, explicitly mocked model inference."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def temp_db_path(tmp_path):
    return tmp_path / 'test.db'


@pytest.fixture
def video_bytes(tmp_path):
    path = tmp_path / 'input.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (160, 96))
    assert writer.isOpened()
    for i in range(12):
        frame = np.full((96, 160, 3), i * 15, dtype=np.uint8)
        cv2.rectangle(frame, (i * 4, 30), (i * 4 + 25, 60), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


@pytest.fixture
def test_client(tmp_path, monkeypatch):
    for key, value in {
        'TRACKING_MOCK': 'true', 'VLM_MOCK': 'true', 'REASONING_MOCK': 'true',
        'RAG_MOCK': 'true', 'DATA_DIR': str(tmp_path / 'data'), 'DB_PATH': str(tmp_path / 'test.db'),
    }.items():
        monkeypatch.setenv(key, value)
    from src.api import dependencies
    from src.api.main import create_app
    dependencies.get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            yield client
    finally:
        dependencies.get_settings.cache_clear()
