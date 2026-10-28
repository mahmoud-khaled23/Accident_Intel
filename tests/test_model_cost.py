"""Tests for the model_cost package — pricing, usage recording, and analysis."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# pricing.py
# ---------------------------------------------------------------------------


def test_compute_cost_known_model():
    from src.api.model_cost.pricing import compute_cost

    result = compute_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert isinstance(result, float)
    assert result > 0


def test_compute_cost_prefix_normalization():
    from src.api.model_cost.pricing import compute_cost

    with_prefix = compute_cost("openai/gpt-4o-mini", 1000, 500)
    without_prefix = compute_cost("gpt-4o-mini", 1000, 500)
    assert with_prefix == without_prefix


def test_compute_cost_unknown_no_raise():
    from src.api.model_cost.pricing import compute_cost

    result = compute_cost("unknown-xyz", 100, 50)
    assert result == 0.0


# ---------------------------------------------------------------------------
# usage.py + analysis.py — integration via tmp SQLite
# ---------------------------------------------------------------------------


def _make_mock_settings(db_path: Path):
    """Return a mock Settings object with db_path set."""
    mock = MagicMock()
    mock.db_path = db_path
    return mock


def test_record_usage_and_aggregation(tmp_path):
    db = tmp_path / "test.db"
    mock_settings = _make_mock_settings(db)

    with (
        patch("src.api.model_cost.usage.get_settings", return_value=mock_settings),
        patch("src.api.model_cost.analysis.get_settings", return_value=mock_settings),
    ):
        from src.api.model_cost.usage import record_usage
        from src.api.model_cost.analysis import get_cost_analysis

        record_usage("gpt-4o-mini", "reasoning", 100, 50)

        result = get_cost_analysis()

    assert len(result["models"]) == 1
    assert result["models"][0]["model_id"] == "gpt-4o-mini"
    assert result["models"][0]["calls"] == 1
    assert result["models"][0]["prompt_tokens"] == 100
    assert result["grand_total"]["calls"] == 1
    assert isinstance(result["note"], str)
    assert len(result["note"]) > 0


def test_empty_table_returns_zero_grand_total(tmp_path):
    db = tmp_path / "empty.db"
    mock_settings = _make_mock_settings(db)

    with (
        patch("src.api.model_cost.usage.get_settings", return_value=mock_settings),
        patch("src.api.model_cost.analysis.get_settings", return_value=mock_settings),
    ):
        from src.api.model_cost.usage import init_model_usage
        from src.api.model_cost.analysis import get_cost_analysis

        # Create tables without inserting any rows
        init_model_usage(str(db))

        result = get_cost_analysis()

    assert result["models"] == []
    assert result["grand_total"]["calls"] == 0
    assert result["grand_total"]["prompt_tokens"] == 0
    assert result["grand_total"]["completion_tokens"] == 0
    assert result["grand_total"]["total_tokens"] == 0
    assert result["grand_total"]["cost_usd"] == 0.0


def test_record_usage_swallows_bad_db(tmp_path):
    # Create a file where the DB directory should be, making mkdir fail.
    fake_dir = tmp_path / "not_a_dir"
    fake_dir.write_bytes(b"i am a file, not a directory")
    db = fake_dir / "db.sqlite"  # parent is a file, mkdir will fail
    mock_settings = _make_mock_settings(db)

    with patch("src.api.model_cost.usage.get_settings", return_value=mock_settings):
        from src.api.model_cost.usage import record_usage

        # Should NOT raise — errors are swallowed
        record_usage("gpt-4o-mini", "reasoning", 10, 5)
