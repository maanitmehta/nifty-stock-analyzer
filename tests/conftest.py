"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """Minimal OHLCV DataFrame mimicking a yfinance response."""
    dates = pd.date_range("2023-01-02", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "Open":   [100.0, 101.0, 102.0, 101.5, 103.0],
            "High":   [102.0, 103.0, 104.0, 103.0, 105.0],
            "Low":    [ 99.0, 100.0, 101.0, 100.5, 102.0],
            "Close":  [101.0, 102.0, 103.0, 102.0, 104.0],
            "Volume": [1_000_000, 1_100_000, 900_000, 1_200_000, 950_000],
        },
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture()
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the cache to a throwaway temp directory for each test."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr("nifty_analyzer.config.settings.cache_dir", cache)
    return cache
