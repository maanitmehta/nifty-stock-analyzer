"""Tests for nifty_analyzer.data.fetcher — NSA-4."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from nifty_analyzer.data.fetcher import PriceDataFetcher, _cache_key


class TestCacheKey:
    def test_format(self) -> None:
        assert _cache_key("RELIANCE.NS", 3) == "RELIANCE.NS_3y"
        assert _cache_key("^NSEI", 1) == "^NSEI_1y"


class TestPriceDataFetcher:
    # ------------------------------------------------------------------
    # Cache hit: _download should NOT be called
    # ------------------------------------------------------------------
    def test_returns_cached_data_when_fresh(
        self, sample_ohlcv: pd.DataFrame, tmp_cache_dir: Path
    ) -> None:
        fetcher = PriceDataFetcher()
        # Prime the cache manually
        fetcher._cache.put(_cache_key("RELIANCE.NS", 3), sample_ohlcv)
        # Force cache to appear fresh (mtime = now)
        cache_file = fetcher._cache._path(_cache_key("RELIANCE.NS", 3))
        cache_file.touch()

        with patch.object(fetcher, "_download") as mock_dl:
            result = fetcher.fetch("RELIANCE.NS", lookback_years=3)

        mock_dl.assert_not_called()
        assert not result.empty

    # ------------------------------------------------------------------
    # Cache miss: _download MUST be called and result cached
    # ------------------------------------------------------------------
    def test_downloads_when_cache_is_stale(
        self, sample_ohlcv: pd.DataFrame, tmp_cache_dir: Path
    ) -> None:
        fetcher = PriceDataFetcher()

        with patch.object(fetcher, "_download", return_value=sample_ohlcv) as mock_dl:
            result = fetcher.fetch("RELIANCE.NS", lookback_years=1)

        mock_dl.assert_called_once_with("RELIANCE.NS", 1)
        assert len(result) == len(sample_ohlcv)

    # ------------------------------------------------------------------
    # force_refresh bypasses a warm cache
    # ------------------------------------------------------------------
    def test_force_refresh_bypasses_cache(
        self, sample_ohlcv: pd.DataFrame, tmp_cache_dir: Path
    ) -> None:
        fetcher = PriceDataFetcher()
        fetcher._cache.put(_cache_key("TCS.NS", 3), sample_ohlcv)
        cache_file = fetcher._cache._path(_cache_key("TCS.NS", 3))
        cache_file.touch()

        with patch.object(fetcher, "_download", return_value=sample_ohlcv) as mock_dl:
            fetcher.fetch("TCS.NS", lookback_years=3, force_refresh=True)

        mock_dl.assert_called_once()

    # ------------------------------------------------------------------
    # Empty download → ValueError
    # ------------------------------------------------------------------
    def test_raises_on_empty_download(self, tmp_cache_dir: Path) -> None:
        fetcher = PriceDataFetcher()
        with patch.object(fetcher, "_download", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="no data"):
                fetcher.fetch("INVALIDTICKER.NS", lookback_years=1)

    # ------------------------------------------------------------------
    # fetch_benchmark delegates to the benchmark ticker from config
    # ------------------------------------------------------------------
    def test_fetch_benchmark_uses_nsei(
        self, sample_ohlcv: pd.DataFrame, tmp_cache_dir: Path
    ) -> None:
        fetcher = PriceDataFetcher()
        with patch.object(fetcher, "_download", return_value=sample_ohlcv) as mock_dl:
            fetcher.fetch_benchmark(lookback_years=1)

        mock_dl.assert_called_once_with("^NSEI", 1)

    # ------------------------------------------------------------------
    # fetch_many: bad tickers are skipped, not raised
    # ------------------------------------------------------------------
    def test_fetch_many_skips_bad_tickers(
        self, sample_ohlcv: pd.DataFrame, tmp_cache_dir: Path
    ) -> None:
        fetcher = PriceDataFetcher()

        def side_effect(ticker: str, lookback: int) -> pd.DataFrame:
            if ticker == "BAD.NS":
                return pd.DataFrame()  # triggers ValueError in fetch()
            return sample_ohlcv

        with patch.object(fetcher, "_download", side_effect=side_effect):
            results = fetcher.fetch_many(["RELIANCE.NS", "BAD.NS"], lookback_years=1)

        assert "RELIANCE.NS" in results
        assert "BAD.NS" not in results

    # ------------------------------------------------------------------
    # Output schema validation
    # ------------------------------------------------------------------
    def test_output_has_correct_columns(
        self, sample_ohlcv: pd.DataFrame, tmp_cache_dir: Path
    ) -> None:
        fetcher = PriceDataFetcher()
        with patch.object(fetcher, "_download", return_value=sample_ohlcv):
            df = fetcher.fetch("INFY.NS", lookback_years=1)

        assert set(df.columns) == {"Open", "High", "Low", "Close", "Volume"}
        assert df.index.name == "date"

    def test_output_index_is_datetime(
        self, sample_ohlcv: pd.DataFrame, tmp_cache_dir: Path
    ) -> None:
        fetcher = PriceDataFetcher()
        with patch.object(fetcher, "_download", return_value=sample_ohlcv):
            df = fetcher.fetch("WIPRO.NS", lookback_years=1)

        assert isinstance(df.index, pd.DatetimeIndex)

    def test_output_sorted_ascending(
        self, sample_ohlcv: pd.DataFrame, tmp_cache_dir: Path
    ) -> None:
        fetcher = PriceDataFetcher()
        shuffled = sample_ohlcv.sample(frac=1, random_state=42)
        with patch.object(fetcher, "_download", return_value=shuffled):
            df = fetcher.fetch("SBIN.NS", lookback_years=1)

        assert df.index.is_monotonic_increasing
