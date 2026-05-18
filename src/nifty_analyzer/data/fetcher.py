"""Price data pipeline: fetch adjusted OHLCV from yfinance with parquet caching."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Literal

import pandas as pd
import yfinance as yf

from ..config import settings
from .cache import CacheManager

logger = logging.getLogger(__name__)

LookbackYears = Literal[1, 3, 5]

_LOOKBACK_DAYS: dict[int, int] = {1: 365, 3: 1095, 5: 1825}

# Extra buffer days to account for weekends / holidays / data latency
_BUFFER_DAYS = 45

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class PriceDataFetcher:
    """Fetch and cache adjusted OHLCV price data for NSE-listed stocks.

    Tickers must be in yfinance format:
        - Equities  → 'RELIANCE.NS'
        - Index     → '^NSEI'

    Cache keys encode both the ticker and the lookback window so that
    switching lookback periods triggers a fresh download.
    """

    def __init__(self) -> None:
        self._cache = CacheManager("prices")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        ticker: str,
        lookback_years: LookbackYears = 3,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return an adjusted OHLCV DataFrame for *ticker*.

        Columns: Open, High, Low, Close, Volume
        Index:   DatetimeIndex named 'date', timezone-naive, sorted ascending

        Raises ValueError if yfinance returns no data for the ticker.
        """
        cache_key = _cache_key(ticker, lookback_years)

        if not force_refresh and not self._cache.is_stale(cache_key):
            cached = self._cache.get(cache_key)
            if cached is not None and not cached.empty:
                logger.debug("Cache hit: %s", cache_key)
                return cached

        df = self._download(ticker, lookback_years)

        if df.empty:
            raise ValueError(
                f"yfinance returned no data for ticker {ticker!r}. "
                "Check that the symbol is correct and listed on NSE (use the .NS suffix)."
            )

        df = df.sort_index()
        self._cache.put(cache_key, df)
        return df

    def fetch_benchmark(
        self,
        lookback_years: LookbackYears = 3,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch the Nifty 50 index (^NSEI) as the market benchmark."""
        return self.fetch(settings.benchmark_ticker, lookback_years, force_refresh)

    def fetch_many(
        self,
        tickers: list[str],
        lookback_years: LookbackYears = 3,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Fetch multiple tickers, returning a dict of {ticker: DataFrame}.

        Tickers that fail (delisted, bad symbol) are logged and excluded from
        the result rather than raising.
        """
        results: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch(ticker, lookback_years, force_refresh)
            except (ValueError, RuntimeError) as exc:
                logger.warning("Skipping %s: %s", ticker, exc)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download(self, ticker: str, lookback_years: int) -> pd.DataFrame:
        end = date.today()
        start = end - timedelta(days=_LOOKBACK_DAYS[lookback_years] + _BUFFER_DAYS)

        logger.info("Downloading %s (%dY) from yfinance", ticker, lookback_years)

        try:
            raw: pd.DataFrame = yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=True,   # adjusts for splits and dividends
                progress=False,
                actions=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"yfinance download failed for {ticker!r}: {exc}"
            ) from exc

        if raw.empty:
            return raw

        # yfinance >=0.2.38 returns a MultiIndex when downloading a single ticker
        # via the batch path; flatten it when present.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Ensure all expected columns are present
        missing = [c for c in _OHLCV if c not in raw.columns]
        if missing:
            raise RuntimeError(
                f"yfinance response for {ticker!r} is missing columns: {missing}"
            )

        df = raw[_OHLCV].copy()

        # Normalise index
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"

        # Drop rows where Close is NaN (e.g. non-trading days in raw data)
        df = df.dropna(subset=["Close"])

        # Trim to the requested lookback (after the download buffer)
        cutoff = pd.Timestamp(end) - pd.Timedelta(days=_LOOKBACK_DAYS[lookback_years])
        df = df[df.index >= cutoff]

        return df


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------


def _cache_key(ticker: str, lookback_years: int) -> str:
    return f"{ticker}_{lookback_years}y"
