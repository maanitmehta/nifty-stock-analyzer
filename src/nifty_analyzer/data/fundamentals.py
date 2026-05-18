"""Fundamental data pipeline: fetch income statement, balance sheet, and key ratios via yfinance."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import yfinance as yf

from .cache import CacheManager

logger = logging.getLogger(__name__)

# Fundamentals change slowly; refresh weekly
_FUNDAMENTAL_CACHE_HOURS = 168


@dataclass
class FundamentalData:
    """All fetched fundamental data for one stock."""

    ticker: str

    # ── Valuation multiples ────────────────────────────────────────────────
    pe_trailing: float | None = None
    pe_forward: float | None = None
    pb: float | None = None
    ev_ebitda: float | None = None
    ps: float | None = None
    market_cap: float | None = None
    enterprise_value: float | None = None

    # ── Profitability & quality ────────────────────────────────────────────
    roe: float | None = None           # Return on Equity (TTM)
    roa: float | None = None           # Return on Assets (TTM)
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    interest_coverage: float | None = None

    # ── Growth ────────────────────────────────────────────────────────────
    revenue_growth_yoy: float | None = None
    earnings_growth_yoy: float | None = None

    # ── Raw financial statements (for Piotroski) ──────────────────────────
    financials: pd.DataFrame = field(default_factory=pd.DataFrame)    # income statement
    balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)

    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class FundamentalFetcher:
    """Fetch and cache fundamental data for NSE-listed stocks.

    Uses yfinance `.info` for valuation multiples and the financial statement
    DataFrames for Piotroski F-score computation. Data is cached as parquet/JSON
    for a week since it changes slowly.
    """

    def __init__(self) -> None:
        self._cache = CacheManager("fundamentals")

    def fetch(self, ticker: str, force_refresh: bool = False) -> FundamentalData:
        """Return FundamentalData for *ticker*, using cache when fresh.

        Args:
            ticker:        yfinance ticker string, e.g. 'RELIANCE.NS'.
            force_refresh: Bypass the cache and re-download.

        Returns:
            FundamentalData dataclass. Fields are None when yfinance lacks the data.
        """
        cache_key = f"fund_{ticker}"

        if not force_refresh and not self._cache.is_stale(cache_key, _FUNDAMENTAL_CACHE_HOURS):
            cached = self._cache.get(cache_key)
            if cached is not None and not cached.empty:
                logger.debug("Fundamental cache hit: %s", ticker)
                return _deserialise(cached, ticker)

        logger.info("Fetching fundamentals for %s", ticker)
        data = self._download(ticker)
        self._cache.put(cache_key, _serialise(data))
        return data

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _download(self, ticker: str) -> FundamentalData:
        try:
            yf_ticker = yf.Ticker(ticker)
            info: dict = yf_ticker.info or {}
        except Exception as exc:
            logger.warning("yfinance.info failed for %s: %s", ticker, exc)
            info = {}

        def _g(key: str) -> float | None:
            val = info.get(key)
            try:
                return float(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        # Operating income and interest expense for interest coverage
        ebit = _g("ebit")
        interest_exp = _g("interestExpense")
        interest_coverage: float | None = None
        if ebit is not None and interest_exp and interest_exp != 0:
            interest_coverage = ebit / abs(interest_exp)

        data = FundamentalData(
            ticker=ticker,
            pe_trailing=_g("trailingPE"),
            pe_forward=_g("forwardPE"),
            pb=_g("priceToBook"),
            ev_ebitda=_g("enterpriseToEbitda"),
            ps=_g("priceToSalesTrailingTwelveMonths"),
            market_cap=_g("marketCap"),
            enterprise_value=_g("enterpriseValue"),
            roe=_g("returnOnEquity"),
            roa=_g("returnOnAssets"),
            debt_to_equity=_g("debtToEquity"),
            current_ratio=_g("currentRatio"),
            gross_margin=_g("grossMargins"),
            operating_margin=_g("operatingMargins"),
            net_margin=_g("profitMargins"),
            interest_coverage=interest_coverage,
            revenue_growth_yoy=_g("revenueGrowth"),
            earnings_growth_yoy=_g("earningsGrowth"),
        )

        # Financial statements — best-effort, don't crash if unavailable
        try:
            data.financials = yf_ticker.financials
        except Exception:
            pass
        try:
            data.balance_sheet = yf_ticker.balance_sheet
        except Exception:
            pass
        try:
            data.cashflow = yf_ticker.cashflow
        except Exception:
            pass

        return data


# ---------------------------------------------------------------------------
# Cache serialisation helpers
# ---------------------------------------------------------------------------
# We store the scalar fields in a single-row parquet. The three statement
# DataFrames are stored as separate parquet files using a composite key.


def _serialise(data: FundamentalData) -> pd.DataFrame:
    """Pack scalar fields into a one-row DataFrame for parquet storage."""
    scalars = {k: v for k, v in data.__dict__.items()
               if not isinstance(v, pd.DataFrame) and k != "ticker"}
    return pd.DataFrame([scalars])


def _deserialise(df: pd.DataFrame, ticker: str) -> FundamentalData:
    """Unpack a one-row DataFrame back into FundamentalData (scalars only)."""
    row = df.iloc[0].to_dict()

    def _safe(key: str) -> float | None:
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    return FundamentalData(
        ticker=ticker,
        pe_trailing=_safe("pe_trailing"),
        pe_forward=_safe("pe_forward"),
        pb=_safe("pb"),
        ev_ebitda=_safe("ev_ebitda"),
        ps=_safe("ps"),
        market_cap=_safe("market_cap"),
        enterprise_value=_safe("enterprise_value"),
        roe=_safe("roe"),
        roa=_safe("roa"),
        debt_to_equity=_safe("debt_to_equity"),
        current_ratio=_safe("current_ratio"),
        gross_margin=_safe("gross_margin"),
        operating_margin=_safe("operating_margin"),
        net_margin=_safe("net_margin"),
        interest_coverage=_safe("interest_coverage"),
        revenue_growth_yoy=_safe("revenue_growth_yoy"),
        earnings_growth_yoy=_safe("earnings_growth_yoy"),
        fetched_at=str(row.get("fetched_at", "")),
        # Statements not cached — re-fetch if Piotroski is needed
    )
