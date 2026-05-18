"""High-level orchestrator: compute all risk metrics for a stock in one call."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .capm import (
    CAPMResult,
    RatioMetrics,
    calmar_ratio,
    capm_regression,
    rolling_capm,
    sharpe_ratio,
    sortino_ratio,
)
from .returns import (
    TRADING_DAYS,
    annualized_return,
    annualized_std,
    historical_var,
    log_returns,
    rolling_volatility,
    trim_to_lookback,
)
from .risk import DrawdownResult, compute_drawdown


@dataclass
class RiskSnapshot:
    """All risk/return metrics for a single stock over a given lookback window."""

    ticker: str
    lookback_years: int

    # Return & volatility
    annualized_return: float
    annualized_std: float
    rolling_vol_30d: pd.Series
    rolling_vol_90d: pd.Series

    # VaR
    var_95: float
    var_99: float

    # Drawdown
    drawdown: DrawdownResult

    # CAPM
    capm: CAPMResult | None
    rolling_capm_df: pd.DataFrame   # columns: alpha_annualized, beta

    # Ratios
    ratios: RatioMetrics

    # Raw series (available for charting)
    returns: pd.Series
    prices: pd.Series


def compute_snapshot(
    ticker: str,
    stock_prices: pd.DataFrame,
    market_prices: pd.DataFrame,
    lookback_years: int = 3,
    rf_annual: float | None = None,
) -> RiskSnapshot:
    """Compute all Sprint-2 risk metrics for a stock.

    Args:
        ticker:         Stock identifier string (e.g. 'RELIANCE.NS').
        stock_prices:   OHLCV DataFrame from PriceDataFetcher (must have 'Close' column).
        market_prices:  OHLCV DataFrame for Nifty 50 benchmark (must have 'Close').
        lookback_years: Window to trim prices to before computing metrics.
        rf_annual:      Override risk-free rate; defaults to settings value.

    Returns:
        RiskSnapshot dataclass with all metrics pre-computed.
    """
    # --- Extract and trim close price series ---
    stock_close = trim_to_lookback(stock_prices["Close"], lookback_years)
    market_close = trim_to_lookback(market_prices["Close"], lookback_years)

    stock_ret = log_returns(stock_close)
    market_ret = log_returns(market_close)

    # --- Returns & volatility ---
    ann_ret = annualized_return(stock_ret)
    ann_std = annualized_std(stock_ret)
    vol_30 = rolling_volatility(stock_ret, 30)
    vol_90 = rolling_volatility(stock_ret, 90)

    # --- VaR ---
    var_95 = historical_var(stock_ret, 0.95)
    var_99 = historical_var(stock_ret, 0.99)

    # --- Drawdown ---
    dd = compute_drawdown(stock_close)

    # --- CAPM ---
    capm: CAPMResult | None = None
    try:
        capm = capm_regression(stock_ret, market_ret, rf_annual)
    except ValueError:
        pass  # too few observations — capm stays None

    roll_capm_window = min(TRADING_DAYS, len(stock_ret) - 1)
    roll_df = rolling_capm(stock_ret, market_ret, rf_annual, window=roll_capm_window)

    # --- Ratios ---
    ratios = RatioMetrics(
        sharpe=sharpe_ratio(stock_ret, rf_annual),
        sortino=sortino_ratio(stock_ret, rf_annual),
        calmar=calmar_ratio(stock_ret, stock_close),
    )

    return RiskSnapshot(
        ticker=ticker,
        lookback_years=lookback_years,
        annualized_return=ann_ret,
        annualized_std=ann_std,
        rolling_vol_30d=vol_30,
        rolling_vol_90d=vol_90,
        var_95=var_95,
        var_99=var_99,
        drawdown=dd,
        capm=capm,
        rolling_capm_df=roll_df,
        ratios=ratios,
        returns=stock_ret,
        prices=stock_close,
    )
