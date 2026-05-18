"""Return and volatility calculations from daily adjusted close prices."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns from an adjusted close series. First row NaN is dropped."""
    return np.log(prices / prices.shift(1)).dropna()


def annualized_return(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Geometric annualized return from daily log returns.

    Converts total log return to a CAGR:
        CAGR = exp(mean_log_return * 252) - 1
    """
    n = len(returns)
    if n < 2:
        return float("nan")
    return float(np.exp(returns.mean() * trading_days) - 1)


def annualized_std(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Annualized realized volatility: daily_std * sqrt(252)."""
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(trading_days))


def rolling_volatility(
    returns: pd.Series, window: int, trading_days: int = TRADING_DAYS
) -> pd.Series:
    """Rolling annualized volatility. First (window-1) values are NaN."""
    return (returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)).rename(
        f"rolling_vol_{window}d"
    )


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical simulation VaR — a positive number representing the daily loss threshold.

    E.g. VaR 95% = 0.02 means "on 95% of days losses are below 2%".
    Requires at least 20 observations.
    """
    if len(returns) < 20:
        return float("nan")
    return float(-np.percentile(returns, (1 - confidence) * 100))


def trim_to_lookback(prices: pd.Series, lookback_years: int) -> pd.Series:
    """Return the trailing *lookback_years* of a daily price series."""
    cutoff = prices.index[-1] - pd.DateOffset(years=lookback_years)
    return prices[prices.index >= cutoff]
