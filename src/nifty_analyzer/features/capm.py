"""CAPM regression, rolling alpha/beta, and risk-adjusted return ratios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..config import settings
from .returns import TRADING_DAYS
from .risk import compute_drawdown


@dataclass
class CAPMResult:
    """Full output of a single-period CAPM OLS regression."""

    alpha_annualized: float   # Jensen's Alpha, annualized
    alpha_daily: float        # raw regression intercept (daily)
    alpha_tstat: float
    alpha_pvalue: float
    beta: float               # market sensitivity
    beta_tstat: float
    beta_pvalue: float
    r_squared: float
    adj_r_squared: float
    n_observations: int


@dataclass
class RatioMetrics:
    sharpe: float | None
    sortino: float | None
    calmar: float | None


# ---------------------------------------------------------------------------
# CAPM regression
# ---------------------------------------------------------------------------


def capm_regression(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    rf_annual: float | None = None,
) -> CAPMResult:
    """OLS regression of daily stock excess returns on daily market excess returns.

    Uses the Nifty 50 index as the market proxy and an annualized risk-free
    rate (RBI 91-day T-bill by default) converted to a daily rate.

    Args:
        stock_returns:  Daily log returns for the stock.
        market_returns: Daily log returns for the market benchmark (^NSEI).
        rf_annual:      Annualized risk-free rate. Defaults to settings value.

    Returns:
        CAPMResult with annualized alpha, beta, R², and inference statistics.

    Raises:
        ValueError: If fewer than 30 overlapping observations are available.
    """
    rf_daily = _daily_rf(rf_annual)

    aligned = pd.DataFrame({"stock": stock_returns, "market": market_returns}).dropna()

    if len(aligned) < 30:
        raise ValueError(
            f"Only {len(aligned)} overlapping observations — need at least 30 for CAPM."
        )

    excess_stock = aligned["stock"] - rf_daily
    excess_market = aligned["market"] - rf_daily

    X = sm.add_constant(excess_market.values, prepend=True)  # noqa: N806
    model = sm.OLS(excess_stock.values, X).fit()

    alpha_daily = float(model.params[0])
    beta = float(model.params[1])
    alpha_annualized = float((1 + alpha_daily) ** TRADING_DAYS - 1)

    return CAPMResult(
        alpha_annualized=alpha_annualized,
        alpha_daily=alpha_daily,
        alpha_tstat=float(model.tvalues[0]),
        alpha_pvalue=float(model.pvalues[0]),
        beta=beta,
        beta_tstat=float(model.tvalues[1]),
        beta_pvalue=float(model.pvalues[1]),
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        n_observations=int(model.nobs),
    )


def rolling_capm(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    rf_annual: float | None = None,
    window: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Fast rolling alpha and beta using the covariance decomposition of OLS.

    beta  = cov(r_s - rf, r_m - rf) / var(r_m - rf)
    alpha = mean(r_s - rf) - beta * mean(r_m - rf)   [daily, then annualized]

    This avoids looping statsmodels fits and is O(n) per window step.

    Returns:
        DataFrame[alpha_annualized, beta] with date index. NaN for first
        (window - 1) rows.
    """
    rf_daily = _daily_rf(rf_annual)

    aligned = pd.DataFrame({"stock": stock_returns, "market": market_returns}).dropna()
    excess_stock = aligned["stock"] - rf_daily
    excess_market = aligned["market"] - rf_daily

    roll_cov = excess_stock.rolling(window).cov(excess_market)
    roll_var = excess_market.rolling(window).var(ddof=1)

    beta = roll_cov / roll_var
    alpha_daily = (
        excess_stock.rolling(window).mean() - beta * excess_market.rolling(window).mean()
    )
    alpha_annualized = (1 + alpha_daily) ** TRADING_DAYS - 1

    return pd.DataFrame(
        {"alpha_annualized": alpha_annualized, "beta": beta},
        index=aligned.index,
    ).dropna()


# ---------------------------------------------------------------------------
# Risk-adjusted return ratios
# ---------------------------------------------------------------------------


def sharpe_ratio(
    returns: pd.Series,
    rf_annual: float | None = None,
    trading_days: int = TRADING_DAYS,
) -> float | None:
    """Sharpe ratio: (annualized_return - Rf) / annualized_std.

    Returns None if the series has fewer than 2 observations or zero variance.
    """
    if len(returns) < 2:
        return None
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return None
    rf = rf_annual if rf_annual is not None else settings.risk_free_rate
    ann_ret = float(np.exp(returns.mean() * trading_days) - 1)
    ann_std = float(std * np.sqrt(trading_days))
    return (ann_ret - rf) / ann_std


def sortino_ratio(
    returns: pd.Series,
    rf_annual: float | None = None,
    trading_days: int = TRADING_DAYS,
) -> float | None:
    """Sortino ratio: (annualized_return - Rf) / annualized_downside_std.

    Downside std uses only returns below zero (target = 0).
    Returns None if the series has no negative returns or fewer than 2 observations.
    """
    if len(returns) < 2:
        return None
    rf = rf_annual if rf_annual is not None else settings.risk_free_rate
    downside = returns[returns < 0]
    if len(downside) < 2:
        return None
    downside_std = float(downside.std(ddof=1) * np.sqrt(trading_days))
    if downside_std == 0 or np.isnan(downside_std):
        return None
    ann_ret = float(np.exp(returns.mean() * trading_days) - 1)
    return (ann_ret - rf) / downside_std


def calmar_ratio(
    returns: pd.Series,
    prices: pd.Series,
    trading_days: int = TRADING_DAYS,
) -> float | None:
    """Calmar ratio: annualized_return / |max_drawdown|.

    Returns None if max drawdown is zero (no loss period) or the series is too short.
    """
    if len(returns) < 2 or len(prices) < 2:
        return None
    ann_ret = float(np.exp(returns.mean() * trading_days) - 1)
    mdd = compute_drawdown(prices).max_drawdown
    if mdd == 0.0:
        return None
    return ann_ret / abs(mdd)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _daily_rf(rf_annual: float | None) -> float:
    annual = rf_annual if rf_annual is not None else settings.risk_free_rate
    return annual / TRADING_DAYS
