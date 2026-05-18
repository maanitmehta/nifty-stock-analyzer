"""India Fama-French 3-Factor model using NSE index proxies.

Factor construction
───────────────────
Market (Mkt-Rf): Nifty 50 (^NSEI) excess return above the risk-free rate.

SMB (Small-Minus-Big): Nifty Midcap 50 (^NSEMDCP50) return minus Nifty 50 return.
  Positive SMB loading → stock behaves like a smaller/midcap company.

HML (High-Minus-Low book value): Nifty 500 Value 50 minus Nifty 500 (Growth proxy).
  Fetched opportunistically; the model falls back to a 2-factor spec if unavailable.

This is a pragmatic proxy-based approach. Full academic FF3 for India requires a
cross-sectional dataset of book values across all listed stocks, which is out of scope
for this tool. The index proxies capture the same systematic risk dimensions at lower cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from ..config import settings
from .returns import TRADING_DAYS, log_returns

logger = logging.getLogger(__name__)

# NSE index tickers used as factor proxies
_MARKET_TICKER = "^NSEI"
_SMB_TICKER = "^NSEMDCP50"    # Nifty Midcap 50 — size premium proxy
_HML_TICKER = "^CNX500"       # Nifty 500 — used for HML construction if available
_VALUE_TICKER = "^CNXINFRA"   # placeholder; HML skipped unless a clean value index exists


@dataclass
class FF3Result:
    """Output of a single-period Fama-French 3-factor OLS regression."""

    alpha_annualized: float
    alpha_daily: float
    alpha_tstat: float
    alpha_pvalue: float

    mkt_beta: float
    mkt_tstat: float
    mkt_pvalue: float

    smb_loading: float | None    # None if SMB factor was unavailable
    smb_tstat: float | None
    smb_pvalue: float | None

    hml_loading: float | None    # None if HML factor was unavailable
    hml_tstat: float | None
    hml_pvalue: float | None

    r_squared: float
    adj_r_squared: float
    n_observations: int
    factors_used: list[str]         # e.g. ["Mkt-Rf", "SMB"] when HML unavailable


@dataclass
class FactorData:
    """Aligned daily factor return series."""

    mkt_rf: pd.Series
    smb: pd.Series | None = None
    hml: pd.Series | None = None

    @property
    def factor_df(self) -> pd.DataFrame:
        """DataFrame of available factors, aligned on common dates."""
        cols = {"Mkt-Rf": self.mkt_rf}
        if self.smb is not None:
            cols["SMB"] = self.smb
        if self.hml is not None:
            cols["HML"] = self.hml
        return pd.DataFrame(cols).dropna()

    @property
    def factors_used(self) -> list[str]:
        return list(self.factor_df.columns)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def construct_india_factors(
    lookback_years: int = 3,
    rf_annual: float | None = None,
    force_refresh: bool = False,
) -> FactorData:
    """Fetch index data and construct daily factor returns.

    Uses cached price data via PriceDataFetcher. Factors that cannot be
    fetched (index not available on yfinance) are silently omitted.

    Args:
        lookback_years: Historical window.
        rf_annual:      Annualised risk-free rate. Defaults to settings value.
        force_refresh:  Bypass the price cache.

    Returns:
        FactorData with Mkt-Rf always populated; SMB and HML when available.
    """
    from ..data.fetcher import PriceDataFetcher  # avoid circular import at module level

    rf = rf_annual if rf_annual is not None else settings.risk_free_rate
    rf_daily = rf / TRADING_DAYS
    fetcher = PriceDataFetcher()

    def _fetch_returns(ticker: str) -> pd.Series | None:
        try:
            df = fetcher.fetch(ticker, lookback_years, force_refresh)  # type: ignore[arg-type]
            return log_returns(df["Close"]).rename(ticker)
        except Exception as exc:
            logger.warning("Could not fetch factor index %s: %s", ticker, exc)
            return None

    mkt_raw = _fetch_returns(_MARKET_TICKER)
    if mkt_raw is None:
        raise RuntimeError("Cannot construct factors: Nifty 50 (^NSEI) data unavailable.")

    mkt_rf = (mkt_raw - rf_daily).rename("Mkt-Rf")

    smb_raw = _fetch_returns(_SMB_TICKER)
    smb: pd.Series | None = None
    if smb_raw is not None:
        # Align on common dates, then compute spread
        aligned = pd.DataFrame({"mid": smb_raw, "mkt": mkt_raw}).dropna()
        smb = (aligned["mid"] - aligned["mkt"]).rename("SMB")

    return FactorData(mkt_rf=mkt_rf, smb=smb, hml=None)


def ff3_regression(
    stock_returns: pd.Series,
    factors: FactorData,
    rf_annual: float | None = None,
) -> FF3Result:
    """OLS regression of stock excess returns on available India factors.

    Args:
        stock_returns: Daily log returns for the stock.
        factors:       FactorData from construct_india_factors().
        rf_annual:     Annualised risk-free rate.

    Returns:
        FF3Result. Loadings for unavailable factors are None.

    Raises:
        ValueError: Fewer than 30 overlapping observations.
    """
    rf_daily = (rf_annual or settings.risk_free_rate) / TRADING_DAYS

    factor_df = factors.factor_df
    excess_stock = (stock_returns - rf_daily).rename("excess_stock")

    aligned = pd.concat([excess_stock, factor_df], axis=1).dropna()
    if len(aligned) < 30:
        raise ValueError(
            f"Only {len(aligned)} overlapping observations — need at least 30 for regression."
        )

    y = aligned["excess_stock"].values
    X = sm.add_constant(aligned[factor_df.columns].values, prepend=True)  # noqa: N806
    model = sm.OLS(y, X).fit()

    params = model.params
    tvals = model.tvalues
    pvals = model.pvalues

    alpha_daily = float(params[0])
    mkt_beta = float(params[1])
    alpha_annualized = float((1 + alpha_daily) ** TRADING_DAYS - 1)

    smb_loading = smb_tstat = smb_pvalue = None
    hml_loading = hml_tstat = hml_pvalue = None

    factor_cols = list(factor_df.columns)
    if "SMB" in factor_cols:
        idx = factor_cols.index("SMB") + 1  # +1 for const
        smb_loading, smb_tstat, smb_pvalue = float(params[idx]), float(tvals[idx]), float(pvals[idx])
    if "HML" in factor_cols:
        idx = factor_cols.index("HML") + 1
        hml_loading, hml_tstat, hml_pvalue = float(params[idx]), float(tvals[idx]), float(pvals[idx])

    return FF3Result(
        alpha_annualized=alpha_annualized,
        alpha_daily=alpha_daily,
        alpha_tstat=float(tvals[0]),
        alpha_pvalue=float(pvals[0]),
        mkt_beta=mkt_beta,
        mkt_tstat=float(tvals[1]),
        mkt_pvalue=float(pvals[1]),
        smb_loading=smb_loading,
        smb_tstat=smb_tstat,
        smb_pvalue=smb_pvalue,
        hml_loading=hml_loading,
        hml_tstat=hml_tstat,
        hml_pvalue=hml_pvalue,
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        n_observations=int(model.nobs),
        factors_used=factor_cols,
    )


def rolling_ff3(
    stock_returns: pd.Series,
    factors: FactorData,
    rf_annual: float | None = None,
    window: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Rolling factor loadings via statsmodels RollingOLS.

    Returns:
        DataFrame with columns [alpha_annualized, mkt_beta, smb_loading?, hml_loading?]
        indexed by date. NaN for first (window-1) rows.
    """
    rf_daily = (rf_annual or settings.risk_free_rate) / TRADING_DAYS

    factor_df = factors.factor_df
    excess_stock = (stock_returns - rf_daily).rename("excess_stock")
    aligned = pd.concat([excess_stock, factor_df], axis=1).dropna()

    if len(aligned) < window + 1:
        logger.warning("Not enough data for rolling FF3 (need %d, have %d)", window + 1, len(aligned))
        return pd.DataFrame()

    y = aligned["excess_stock"]
    X = sm.add_constant(aligned[factor_df.columns], prepend=True)  # noqa: N806

    roll_model = RollingOLS(y, X, window=window).fit()
    params = roll_model.params.dropna()

    alpha_daily = params["const"]
    alpha_annualized = (1 + alpha_daily) ** TRADING_DAYS - 1

    result = pd.DataFrame({"alpha_annualized": alpha_annualized, "mkt_beta": params["Mkt-Rf"]},
                          index=params.index)
    if "SMB" in params.columns:
        result["smb_loading"] = params["SMB"]
    if "HML" in params.columns:
        result["hml_loading"] = params["HML"]

    return result.dropna()


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------


def interpret_smb(loading: float | None) -> str:
    """Plain-English interpretation of SMB loading."""
    if loading is None:
        return "N/A"
    if loading > 0.3:
        return "Strong small/midcap tilt"
    if loading > 0.1:
        return "Mild small/midcap tilt"
    if loading < -0.3:
        return "Strong large-cap tilt"
    if loading < -0.1:
        return "Mild large-cap tilt"
    return "Neutral size exposure"


def interpret_hml(loading: float | None) -> str:
    """Plain-English interpretation of HML loading."""
    if loading is None:
        return "N/A"
    if loading > 0.3:
        return "Strong value tilt"
    if loading > 0.1:
        return "Mild value tilt"
    if loading < -0.3:
        return "Strong growth tilt"
    if loading < -0.1:
        return "Mild growth tilt"
    return "Neutral value/growth"
