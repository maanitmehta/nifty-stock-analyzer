"""Drawdown analysis: max drawdown, drawdown series, duration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DrawdownResult:
    max_drawdown: float           # largest peak-to-trough loss, e.g. -0.35 = -35%
    drawdown_series: pd.Series    # rolling drawdown at each date (0 at peaks, negative in troughs)
    peak_date: pd.Timestamp | None
    trough_date: pd.Timestamp | None
    duration_days: int            # calendar days from peak to trough of the worst drawdown


def compute_drawdown(prices: pd.Series) -> DrawdownResult:
    """Compute max drawdown and drawdown time-series from a price series.

    Args:
        prices: Daily adjusted close prices with a DatetimeIndex, sorted ascending.

    Returns:
        DrawdownResult with the largest peak-to-trough drawdown and associated dates.
    """
    if len(prices) < 2:
        return DrawdownResult(
            max_drawdown=0.0,
            drawdown_series=pd.Series(dtype=float),
            peak_date=None,
            trough_date=None,
            duration_days=0,
        )

    running_max = prices.cummax()
    dd_series = (prices - running_max) / running_max
    dd_series.name = "drawdown"

    mdd = float(dd_series.min())

    if mdd == 0.0:
        return DrawdownResult(
            max_drawdown=0.0,
            drawdown_series=dd_series,
            peak_date=None,
            trough_date=None,
            duration_days=0,
        )

    trough_date: pd.Timestamp = dd_series.idxmin()
    # Peak is the last all-time high on or before the trough
    pre_trough_max = running_max.loc[:trough_date]
    peak_date: pd.Timestamp = pre_trough_max.idxmax()

    duration = int((trough_date - peak_date).days)

    return DrawdownResult(
        max_drawdown=mdd,
        drawdown_series=dd_series,
        peak_date=peak_date,
        trough_date=trough_date,
        duration_days=duration,
    )
