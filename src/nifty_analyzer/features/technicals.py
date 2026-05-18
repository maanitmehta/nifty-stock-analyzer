"""Technical indicator computations from OHLCV price data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MovingAverageResult:
    sma_20: pd.Series
    sma_50: pd.Series
    sma_100: pd.Series
    sma_200: pd.Series
    ema_12: pd.Series
    ema_26: pd.Series
    ema_50: pd.Series

    price_above_sma_50: bool
    price_above_sma_200: bool
    pct_from_sma_50: float     # positive = price above SMA50
    pct_from_sma_200: float
    golden_cross: bool          # SMA50 crossed above SMA200 within last 5 bars
    death_cross: bool           # SMA50 crossed below SMA200 within last 5 bars


@dataclass
class RSIResult:
    series: pd.Series
    current: float
    is_overbought: bool     # > 70
    is_oversold: bool       # < 30


@dataclass
class MACDResult:
    macd_line: pd.Series
    signal_line: pd.Series
    histogram: pd.Series
    current_macd: float
    current_signal: float
    current_histogram: float
    is_bullish_crossover: bool   # MACD crossed above signal within last 3 bars
    is_bearish_crossover: bool


@dataclass
class BollingerResult:
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series
    current_price: float
    above_upper: bool
    below_lower: bool
    pct_b: float        # 0 = at lower band, 1 = at upper band, >1 or <0 = outside
    bandwidth: float    # (upper - lower) / middle


@dataclass
class VolumeResult:
    current: float
    avg_20d: float
    ratio: float            # current / avg_20d
    is_above_average: bool
    obv: pd.Series


@dataclass
class Range52W:
    high: float
    low: float
    current: float
    pct_from_high: float    # negative: price is below 52W high
    pct_from_low: float     # positive: price is above 52W low


@dataclass
class TechnicalSnapshot:
    ma: MovingAverageResult
    rsi: RSIResult
    macd: MACDResult
    bollinger: BollingerResult
    atr: pd.Series          # 14-day ATR series
    atr_current: float
    volume: VolumeResult
    range_52w: Range52W


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_technicals(ohlcv: pd.DataFrame) -> TechnicalSnapshot:
    """Compute all technical indicators from an OHLCV DataFrame.

    Args:
        ohlcv: DataFrame with columns Open, High, Low, Close, Volume and a
               DatetimeIndex sorted ascending. Typically from PriceDataFetcher.

    Returns:
        TechnicalSnapshot with all indicator series and latest-bar scalar values.
    """
    close = ohlcv["Close"]
    high = ohlcv["High"]
    low = ohlcv["Low"]
    volume = ohlcv["Volume"]

    return TechnicalSnapshot(
        ma=_moving_averages(close),
        rsi=_rsi(close),
        macd=_macd(close),
        bollinger=_bollinger(close),
        atr=_atr_series(high, low, close),
        atr_current=_last(AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()),
        volume=_volume(close, volume),
        range_52w=_range_52w(close),
    )


# ---------------------------------------------------------------------------
# Individual indicator functions
# ---------------------------------------------------------------------------


def _moving_averages(close: pd.Series) -> MovingAverageResult:
    sma_20 = SMAIndicator(close=close, window=20).sma_indicator()
    sma_50 = SMAIndicator(close=close, window=50).sma_indicator()
    sma_100 = SMAIndicator(close=close, window=100).sma_indicator()
    sma_200 = SMAIndicator(close=close, window=200).sma_indicator()
    ema_12 = EMAIndicator(close=close, window=12).ema_indicator()
    ema_26 = EMAIndicator(close=close, window=26).ema_indicator()
    ema_50 = EMAIndicator(close=close, window=50).ema_indicator()

    current = float(close.iloc[-1])
    sma50_now = _last(sma_50)
    sma200_now = _last(sma_200)

    pct_from_sma_50 = (current / sma50_now - 1) if sma50_now and sma50_now != 0 else float("nan")
    pct_from_sma_200 = (current / sma200_now - 1) if sma200_now and sma200_now != 0 else float("nan")

    golden = _recent_cross_above(sma_50, sma_200, lookback=5)
    death = _recent_cross_below(sma_50, sma_200, lookback=5)

    return MovingAverageResult(
        sma_20=sma_20, sma_50=sma_50, sma_100=sma_100, sma_200=sma_200,
        ema_12=ema_12, ema_26=ema_26, ema_50=ema_50,
        price_above_sma_50=current > sma50_now if sma50_now else False,
        price_above_sma_200=current > sma200_now if sma200_now else False,
        pct_from_sma_50=pct_from_sma_50,
        pct_from_sma_200=pct_from_sma_200,
        golden_cross=golden,
        death_cross=death,
    )


def _rsi(close: pd.Series, window: int = 14) -> RSIResult:
    series = RSIIndicator(close=close, window=window).rsi()
    current = _last(series)
    return RSIResult(
        series=series,
        current=current,
        is_overbought=current > 70 if not np.isnan(current) else False,
        is_oversold=current < 30 if not np.isnan(current) else False,
    )


def _macd(close: pd.Series) -> MACDResult:
    ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = ind.macd()
    signal_line = ind.macd_signal()
    histogram = ind.macd_diff()

    curr_macd = _last(macd_line)
    curr_signal = _last(signal_line)
    curr_hist = _last(histogram)

    return MACDResult(
        macd_line=macd_line,
        signal_line=signal_line,
        histogram=histogram,
        current_macd=curr_macd,
        current_signal=curr_signal,
        current_histogram=curr_hist,
        is_bullish_crossover=_recent_cross_above(macd_line, signal_line, lookback=3),
        is_bearish_crossover=_recent_cross_below(macd_line, signal_line, lookback=3),
    )


def _bollinger(close: pd.Series, window: int = 20, dev: int = 2) -> BollingerResult:
    ind = BollingerBands(close=close, window=window, window_dev=dev)
    upper = ind.bollinger_hband()
    middle = ind.bollinger_mavg()
    lower = ind.bollinger_lband()

    current = float(close.iloc[-1])
    u = _last(upper)
    m = _last(middle)
    lo = _last(lower)

    pct_b = (current - lo) / (u - lo) if u and lo and (u - lo) != 0 else float("nan")
    bw = (u - lo) / m if m and m != 0 else float("nan")

    return BollingerResult(
        upper=upper, middle=middle, lower=lower,
        current_price=current,
        above_upper=current > u if u else False,
        below_lower=current < lo if lo else False,
        pct_b=pct_b,
        bandwidth=bw,
    )


def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return AverageTrueRange(high=high, low=low, close=close, window=window).average_true_range()


def _volume(close: pd.Series, volume: pd.Series) -> VolumeResult:
    obv = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    current_vol = float(volume.iloc[-1])
    avg_20d = float(volume.rolling(20).mean().iloc[-1])
    ratio = current_vol / avg_20d if avg_20d and avg_20d != 0 else float("nan")
    return VolumeResult(
        current=current_vol,
        avg_20d=avg_20d,
        ratio=ratio,
        is_above_average=ratio > 1.0 if not np.isnan(ratio) else False,
        obv=obv,
    )


def _range_52w(close: pd.Series) -> Range52W:
    window = min(252, len(close))
    recent = close.iloc[-window:]
    high_52w = float(recent.max())
    low_52w = float(recent.min())
    current = float(close.iloc[-1])
    return Range52W(
        high=high_52w,
        low=low_52w,
        current=current,
        pct_from_high=(current / high_52w - 1),
        pct_from_low=(current / low_52w - 1),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last(series: pd.Series) -> float:
    """Return the last non-NaN value of a series, or NaN if all are NaN."""
    valid = series.dropna()
    return float(valid.iloc[-1]) if len(valid) > 0 else float("nan")


def _recent_cross_above(fast: pd.Series, slow: pd.Series, lookback: int = 5) -> bool:
    """True if fast crossed above slow within the last *lookback* bars.

    Uses .values to avoid pandas index-label alignment when comparing shifted slices.
    """
    aligned = pd.DataFrame({"fast": fast, "slow": slow}).dropna()
    if len(aligned) < lookback + 1:
        return False
    tail = (aligned["fast"] > aligned["slow"]).values[-(lookback + 1):]
    return bool((~tail[:-1] & tail[1:]).any())


def _recent_cross_below(fast: pd.Series, slow: pd.Series, lookback: int = 5) -> bool:
    """True if fast crossed below slow within the last *lookback* bars."""
    aligned = pd.DataFrame({"fast": fast, "slow": slow}).dropna()
    if len(aligned) < lookback + 1:
        return False
    tail = (aligned["fast"] < aligned["slow"]).values[-(lookback + 1):]
    return bool((~tail[:-1] & tail[1:]).any())
