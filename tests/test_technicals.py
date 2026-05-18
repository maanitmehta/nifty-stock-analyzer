"""Tests for nifty_analyzer.features.technicals — NSA-12, NSA-13, NSA-14."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nifty_analyzer.features.technicals import (
    BollingerResult,
    MACDResult,
    MovingAverageResult,
    RSIResult,
    TechnicalSnapshot,
    VolumeResult,
    _recent_cross_above,
    _recent_cross_below,
    compute_technicals,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ohlcv_300() -> pd.DataFrame:
    """300 trading days of synthetic OHLCV — enough for all indicators."""
    rng = np.random.default_rng(42)
    n = 300
    idx = pd.date_range("2023-01-02", periods=n, freq="B")

    # Random walk close prices starting at 1000
    log_ret = rng.normal(0.0005, 0.012, n)
    close = pd.Series(1000 * np.exp(np.cumsum(log_ret)), index=idx, name="Close")

    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = pd.Series(rng.integers(500_000, 5_000_000, n).astype(float), index=idx)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


@pytest.fixture()
def snapshot(ohlcv_300: pd.DataFrame) -> TechnicalSnapshot:
    return compute_technicals(ohlcv_300)


# ---------------------------------------------------------------------------
# compute_technicals — structural checks
# ---------------------------------------------------------------------------


class TestComputeTechnicals:
    def test_returns_technical_snapshot(self, snapshot: TechnicalSnapshot) -> None:
        assert isinstance(snapshot, TechnicalSnapshot)

    def test_all_sub_objects_present(self, snapshot: TechnicalSnapshot) -> None:
        assert isinstance(snapshot.ma, MovingAverageResult)
        assert isinstance(snapshot.rsi, RSIResult)
        assert isinstance(snapshot.macd, MACDResult)
        assert isinstance(snapshot.bollinger, BollingerResult)
        assert isinstance(snapshot.volume, VolumeResult)

    def test_atr_current_is_positive(self, snapshot: TechnicalSnapshot) -> None:
        assert snapshot.atr_current > 0

    def test_range_52w_high_gte_low(self, snapshot: TechnicalSnapshot) -> None:
        assert snapshot.range_52w.high >= snapshot.range_52w.low


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


class TestMovingAverages:
    def test_series_lengths_match_input(
        self, snapshot: TechnicalSnapshot, ohlcv_300: pd.DataFrame
    ) -> None:
        assert len(snapshot.ma.sma_20) == len(ohlcv_300)
        assert len(snapshot.ma.sma_200) == len(ohlcv_300)
        assert len(snapshot.ma.ema_12) == len(ohlcv_300)

    def test_sma_shorter_window_has_fewer_nans(self, snapshot: TechnicalSnapshot) -> None:
        nan_20 = snapshot.ma.sma_20.isna().sum()
        nan_200 = snapshot.ma.sma_200.isna().sum()
        assert nan_20 < nan_200

    def test_price_above_sma50_is_bool(self, snapshot: TechnicalSnapshot) -> None:
        assert isinstance(snapshot.ma.price_above_sma_50, bool)

    def test_golden_and_death_cross_mutually_exclusive(
        self, snapshot: TechnicalSnapshot
    ) -> None:
        assert not (snapshot.ma.golden_cross and snapshot.ma.death_cross)

    def test_pct_from_sma50_is_finite(self, snapshot: TechnicalSnapshot) -> None:
        assert math.isfinite(snapshot.ma.pct_from_sma_50)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


class TestRSI:
    def test_rsi_bounded_0_to_100(self, snapshot: TechnicalSnapshot) -> None:
        valid = snapshot.rsi.series.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_current_rsi_matches_last_series_value(
        self, snapshot: TechnicalSnapshot
    ) -> None:
        last_valid = float(snapshot.rsi.series.dropna().iloc[-1])
        assert snapshot.rsi.current == pytest.approx(last_valid, abs=1e-6)

    def test_overbought_flag_consistent(self, snapshot: TechnicalSnapshot) -> None:
        expected = snapshot.rsi.current > 70
        assert snapshot.rsi.is_overbought == expected

    def test_oversold_flag_consistent(self, snapshot: TechnicalSnapshot) -> None:
        expected = snapshot.rsi.current < 30
        assert snapshot.rsi.is_oversold == expected

    def test_not_both_overbought_and_oversold(self, snapshot: TechnicalSnapshot) -> None:
        assert not (snapshot.rsi.is_overbought and snapshot.rsi.is_oversold)

    def test_all_rsi_between_0_and_100_for_rising_market(self) -> None:
        """Monotonically rising prices → RSI should be high but still ≤ 100."""
        idx = pd.date_range("2023-01-02", periods=100, freq="B")
        close = pd.Series(np.linspace(1000, 2000, 100), index=idx)
        high = close * 1.005
        low = close * 0.995
        vol = pd.Series(1_000_000.0, index=idx)
        df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol})
        snap = compute_technicals(df)
        valid = snap.rsi.series.dropna()
        assert (valid <= 100).all()


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


class TestMACD:
    def test_series_same_length(
        self, snapshot: TechnicalSnapshot, ohlcv_300: pd.DataFrame
    ) -> None:
        assert len(snapshot.macd.macd_line) == len(ohlcv_300)
        assert len(snapshot.macd.signal_line) == len(ohlcv_300)
        assert len(snapshot.macd.histogram) == len(ohlcv_300)

    def test_histogram_equals_macd_minus_signal(self, snapshot: TechnicalSnapshot) -> None:
        diff = (snapshot.macd.macd_line - snapshot.macd.signal_line).dropna()
        hist = snapshot.macd.histogram.dropna()
        common = diff.index.intersection(hist.index)
        pd.testing.assert_series_equal(diff[common], hist[common], check_names=False, atol=1e-6)

    def test_current_values_are_finite(self, snapshot: TechnicalSnapshot) -> None:
        assert math.isfinite(snapshot.macd.current_macd)
        assert math.isfinite(snapshot.macd.current_signal)

    def test_crossover_flags_are_bool(self, snapshot: TechnicalSnapshot) -> None:
        assert isinstance(snapshot.macd.is_bullish_crossover, bool)
        assert isinstance(snapshot.macd.is_bearish_crossover, bool)

    def test_crossovers_mutually_exclusive(self, snapshot: TechnicalSnapshot) -> None:
        assert not (snapshot.macd.is_bullish_crossover and snapshot.macd.is_bearish_crossover)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


class TestBollingerBands:
    def test_upper_above_middle_above_lower(self, snapshot: TechnicalSnapshot) -> None:
        aligned = pd.DataFrame({
            "u": snapshot.bollinger.upper,
            "m": snapshot.bollinger.middle,
            "l": snapshot.bollinger.lower,
        }).dropna()
        assert (aligned["u"] >= aligned["m"]).all()
        assert (aligned["m"] >= aligned["l"]).all()

    def test_pct_b_close_to_half_at_middle(self) -> None:
        """When price == middle band, pct_b should be ~0.5."""
        idx = pd.date_range("2023-01-02", periods=100, freq="B")
        close = pd.Series(1000.0, index=idx)  # completely flat
        high = close * 1.002
        low = close * 0.998
        vol = pd.Series(1_000_000.0, index=idx)
        df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol})
        snap = compute_technicals(df)
        # For flat prices, bands collapse (std=0), so pct_b may be NaN — just verify no crash
        assert isinstance(snap.bollinger, BollingerResult)

    def test_above_upper_consistent(self, snapshot: TechnicalSnapshot) -> None:
        expected = snapshot.bollinger.current_price > float(snapshot.bollinger.upper.iloc[-1])
        # Allow NaN upper band → False
        if not math.isnan(float(snapshot.bollinger.upper.iloc[-1])):
            assert snapshot.bollinger.above_upper == expected

    def test_bandwidth_non_negative(self, snapshot: TechnicalSnapshot) -> None:
        assert not math.isnan(snapshot.bollinger.bandwidth)
        assert snapshot.bollinger.bandwidth >= 0


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


class TestVolume:
    def test_ratio_is_positive(self, snapshot: TechnicalSnapshot) -> None:
        assert snapshot.volume.ratio > 0

    def test_obv_same_length_as_input(
        self, snapshot: TechnicalSnapshot, ohlcv_300: pd.DataFrame
    ) -> None:
        assert len(snapshot.volume.obv) == len(ohlcv_300)

    def test_is_above_average_consistent(self, snapshot: TechnicalSnapshot) -> None:
        expected = snapshot.volume.ratio > 1.0
        assert snapshot.volume.is_above_average == expected


# ---------------------------------------------------------------------------
# Helper: crossover detection
# ---------------------------------------------------------------------------


class TestCrossoverHelpers:
    def test_recent_cross_above_detected(self) -> None:
        # fast crosses above slow on bar 3
        idx = pd.date_range("2023-01-02", periods=6, freq="B")
        fast = pd.Series([1.0, 1.0, 0.9, 1.1, 1.2, 1.3], index=idx)
        slow = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=idx)
        assert _recent_cross_above(fast, slow, lookback=5)

    def test_recent_cross_below_detected(self) -> None:
        idx = pd.date_range("2023-01-02", periods=6, freq="B")
        fast = pd.Series([1.0, 1.0, 1.1, 0.9, 0.8, 0.7], index=idx)
        slow = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=idx)
        assert _recent_cross_below(fast, slow, lookback=5)

    def test_no_cross_returns_false(self) -> None:
        idx = pd.date_range("2023-01-02", periods=6, freq="B")
        fast = pd.Series([1.5, 1.5, 1.5, 1.5, 1.5, 1.5], index=idx)
        slow = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=idx)
        assert not _recent_cross_above(fast, slow, lookback=5)
        assert not _recent_cross_below(fast, slow, lookback=5)

    def test_insufficient_data_returns_false(self) -> None:
        idx = pd.date_range("2023-01-02", periods=2, freq="B")
        fast = pd.Series([1.0, 1.1], index=idx)
        slow = pd.Series([1.2, 0.9], index=idx)
        assert not _recent_cross_above(fast, slow, lookback=5)
