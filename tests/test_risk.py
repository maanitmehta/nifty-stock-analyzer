"""Tests for nifty_analyzer.features.risk — NSA-7 (drawdown)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_analyzer.features.risk import compute_drawdown


@pytest.fixture()
def flat_prices() -> pd.Series:
    idx = pd.date_range("2022-01-03", periods=100, freq="B")
    return pd.Series(100.0, index=idx)


@pytest.fixture()
def crash_recovery_prices() -> pd.Series:
    """Rises to 150, crashes to 90, recovers to 120."""
    idx = pd.date_range("2022-01-03", periods=6, freq="B")
    return pd.Series([100.0, 120.0, 150.0, 90.0, 110.0, 120.0], index=idx)


@pytest.fixture()
def monotone_declining() -> pd.Series:
    idx = pd.date_range("2022-01-03", periods=10, freq="B")
    return pd.Series(np.linspace(100, 50, 10), index=idx)


class TestComputeDrawdown:
    def test_flat_prices_zero_drawdown(self, flat_prices: pd.Series) -> None:
        result = compute_drawdown(flat_prices)
        assert result.max_drawdown == pytest.approx(0.0)
        assert result.duration_days == 0

    def test_crash_recovery_max_drawdown(self, crash_recovery_prices: pd.Series) -> None:
        result = compute_drawdown(crash_recovery_prices)
        # Peak = 150, trough = 90 → drawdown = (90 - 150) / 150 = -0.40
        assert result.max_drawdown == pytest.approx(-0.40, abs=1e-6)

    def test_crash_recovery_peak_date(self, crash_recovery_prices: pd.Series) -> None:
        result = compute_drawdown(crash_recovery_prices)
        assert result.peak_date == crash_recovery_prices.index[2]  # 150 is at index 2

    def test_crash_recovery_trough_date(self, crash_recovery_prices: pd.Series) -> None:
        result = compute_drawdown(crash_recovery_prices)
        assert result.trough_date == crash_recovery_prices.index[3]  # 90 is at index 3

    def test_duration_is_positive(self, crash_recovery_prices: pd.Series) -> None:
        result = compute_drawdown(crash_recovery_prices)
        assert result.duration_days > 0

    def test_monotone_decline_peak_is_first(self, monotone_declining: pd.Series) -> None:
        result = compute_drawdown(monotone_declining)
        assert result.peak_date == monotone_declining.index[0]
        assert result.trough_date == monotone_declining.index[-1]
        assert result.max_drawdown < -0.40

    def test_drawdown_series_length(self, crash_recovery_prices: pd.Series) -> None:
        result = compute_drawdown(crash_recovery_prices)
        assert len(result.drawdown_series) == len(crash_recovery_prices)

    def test_drawdown_series_never_positive(self, crash_recovery_prices: pd.Series) -> None:
        result = compute_drawdown(crash_recovery_prices)
        assert (result.drawdown_series <= 1e-10).all()

    def test_drawdown_at_all_time_high_is_zero(self, crash_recovery_prices: pd.Series) -> None:
        result = compute_drawdown(crash_recovery_prices)
        # First obs and the peak (index 2) should be 0
        assert result.drawdown_series.iloc[0] == pytest.approx(0.0, abs=1e-10)
        assert result.drawdown_series.iloc[2] == pytest.approx(0.0, abs=1e-10)

    def test_single_price_returns_zero_drawdown(self) -> None:
        prices = pd.Series([100.0], index=pd.date_range("2023-01-02", periods=1))
        result = compute_drawdown(prices)
        assert result.max_drawdown == 0.0

    def test_empty_series_returns_zero(self) -> None:
        result = compute_drawdown(pd.Series(dtype=float))
        assert result.max_drawdown == 0.0
        assert result.peak_date is None
