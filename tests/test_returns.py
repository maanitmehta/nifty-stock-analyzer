"""Tests for nifty_analyzer.features.returns — NSA-7."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nifty_analyzer.features.returns import (
    annualized_return,
    annualized_std,
    historical_var,
    log_returns,
    rolling_volatility,
    trim_to_lookback,
)


@pytest.fixture()
def flat_prices() -> pd.Series:
    """Price series that never changes — zero returns."""
    idx = pd.date_range("2022-01-03", periods=252, freq="B")
    return pd.Series(100.0, index=idx, name="Close")


@pytest.fixture()
def trending_prices() -> pd.Series:
    """Price series with exact +1% daily compound growth."""
    idx = pd.date_range("2022-01-03", periods=252, freq="B")
    prices = 100 * (1.01 ** np.arange(252))
    return pd.Series(prices, index=idx, name="Close")


@pytest.fixture()
def realistic_returns() -> pd.Series:
    """Random log returns with known seed for reproducibility."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2021-01-04", periods=756, freq="B")  # 3 years
    returns = pd.Series(rng.normal(0.0004, 0.012, 756), index=idx)
    returns.name = "returns"
    return returns


# -----------------------------------------------------------------------
# log_returns
# -----------------------------------------------------------------------


class TestLogReturns:
    def test_length_is_n_minus_1(self, trending_prices: pd.Series) -> None:
        r = log_returns(trending_prices)
        assert len(r) == len(trending_prices) - 1

    def test_no_nan_in_output(self, trending_prices: pd.Series) -> None:
        r = log_returns(trending_prices)
        assert not r.isna().any()

    def test_flat_prices_give_zero_returns(self, flat_prices: pd.Series) -> None:
        r = log_returns(flat_prices)
        assert (r == 0.0).all()

    def test_known_return_value(self) -> None:
        prices = pd.Series([100.0, 110.0], index=pd.date_range("2023-01-02", periods=2))
        r = log_returns(prices)
        expected = math.log(110 / 100)
        assert abs(float(r.iloc[0]) - expected) < 1e-10


# -----------------------------------------------------------------------
# annualized_return
# -----------------------------------------------------------------------


class TestAnnualizedReturn:
    def test_zero_returns_give_zero_annualized(self, flat_prices: pd.Series) -> None:
        r = log_returns(flat_prices)
        assert annualized_return(r) == pytest.approx(0.0, abs=1e-9)

    def test_positive_trend(self, trending_prices: pd.Series) -> None:
        r = log_returns(trending_prices)
        result = annualized_return(r)
        assert result > 0

    def test_short_series_returns_nan(self) -> None:
        r = pd.Series([0.01])
        assert math.isnan(annualized_return(r))

    def test_empty_series_returns_nan(self) -> None:
        assert math.isnan(annualized_return(pd.Series(dtype=float)))


# -----------------------------------------------------------------------
# annualized_std
# -----------------------------------------------------------------------


class TestAnnualizedStd:
    def test_zero_returns_give_zero_vol(self, flat_prices: pd.Series) -> None:
        r = log_returns(flat_prices)
        assert annualized_std(r) == pytest.approx(0.0, abs=1e-12)

    def test_vol_scales_with_sqrt_time(self) -> None:
        # Alternating [+k, -k] has mean=0 and std≈k (exactly k for even n, ddof=1)
        k = 0.01
        r = pd.Series([k, -k] * 126)  # 252 obs, std(ddof=1) ≈ k
        result = annualized_std(r)
        assert result == pytest.approx(k * math.sqrt(252), rel=0.01)

    def test_single_obs_returns_nan(self) -> None:
        assert math.isnan(annualized_std(pd.Series([0.01])))


# -----------------------------------------------------------------------
# rolling_volatility
# -----------------------------------------------------------------------


class TestRollingVolatility:
    def test_output_length_matches_input(self, realistic_returns: pd.Series) -> None:
        vol = rolling_volatility(realistic_returns, 30)
        assert len(vol) == len(realistic_returns)

    def test_first_window_minus_1_values_are_nan(self, realistic_returns: pd.Series) -> None:
        vol = rolling_volatility(realistic_returns, 30)
        assert vol.iloc[:29].isna().all()
        assert not math.isnan(float(vol.iloc[29]))

    def test_series_name_contains_window(self, realistic_returns: pd.Series) -> None:
        vol = rolling_volatility(realistic_returns, 90)
        assert "90" in vol.name


# -----------------------------------------------------------------------
# historical_var
# -----------------------------------------------------------------------


class TestHistoricalVar:
    def test_var_is_positive(self, realistic_returns: pd.Series) -> None:
        var = historical_var(realistic_returns, 0.95)
        assert var > 0

    def test_var99_greater_than_var95(self, realistic_returns: pd.Series) -> None:
        assert historical_var(realistic_returns, 0.99) > historical_var(realistic_returns, 0.95)

    def test_short_series_returns_nan(self) -> None:
        r = pd.Series([0.01] * 5)
        assert math.isnan(historical_var(r))

    def test_known_symmetric_distribution(self) -> None:
        # For a series of returns [−0.10, −0.05, 0.0, 0.05, 0.10] x 20 reps,
        # VaR 95% should be close to 0.10
        vals = np.tile([-0.10, -0.05, 0.0, 0.05, 0.10], 20)
        r = pd.Series(vals)
        var = historical_var(r, 0.95)
        assert var == pytest.approx(0.10, abs=0.01)


# -----------------------------------------------------------------------
# trim_to_lookback
# -----------------------------------------------------------------------


class TestTrimToLookback:
    def test_trims_to_correct_length(self, realistic_returns: pd.Series) -> None:
        trimmed = trim_to_lookback(realistic_returns, 1)
        latest = realistic_returns.index[-1]
        cutoff = latest - pd.DateOffset(years=1)
        assert trimmed.index[0] >= cutoff

    def test_returns_full_series_when_shorter_than_lookback(
        self, flat_prices: pd.Series
    ) -> None:
        trimmed = trim_to_lookback(flat_prices, 10)
        assert len(trimmed) == len(flat_prices)
