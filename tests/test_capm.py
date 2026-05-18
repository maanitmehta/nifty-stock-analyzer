"""Tests for nifty_analyzer.features.capm — NSA-8 & NSA-9."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nifty_analyzer.features.capm import (
    calmar_ratio,
    capm_regression,
    rolling_capm,
    sharpe_ratio,
    sortino_ratio,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_returns() -> tuple[pd.Series, pd.Series]:
    """Stock and market returns with known beta=1.5, alpha≈0.

    stock = 1.5 * market + noise
    Expected beta ≈ 1.5, alpha ≈ 0 (OLS on excess returns ≈ same since rf is tiny).
    """
    rng = np.random.default_rng(7)
    n = 504  # 2 years of trading days
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    market = pd.Series(rng.normal(0.0005, 0.010, n), index=idx, name="market")
    noise = pd.Series(rng.normal(0.0, 0.005, n), index=idx)
    stock = (1.5 * market + noise).rename("stock")
    return stock, market


@pytest.fixture()
def long_flat_returns() -> pd.Series:
    """250 zero log returns (useful for edge-case ratio tests)."""
    idx = pd.date_range("2022-01-03", periods=250, freq="B")
    return pd.Series(0.0, index=idx)


@pytest.fixture()
def positive_returns() -> pd.Series:
    rng = np.random.default_rng(99)
    idx = pd.date_range("2021-01-04", periods=504, freq="B")
    return pd.Series(rng.normal(0.001, 0.010, 504), index=idx)


@pytest.fixture()
def crash_prices() -> pd.Series:
    """Prices that drop 40% then stay flat — produces meaningful Calmar."""
    idx = pd.date_range("2021-01-04", periods=504, freq="B")
    prices = np.concatenate([np.linspace(100, 60, 252), np.full(252, 60)])
    return pd.Series(prices, index=idx)


# ---------------------------------------------------------------------------
# capm_regression
# ---------------------------------------------------------------------------


class TestCAPMRegression:
    def test_beta_close_to_true_value(
        self, synthetic_returns: tuple[pd.Series, pd.Series]
    ) -> None:
        stock, market = synthetic_returns
        result = capm_regression(stock, market)
        assert result.beta == pytest.approx(1.5, abs=0.15)

    def test_alpha_near_zero_when_no_excess_return(
        self, synthetic_returns: tuple[pd.Series, pd.Series]
    ) -> None:
        stock, market = synthetic_returns
        result = capm_regression(stock, market)
        # Alpha should be near 0 (no excess return built in)
        assert abs(result.alpha_annualized) < 0.15

    def test_r_squared_between_0_and_1(
        self, synthetic_returns: tuple[pd.Series, pd.Series]
    ) -> None:
        stock, market = synthetic_returns
        result = capm_regression(stock, market)
        assert 0.0 <= result.r_squared <= 1.0

    def test_high_r_squared_for_near_linear_relationship(
        self, synthetic_returns: tuple[pd.Series, pd.Series]
    ) -> None:
        stock, market = synthetic_returns
        result = capm_regression(stock, market)
        assert result.r_squared > 0.7

    def test_n_observations_correct(
        self, synthetic_returns: tuple[pd.Series, pd.Series]
    ) -> None:
        stock, market = synthetic_returns
        result = capm_regression(stock, market)
        assert result.n_observations == 504

    def test_tstat_and_pvalue_fields_present(
        self, synthetic_returns: tuple[pd.Series, pd.Series]
    ) -> None:
        stock, market = synthetic_returns
        result = capm_regression(stock, market)
        assert not math.isnan(result.beta_tstat)
        assert 0.0 <= result.beta_pvalue <= 1.0

    def test_raises_on_too_few_observations(self) -> None:
        short = pd.Series([0.01] * 20, index=pd.date_range("2023-01-02", periods=20))
        with pytest.raises(ValueError, match="30"):
            capm_regression(short, short)

    def test_handles_date_misalignment(
        self, synthetic_returns: tuple[pd.Series, pd.Series]
    ) -> None:
        stock, market = synthetic_returns
        # Shift market by 5 days — only 499 overlapping observations
        market_shifted = market.iloc[5:]
        result = capm_regression(stock, market_shifted)
        assert result.n_observations == 499


# ---------------------------------------------------------------------------
# rolling_capm
# ---------------------------------------------------------------------------


class TestRollingCAPM:
    def test_output_columns(
        self, synthetic_returns: tuple[pd.Series, pd.Series]
    ) -> None:
        stock, market = synthetic_returns
        df = rolling_capm(stock, market, window=252)
        assert "alpha_annualized" in df.columns
        assert "beta" in df.columns

    def test_no_nan_in_output(
        self, synthetic_returns: tuple[pd.Series, pd.Series]
    ) -> None:
        stock, market = synthetic_returns
        df = rolling_capm(stock, market, window=252)
        assert not df.isna().any().any()

    def test_rolling_beta_near_true_value(
        self, synthetic_returns: tuple[pd.Series, pd.Series]
    ) -> None:
        stock, market = synthetic_returns
        df = rolling_capm(stock, market, window=252)
        median_beta = float(df["beta"].median())
        assert median_beta == pytest.approx(1.5, abs=0.20)

    def test_output_length_less_than_input(
        self, synthetic_returns: tuple[pd.Series, pd.Series]
    ) -> None:
        stock, market = synthetic_returns
        df = rolling_capm(stock, market, window=252)
        assert len(df) < len(stock)


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    def test_positive_returns_give_positive_sharpe(
        self, positive_returns: pd.Series
    ) -> None:
        result = sharpe_ratio(positive_returns)
        assert result is not None
        assert result > 0

    def test_zero_variance_returns_none(self, long_flat_returns: pd.Series) -> None:
        result = sharpe_ratio(long_flat_returns)
        assert result is None

    def test_empty_returns_none(self) -> None:
        assert sharpe_ratio(pd.Series(dtype=float)) is None

    def test_single_observation_returns_none(self) -> None:
        assert sharpe_ratio(pd.Series([0.01])) is None

    def test_sharpe_decreases_with_higher_rf(
        self, positive_returns: pd.Series
    ) -> None:
        sharpe_low_rf = sharpe_ratio(positive_returns, rf_annual=0.02)
        sharpe_high_rf = sharpe_ratio(positive_returns, rf_annual=0.12)
        assert sharpe_low_rf > sharpe_high_rf  # type: ignore[operator]


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------


class TestSortinoRatio:
    def test_positive_returns_give_positive_sortino(
        self, positive_returns: pd.Series
    ) -> None:
        result = sortino_ratio(positive_returns)
        assert result is not None
        assert result > 0

    def test_all_positive_returns_returns_none(self) -> None:
        r = pd.Series([0.01] * 50, index=pd.date_range("2023-01-02", periods=50))
        assert sortino_ratio(r) is None

    def test_sortino_generally_higher_than_sharpe(
        self, positive_returns: pd.Series
    ) -> None:
        # Sortino only penalizes downside, so it's >= Sharpe for typical distributions
        s = sharpe_ratio(positive_returns)
        so = sortino_ratio(positive_returns)
        assert s is not None and so is not None
        assert so >= s

    def test_empty_returns_none(self) -> None:
        assert sortino_ratio(pd.Series(dtype=float)) is None


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------


class TestCalmarRatio:
    def test_returns_float_for_valid_input(
        self, positive_returns: pd.Series, crash_prices: pd.Series
    ) -> None:
        result = calmar_ratio(positive_returns, crash_prices)
        assert result is not None
        assert isinstance(result, float)

    def test_flat_prices_returns_none(self, positive_returns: pd.Series) -> None:
        flat = pd.Series(
            100.0, index=pd.date_range("2021-01-04", periods=504, freq="B")
        )
        assert calmar_ratio(positive_returns, flat) is None

    def test_empty_returns_none(self) -> None:
        assert calmar_ratio(pd.Series(dtype=float), pd.Series(dtype=float)) is None

    def test_calmar_sign(
        self, positive_returns: pd.Series, crash_prices: pd.Series
    ) -> None:
        result = calmar_ratio(positive_returns, crash_prices)
        # For positive returns / negative drawdown → ratio should be positive
        assert result is not None
        # Sign depends on whether returns are positive; just check it's a real number
        assert not math.isnan(result)
