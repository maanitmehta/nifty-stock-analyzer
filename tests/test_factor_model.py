"""Tests for nifty_analyzer.features.factor_model — NSA-10, NSA-23."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_analyzer.features.factor_model import (
    FactorData,
    ff3_regression,
    interpret_hml,
    interpret_smb,
    rolling_ff3,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_factors() -> FactorData:
    """504 days of synthetic factor returns with known properties."""
    rng = np.random.default_rng(11)
    n = 504
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    mkt = pd.Series(rng.normal(0.0005, 0.010, n), index=idx, name="Mkt-Rf")
    smb = pd.Series(rng.normal(0.0002, 0.006, n), index=idx, name="SMB")
    return FactorData(mkt_rf=mkt, smb=smb)


@pytest.fixture()
def stock_with_known_loadings(synthetic_factors: FactorData) -> pd.Series:
    """Stock returns built from beta=1.2, smb=0.5, alpha≈0."""
    rng = np.random.default_rng(22)
    factors = synthetic_factors.factor_df
    noise = pd.Series(rng.normal(0, 0.006, len(factors)), index=factors.index)
    ret = 1.2 * factors["Mkt-Rf"] + 0.5 * factors["SMB"] + noise
    return ret.rename("stock")


# ---------------------------------------------------------------------------
# FactorData
# ---------------------------------------------------------------------------


class TestFactorData:
    def test_factor_df_contains_mkt_rf(self, synthetic_factors: FactorData) -> None:
        df = synthetic_factors.factor_df
        assert "Mkt-Rf" in df.columns

    def test_factor_df_contains_smb_when_set(self, synthetic_factors: FactorData) -> None:
        df = synthetic_factors.factor_df
        assert "SMB" in df.columns

    def test_factors_used_matches_columns(self, synthetic_factors: FactorData) -> None:
        assert synthetic_factors.factors_used == list(synthetic_factors.factor_df.columns)

    def test_factor_df_no_nan(self, synthetic_factors: FactorData) -> None:
        assert not synthetic_factors.factor_df.isna().any().any()

    def test_hml_none_by_default(self, synthetic_factors: FactorData) -> None:
        assert synthetic_factors.hml is None

    def test_market_only_factor_data(self) -> None:
        idx = pd.date_range("2022-01-03", periods=100, freq="B")
        mkt = pd.Series(0.001, index=idx)
        fd = FactorData(mkt_rf=mkt)
        assert fd.factors_used == ["Mkt-Rf"]
        assert "SMB" not in fd.factor_df.columns


# ---------------------------------------------------------------------------
# ff3_regression
# ---------------------------------------------------------------------------


class TestFF3Regression:
    def test_mkt_beta_close_to_true(
        self,
        stock_with_known_loadings: pd.Series,
        synthetic_factors: FactorData,
    ) -> None:
        result = ff3_regression(stock_with_known_loadings, synthetic_factors)
        assert result.mkt_beta == pytest.approx(1.2, abs=0.15)

    def test_smb_loading_close_to_true(
        self,
        stock_with_known_loadings: pd.Series,
        synthetic_factors: FactorData,
    ) -> None:
        result = ff3_regression(stock_with_known_loadings, synthetic_factors)
        assert result.smb_loading is not None
        assert result.smb_loading == pytest.approx(0.5, abs=0.15)

    def test_r_squared_high_for_constructed_returns(
        self,
        stock_with_known_loadings: pd.Series,
        synthetic_factors: FactorData,
    ) -> None:
        result = ff3_regression(stock_with_known_loadings, synthetic_factors)
        assert result.r_squared > 0.7

    def test_factors_used_list_matches_input(
        self,
        stock_with_known_loadings: pd.Series,
        synthetic_factors: FactorData,
    ) -> None:
        result = ff3_regression(stock_with_known_loadings, synthetic_factors)
        assert "Mkt-Rf" in result.factors_used
        assert "SMB" in result.factors_used

    def test_hml_is_none_when_not_in_factors(
        self,
        stock_with_known_loadings: pd.Series,
        synthetic_factors: FactorData,
    ) -> None:
        result = ff3_regression(stock_with_known_loadings, synthetic_factors)
        assert result.hml_loading is None
        assert result.hml_tstat is None

    def test_alpha_near_zero_for_constructed_returns(
        self,
        stock_with_known_loadings: pd.Series,
        synthetic_factors: FactorData,
    ) -> None:
        result = ff3_regression(stock_with_known_loadings, synthetic_factors)
        assert abs(result.alpha_annualized) < 0.20

    def test_raises_on_too_few_observations(
        self, synthetic_factors: FactorData
    ) -> None:
        short = pd.Series([0.01] * 20, index=pd.date_range("2023-01-02", periods=20))
        with pytest.raises(ValueError, match="30"):
            ff3_regression(short, synthetic_factors)

    def test_n_observations_correct(
        self,
        stock_with_known_loadings: pd.Series,
        synthetic_factors: FactorData,
    ) -> None:
        result = ff3_regression(stock_with_known_loadings, synthetic_factors)
        assert result.n_observations == 504

    def test_pvalues_between_0_and_1(
        self,
        stock_with_known_loadings: pd.Series,
        synthetic_factors: FactorData,
    ) -> None:
        result = ff3_regression(stock_with_known_loadings, synthetic_factors)
        assert 0 <= result.alpha_pvalue <= 1
        assert 0 <= result.mkt_pvalue <= 1
        assert result.smb_pvalue is not None
        assert 0 <= result.smb_pvalue <= 1


# ---------------------------------------------------------------------------
# rolling_ff3
# ---------------------------------------------------------------------------


class TestRollingFF3:
    def test_output_has_alpha_and_beta_columns(
        self,
        stock_with_known_loadings: pd.Series,
        synthetic_factors: FactorData,
    ) -> None:
        df = rolling_ff3(stock_with_known_loadings, synthetic_factors, window=252)
        assert "alpha_annualized" in df.columns
        assert "mkt_beta" in df.columns

    def test_smb_column_present_when_factor_available(
        self,
        stock_with_known_loadings: pd.Series,
        synthetic_factors: FactorData,
    ) -> None:
        df = rolling_ff3(stock_with_known_loadings, synthetic_factors, window=252)
        assert "smb_loading" in df.columns

    def test_no_nan_in_output(
        self,
        stock_with_known_loadings: pd.Series,
        synthetic_factors: FactorData,
    ) -> None:
        df = rolling_ff3(stock_with_known_loadings, synthetic_factors, window=252)
        assert not df.isna().any().any()

    def test_rolling_mkt_beta_near_true(
        self,
        stock_with_known_loadings: pd.Series,
        synthetic_factors: FactorData,
    ) -> None:
        df = rolling_ff3(stock_with_known_loadings, synthetic_factors, window=252)
        assert float(df["mkt_beta"].median()) == pytest.approx(1.2, abs=0.20)

    def test_insufficient_data_returns_empty(
        self, synthetic_factors: FactorData
    ) -> None:
        short = pd.Series([0.001] * 50, index=pd.date_range("2023-01-02", periods=50))
        df = rolling_ff3(short, synthetic_factors, window=252)
        assert df.empty


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------


class TestInterpretation:
    def test_smb_positive_large(self) -> None:
        assert "small" in interpret_smb(0.5).lower() or "midcap" in interpret_smb(0.5).lower()

    def test_smb_negative_large(self) -> None:
        assert "large" in interpret_smb(-0.5).lower()

    def test_smb_neutral(self) -> None:
        assert "neutral" in interpret_smb(0.05).lower()

    def test_smb_none(self) -> None:
        assert interpret_smb(None) == "N/A"

    def test_hml_none(self) -> None:
        assert interpret_hml(None) == "N/A"

    def test_hml_positive(self) -> None:
        assert "value" in interpret_hml(0.4).lower()

    def test_hml_negative(self) -> None:
        assert "growth" in interpret_hml(-0.4).lower()
