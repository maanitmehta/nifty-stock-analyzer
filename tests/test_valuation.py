"""Tests for nifty_analyzer.features.valuation — NSA-15, NSA-16, NSA-17."""

from __future__ import annotations

import pandas as pd
import pytest

from nifty_analyzer.data.fundamentals import FundamentalData
from nifty_analyzer.features.valuation import (
    PiotroskiResult,
    ValuationSnapshot,
    _get,
    _piotroski,
    _quality_signal,
    _valuation_signal,
    compute_valuation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fund(**kwargs) -> FundamentalData:
    """Build a FundamentalData with sensible defaults, overridden by kwargs."""
    defaults = dict(
        ticker="TEST.NS",
        pe_trailing=20.0,
        pe_forward=18.0,
        pb=3.0,
        ev_ebitda=12.0,
        ps=2.5,
        market_cap=1e12,
        enterprise_value=1.1e12,
        roe=0.18,
        roa=0.08,
        debt_to_equity=0.4,
        current_ratio=2.0,
        gross_margin=0.45,
        operating_margin=0.20,
        net_margin=0.15,
        interest_coverage=8.0,
        revenue_growth_yoy=0.12,
        earnings_growth_yoy=0.15,
    )
    defaults.update(kwargs)
    return FundamentalData(**defaults)


def _make_statements() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Minimal financial statements that produce a clean Piotroski score."""
    dates = pd.to_datetime(["2024-03-31", "2023-03-31"])

    inc = pd.DataFrame(
        {
            dates[0]: {
                "Total Revenue": 100_000,
                "Gross Profit": 45_000,
                "Net Income": 15_000,
            },
            dates[1]: {
                "Total Revenue": 90_000,
                "Gross Profit": 38_000,
                "Net Income": 12_000,
            },
        }
    )

    bs = pd.DataFrame(
        {
            dates[0]: {
                "Total Assets": 200_000,
                "Long Term Debt": 30_000,
                "Current Assets": 80_000,
                "Current Liabilities": 35_000,
                "Ordinary Shares Number": 1_000_000,
            },
            dates[1]: {
                "Total Assets": 180_000,
                "Long Term Debt": 40_000,
                "Current Assets": 70_000,
                "Current Liabilities": 35_000,
                "Ordinary Shares Number": 1_000_000,
            },
        }
    )

    cf = pd.DataFrame(
        {
            dates[0]: {"Operating Cash Flow": 18_000},
            dates[1]: {"Operating Cash Flow": 14_000},
        }
    )

    return inc, bs, cf


# ---------------------------------------------------------------------------
# compute_valuation — structural
# ---------------------------------------------------------------------------


class TestComputeValuation:
    def test_returns_valuation_snapshot(self) -> None:
        fund = _make_fund()
        result = compute_valuation(fund)
        assert isinstance(result, ValuationSnapshot)

    def test_multiples_passed_through(self) -> None:
        fund = _make_fund(pe_trailing=22.5, pb=3.1)
        result = compute_valuation(fund)
        assert result.pe_trailing == pytest.approx(22.5)
        assert result.pb == pytest.approx(3.1)

    def test_none_fields_stay_none(self) -> None:
        fund = _make_fund(pe_trailing=None, ev_ebitda=None)
        result = compute_valuation(fund)
        assert result.pe_trailing is None
        assert result.ev_ebitda is None

    def test_signals_are_strings(self) -> None:
        result = compute_valuation(_make_fund())
        assert isinstance(result.valuation_signal, str)
        assert isinstance(result.quality_signal, str)

    def test_no_statements_piotroski_is_none(self) -> None:
        fund = _make_fund()  # no statements attached
        result = compute_valuation(fund)
        assert result.piotroski is None


# ---------------------------------------------------------------------------
# Piotroski F-score
# ---------------------------------------------------------------------------


class TestPiotroski:
    def test_computes_from_valid_statements(self) -> None:
        fund = _make_fund()
        inc, bs, cf = _make_statements()
        fund.financials = inc
        fund.balance_sheet = bs
        fund.cashflow = cf
        result = _piotroski(fund)
        assert result is not None
        assert isinstance(result, PiotroskiResult)

    def test_score_between_0_and_9(self) -> None:
        fund = _make_fund()
        inc, bs, cf = _make_statements()
        fund.financials = inc
        fund.balance_sheet = bs
        fund.cashflow = cf
        result = _piotroski(fund)
        assert result is not None
        assert 0 <= result.score <= 9

    def test_healthy_firm_scores_high(self) -> None:
        """Growing revenue, improving margins, positive OCF → should score ≥5."""
        fund = _make_fund()
        inc, bs, cf = _make_statements()
        fund.financials = inc
        fund.balance_sheet = bs
        fund.cashflow = cf
        result = _piotroski(fund)
        assert result is not None
        assert result.score >= 5

    def test_label_strong_when_score_gte_7(self) -> None:
        fund = _make_fund()
        inc, bs, cf = _make_statements()
        fund.financials = inc
        fund.balance_sheet = bs
        fund.cashflow = cf
        result = _piotroski(fund)
        if result and result.score >= 7:
            assert result.label == "Strong"

    def test_label_weak_when_score_lte_3(self) -> None:
        # Build statements where everything is deteriorating
        dates = pd.to_datetime(["2024-03-31", "2023-03-31"])
        inc = pd.DataFrame({
            dates[0]: {"Total Revenue": 80_000, "Gross Profit": 20_000, "Net Income": -5_000},
            dates[1]: {"Total Revenue": 90_000, "Gross Profit": 38_000, "Net Income": 12_000},
        })
        bs = pd.DataFrame({
            dates[0]: {"Total Assets": 200_000, "Long Term Debt": 80_000,
                       "Current Assets": 30_000, "Current Liabilities": 50_000,
                       "Ordinary Shares Number": 1_200_000},
            dates[1]: {"Total Assets": 180_000, "Long Term Debt": 40_000,
                       "Current Assets": 70_000, "Current Liabilities": 35_000,
                       "Ordinary Shares Number": 1_000_000},
        })
        cf = pd.DataFrame({dates[0]: {"Operating Cash Flow": -2_000},
                           dates[1]: {"Operating Cash Flow": 14_000}})
        fund = _make_fund()
        fund.financials = inc
        fund.balance_sheet = bs
        fund.cashflow = cf
        result = _piotroski(fund)
        assert result is not None
        assert result.score <= 4
        if result.score <= 3:
            assert result.label == "Weak"

    def test_empty_statements_returns_none(self) -> None:
        fund = _make_fund()
        result = _piotroski(fund)
        assert result is None

    def test_criteria_available_counts_non_none(self) -> None:
        fund = _make_fund()
        inc, bs, cf = _make_statements()
        fund.financials = inc
        fund.balance_sheet = bs
        fund.cashflow = cf
        result = _piotroski(fund)
        assert result is not None
        assert result.criteria_available >= 4


# ---------------------------------------------------------------------------
# Valuation signal
# ---------------------------------------------------------------------------


class TestValuationSignal:
    def test_cheap_when_low_pe_and_pb(self) -> None:
        fund = _make_fund(pe_trailing=10.0, pb=1.2)
        assert _valuation_signal(fund) == "Cheap"

    def test_expensive_when_high_pe(self) -> None:
        fund = _make_fund(pe_trailing=50.0, pb=8.0)
        assert _valuation_signal(fund) == "Expensive"

    def test_fair_for_mixed_signals(self) -> None:
        fund = _make_fund(pe_trailing=10.0, pb=7.0)
        assert _valuation_signal(fund) == "Fair"

    def test_na_when_no_multiples(self) -> None:
        fund = _make_fund(pe_trailing=None, pb=None, ps=None, ev_ebitda=None)
        assert _valuation_signal(fund) == "N/A"


# ---------------------------------------------------------------------------
# Quality signal
# ---------------------------------------------------------------------------


class TestQualitySignal:
    def test_high_roe_low_debt_gives_high(self) -> None:
        fund = _make_fund(roe=0.25, debt_to_equity=0.3)
        assert _quality_signal(fund, None) == "High"

    def test_low_roe_high_debt_gives_low(self) -> None:
        fund = _make_fund(roe=0.02, debt_to_equity=3.0)
        assert _quality_signal(fund, None) == "Low"

    def test_piotroski_strong_overrides(self) -> None:
        fund = _make_fund(roe=0.02, debt_to_equity=3.0)
        piot = PiotroskiResult(score=8, criteria=None, criteria_available=9, label="Strong")  # type: ignore[arg-type]
        assert _quality_signal(fund, piot) == "High"

    def test_piotroski_weak_overrides(self) -> None:
        fund = _make_fund(roe=0.25, debt_to_equity=0.1)
        piot = PiotroskiResult(score=2, criteria=None, criteria_available=9, label="Weak")  # type: ignore[arg-type]
        assert _quality_signal(fund, piot) == "Low"


# ---------------------------------------------------------------------------
# _get helper
# ---------------------------------------------------------------------------


class TestGetHelper:
    def test_returns_correct_value(self) -> None:
        idx = pd.to_datetime(["2024-03-31", "2023-03-31"])
        df = pd.DataFrame({"Net Income": [15_000, 12_000]}, index=idx).T
        assert _get(df, "Net Income", col=0) == pytest.approx(15_000)

    def test_returns_none_for_missing_key(self) -> None:
        idx = pd.to_datetime(["2024-03-31"])
        df = pd.DataFrame({"Revenue": [100_000]}, index=idx).T
        assert _get(df, "NonExistentRow", col=0) is None

    def test_returns_none_for_empty_df(self) -> None:
        assert _get(pd.DataFrame(), "Net Income", col=0) is None

    def test_tries_fallback_keys(self) -> None:
        idx = pd.to_datetime(["2024-03-31"])
        df = pd.DataFrame({"Total Cash From Operating Activities": [5_000]}, index=idx).T
        result = _get(df, "Operating Cash Flow", "Total Cash From Operating Activities", col=0)
        assert result == pytest.approx(5_000)
