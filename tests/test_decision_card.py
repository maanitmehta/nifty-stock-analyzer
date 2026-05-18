"""Tests for nifty_analyzer.features.decision_card — NSA-24."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nifty_analyzer.features.capm import CAPMResult, RatioMetrics
from nifty_analyzer.features.decision_card import (
    DecisionCard,
    _risk_signals,
    _score_to_signal,
    _valuation_signals,
    generate_decision_card,
)
from nifty_analyzer.features.snapshot import RiskSnapshot
from nifty_analyzer.features.technicals import compute_technicals

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_snapshot(**kwargs) -> RiskSnapshot:
    """Build a minimal RiskSnapshot with sensible defaults."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-03", periods=504, freq="B")
    prices = pd.Series(1000 * np.exp(np.cumsum(rng.normal(0.0005, 0.010, 504))), index=idx)
    returns = np.log(prices / prices.shift(1)).dropna()

    capm = CAPMResult(
        alpha_annualized=0.05, alpha_daily=0.0002, alpha_tstat=2.1, alpha_pvalue=0.04,
        beta=1.1, beta_tstat=15.0, beta_pvalue=0.0001,
        r_squared=0.72, adj_r_squared=0.71, n_observations=504,
    )
    ratios = RatioMetrics(sharpe=1.2, sortino=1.8, calmar=0.6)
    from nifty_analyzer.features.risk import compute_drawdown
    dd = compute_drawdown(prices)

    defaults = dict(
        ticker="TEST.NS",
        lookback_years=3,
        annualized_return=0.12,
        annualized_std=0.18,
        rolling_vol_30d=pd.Series(dtype=float),
        rolling_vol_90d=pd.Series(dtype=float),
        var_95=0.015,
        var_99=0.025,
        drawdown=dd,
        capm=capm,
        rolling_capm_df=pd.DataFrame(),
        ratios=ratios,
        returns=returns,
        prices=prices,
    )
    defaults.update(kwargs)
    return RiskSnapshot(**defaults)


def _make_tech() -> object:
    """Build a TechnicalSnapshot from synthetic OHLCV."""
    rng = np.random.default_rng(42)
    n = 300
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    log_ret = rng.normal(0.0005, 0.012, n)
    close = pd.Series(1000 * np.exp(np.cumsum(log_ret)), index=idx)
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    vol = pd.Series(rng.integers(500_000, 5_000_000, n).astype(float), index=idx)
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol})
    return compute_technicals(df)


# ---------------------------------------------------------------------------
# generate_decision_card — structural
# ---------------------------------------------------------------------------


class TestGenerateDecisionCard:
    def test_returns_decision_card(self) -> None:
        snap = _make_snapshot()
        tech = _make_tech()
        card = generate_decision_card("TEST.NS", snap, tech, None)
        assert isinstance(card, DecisionCard)

    def test_ticker_field_set(self) -> None:
        snap = _make_snapshot()
        tech = _make_tech()
        card = generate_decision_card("TEST.NS", snap, tech)
        assert card.ticker == "TEST.NS"

    def test_score_between_0_and_100(self) -> None:
        snap = _make_snapshot()
        tech = _make_tech()
        card = generate_decision_card("TEST.NS", snap, tech)
        assert 0 <= card.score <= 100

    def test_label_is_valid(self) -> None:
        snap = _make_snapshot()
        tech = _make_tech()
        card = generate_decision_card("TEST.NS", snap, tech)
        assert card.label in {"Avoid", "Caution", "Neutral", "Watchlist", "Consider"}

    def test_signals_are_valid_colors(self) -> None:
        snap = _make_snapshot()
        tech = _make_tech()
        card = generate_decision_card("TEST.NS", snap, tech)
        for sig in [card.risk_signal, card.valuation_signal,
                    card.momentum_signal, card.quality_signal]:
            assert sig in {"green", "yellow", "red"}

    def test_positives_and_negatives_are_lists(self) -> None:
        snap = _make_snapshot()
        tech = _make_tech()
        card = generate_decision_card("TEST.NS", snap, tech)
        assert isinstance(card.positives, list)
        assert isinstance(card.negatives, list)
        assert isinstance(card.key_risks, list)

    def test_high_sharpe_contributes_green(self) -> None:
        snap = _make_snapshot()
        tech = _make_tech()
        card = generate_decision_card("TEST.NS", snap, tech)
        # With Sharpe=1.2, risk signal should be at least yellow
        assert card.risk_signal in {"green", "yellow"}

    def test_no_valuation_gives_yellow_valuation(self) -> None:
        snap = _make_snapshot()
        tech = _make_tech()
        card = generate_decision_card("TEST.NS", snap, tech, val=None)
        assert card.valuation_signal == "yellow"


# ---------------------------------------------------------------------------
# _score_to_signal
# ---------------------------------------------------------------------------


class TestScoreToSignal:
    def test_all_points_is_green(self) -> None:
        assert _score_to_signal(4, 4) == "green"

    def test_no_points_is_red(self) -> None:
        assert _score_to_signal(0, 4) == "red"

    def test_half_points_is_yellow(self) -> None:
        assert _score_to_signal(2, 4) == "yellow"

    def test_zero_checked_is_yellow(self) -> None:
        assert _score_to_signal(0, 0) == "yellow"


# ---------------------------------------------------------------------------
# _risk_signals
# ---------------------------------------------------------------------------


class TestRiskSignals:
    def test_good_metrics_give_green(self) -> None:
        snap = _make_snapshot()  # Sharpe=1.2, shallow drawdown, positive alpha
        sig, pos, neg, risks = _risk_signals(snap)
        assert sig in {"green", "yellow"}
        assert len(pos) > 0

    def test_negative_sharpe_is_in_negatives(self) -> None:
        from nifty_analyzer.features.capm import RatioMetrics
        snap = _make_snapshot(ratios=RatioMetrics(sharpe=-0.3, sortino=-0.2, calmar=None))
        sig, pos, neg, risks = _risk_signals(snap)
        assert any("Negative Sharpe" in n for n in neg)


# ---------------------------------------------------------------------------
# _valuation_signals
# ---------------------------------------------------------------------------


class TestValuationSignals:
    def test_none_val_returns_yellow(self) -> None:
        sig, _, _, _ = _valuation_signals(None)
        assert sig == "yellow"
