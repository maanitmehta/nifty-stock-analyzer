"""Decision card: synthesise all signals into a single at-a-glance verdict."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .snapshot import RiskSnapshot
from .technicals import TechnicalSnapshot
from .valuation import ValuationSnapshot


@dataclass
class DecisionCard:
    """Aggregated signal summary for a single stock."""

    ticker: str

    # ── Component traffic lights ────────────────────────────────────────────
    risk_signal: str        # "green" | "yellow" | "red"
    valuation_signal: str
    momentum_signal: str
    quality_signal: str

    # ── Composite score ─────────────────────────────────────────────────────
    score: int              # 0–100
    label: str              # "Avoid" | "Caution" | "Neutral" | "Watchlist" | "Consider"

    # ── Narrative bullets ───────────────────────────────────────────────────
    positives: list[str] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)


# Scoring weights (must sum to 100)
_WEIGHTS = {"risk": 30, "valuation": 25, "momentum": 25, "quality": 20}
_SIGNAL_SCORE = {"green": 100, "yellow": 50, "red": 0}
_LABEL_THRESHOLDS = [
    (75, "Consider"),
    (55, "Watchlist"),
    (40, "Neutral"),
    (20, "Caution"),
    (0,  "Avoid"),
]


def generate_decision_card(
    ticker: str,
    snapshot: RiskSnapshot,
    tech: TechnicalSnapshot,
    val: ValuationSnapshot | None = None,
) -> DecisionCard:
    """Combine risk, technical, and valuation signals into a DecisionCard.

    Signal logic uses transparent, rule-based thresholds (no ML).
    Each signal is independently derivable from the metric definitions
    documented in the Guide tab.
    """
    risk_sig, risk_pos, risk_neg, risk_risks = _risk_signals(snapshot)
    mom_sig, mom_pos, mom_neg, mom_risks = _momentum_signals(tech)
    val_sig, val_pos, val_neg, val_risks = _valuation_signals(val)
    qual_sig, qual_pos, qual_neg, qual_risks = _quality_signals(val, snapshot)

    score = (
        _SIGNAL_SCORE[risk_sig] * _WEIGHTS["risk"]
        + _SIGNAL_SCORE[val_sig] * _WEIGHTS["valuation"]
        + _SIGNAL_SCORE[mom_sig] * _WEIGHTS["momentum"]
        + _SIGNAL_SCORE[qual_sig] * _WEIGHTS["quality"]
    ) // 100

    label = next(lbl for threshold, lbl in _LABEL_THRESHOLDS if score >= threshold)

    return DecisionCard(
        ticker=ticker,
        risk_signal=risk_sig,
        valuation_signal=val_sig,
        momentum_signal=mom_sig,
        quality_signal=qual_sig,
        score=score,
        label=label,
        positives=risk_pos + mom_pos + val_pos + qual_pos,
        negatives=risk_neg + mom_neg + val_neg + qual_neg,
        key_risks=risk_risks + mom_risks + val_risks + qual_risks,
    )


# ---------------------------------------------------------------------------
# Component signal derivers
# ---------------------------------------------------------------------------


def _risk_signals(
    snap: RiskSnapshot,
) -> tuple[str, list[str], list[str], list[str]]:
    pos: list[str] = []
    neg: list[str] = []
    risks: list[str] = []
    points = 0
    checked = 0

    capm = snap.capm
    ratios = snap.ratios

    # Sharpe ratio
    if ratios.sharpe is not None:
        checked += 1
        if ratios.sharpe > 1.0:
            points += 1
            pos.append(f"Sharpe ratio {ratios.sharpe:.2f} — well-compensated for volatility")
        elif ratios.sharpe < 0:
            neg.append(f"Negative Sharpe ({ratios.sharpe:.2f}) — return below risk-free rate")
        else:
            pos.append(f"Sharpe ratio {ratios.sharpe:.2f}")

    # Max drawdown
    mdd = snap.drawdown.max_drawdown
    if not math.isnan(mdd):
        checked += 1
        if mdd > -0.25:
            points += 1
            pos.append(f"Max drawdown contained at {mdd*100:.1f}%")
        elif mdd < -0.50:
            risks.append(f"Severe historical drawdown of {mdd*100:.1f}% — high pain tolerance required")
        else:
            neg.append(f"Max drawdown of {mdd*100:.1f}%")

    # Alpha (only if statistically significant)
    if capm is not None and capm.alpha_pvalue < 0.10:
        checked += 1
        if capm.alpha_annualized > 0.03:
            points += 1
            pos.append(
                f"Statistically significant alpha of {capm.alpha_annualized*100:+.1f}% "
                f"(p={capm.alpha_pvalue:.2f})"
            )
        elif capm.alpha_annualized < -0.03:
            neg.append(
                f"Significant negative alpha of {capm.alpha_annualized*100:+.1f}% "
                f"(p={capm.alpha_pvalue:.2f})"
            )

    # Beta extremes
    if capm is not None:
        checked += 1
        beta = capm.beta
        if 0.5 <= beta <= 1.5:
            points += 1
        elif beta > 2.0:
            risks.append(f"Very high beta ({beta:.2f}) — amplifies market swings significantly")
        elif beta < 0:
            risks.append(f"Negative beta ({beta:.2f}) — moves inversely to market")

    # VaR
    if not math.isnan(snap.var_99):
        if snap.var_99 > 0.05:
            risks.append(f"High tail risk: 1% VaR = {snap.var_99*100:.1f}% daily loss")

    signal = _score_to_signal(points, checked)
    return signal, pos, neg, risks


def _momentum_signals(
    tech: TechnicalSnapshot,
) -> tuple[str, list[str], list[str], list[str]]:
    pos: list[str] = []
    neg: list[str] = []
    risks: list[str] = []
    points = 0
    checked = 0

    # RSI
    rsi = tech.rsi.current
    if not math.isnan(rsi):
        checked += 1
        if tech.rsi.is_oversold:
            pos.append(f"RSI {rsi:.0f} — oversold, potential mean-reversion opportunity")
            points += 1
        elif tech.rsi.is_overbought:
            neg.append(f"RSI {rsi:.0f} — overbought, elevated pullback risk")
        elif 40 <= rsi <= 65:
            points += 1
            pos.append(f"RSI {rsi:.0f} — healthy momentum, not extreme")

    # MA trend
    ma = tech.ma
    checked += 1
    if ma.golden_cross:
        points += 1
        pos.append("Golden Cross: SMA 50 recently crossed above SMA 200 — long-term bullish")
    elif ma.death_cross:
        neg.append("Death Cross: SMA 50 recently crossed below SMA 200 — long-term bearish")
        risks.append("Death Cross active — institutional selling pressure may continue")
    elif ma.price_above_sma_200:
        points += 1
        pos.append(f"Price is {ma.pct_from_sma_200*100:+.1f}% above the 200-day SMA — uptrend intact")
    else:
        neg.append(f"Price is {ma.pct_from_sma_200*100:+.1f}% below the 200-day SMA — downtrend")

    # MACD
    macd = tech.macd
    checked += 1
    if macd.is_bullish_crossover:
        points += 1
        pos.append("MACD bullish crossover — short-term momentum turning positive")
    elif macd.is_bearish_crossover:
        neg.append("MACD bearish crossover — short-term momentum turning negative")
    elif macd.current_histogram > 0:
        points += 1
        pos.append("MACD histogram positive — bullish momentum")
    else:
        neg.append("MACD histogram negative — bearish momentum")

    # 52W range
    r52 = tech.range_52w
    checked += 1
    pct_from_high = r52.pct_from_high
    if pct_from_high > -0.10:
        points += 1
        pos.append(f"Trading near 52-week high ({pct_from_high*100:+.1f}%) — strong price performance")
    elif pct_from_high < -0.40:
        risks.append(f"Trading {pct_from_high*100:+.1f}% below 52-week high — extended weakness")

    signal = _score_to_signal(points, checked)
    return signal, pos, neg, risks


def _valuation_signals(
    val: ValuationSnapshot | None,
) -> tuple[str, list[str], list[str], list[str]]:
    if val is None:
        return "yellow", [], [], ["Fundamental data unavailable — valuation cannot be assessed"]

    pos: list[str] = []
    neg: list[str] = []
    risks: list[str] = []

    v_sig = val.valuation_signal
    if v_sig == "Cheap":
        pos.append(f"Valuation appears inexpensive (P/E {_fmt(val.pe_trailing)}×, P/B {_fmt(val.pb)}×)")
        signal = "green"
    elif v_sig == "Expensive":
        neg.append(f"Valuation appears stretched (P/E {_fmt(val.pe_trailing)}×, P/B {_fmt(val.pb)}×)")
        risks.append("High valuation multiples limit margin of safety")
        signal = "red"
    else:
        signal = "yellow"

    if val.revenue_growth_yoy is not None and val.revenue_growth_yoy > 0.15:
        pos.append(f"Strong revenue growth: {val.revenue_growth_yoy*100:.1f}% YoY")
    if val.revenue_growth_yoy is not None and val.revenue_growth_yoy < 0:
        neg.append(f"Revenue contraction: {val.revenue_growth_yoy*100:.1f}% YoY")

    return signal, pos, neg, risks


def _quality_signals(
    val: ValuationSnapshot | None,
    snap: RiskSnapshot,
) -> tuple[str, list[str], list[str], list[str]]:
    pos: list[str] = []
    neg: list[str] = []
    risks: list[str] = []
    points = 0
    checked = 0

    if val is not None:
        if val.roe is not None:
            checked += 1
            if val.roe > 0.15:
                points += 1
                pos.append(f"Strong ROE of {val.roe*100:.1f}%")
            elif val.roe < 0.05:
                neg.append(f"Weak ROE of {val.roe*100:.1f}%")

        if val.debt_to_equity is not None:
            checked += 1
            if val.debt_to_equity < 0.5:
                points += 1
                pos.append(f"Conservative balance sheet (D/E {val.debt_to_equity:.2f})")
            elif val.debt_to_equity > 2.0:
                risks.append(f"High leverage (D/E {val.debt_to_equity:.2f}) — vulnerable in rate cycles")

        if val.interest_coverage is not None and val.interest_coverage < 2.0:
            risks.append(
                f"Thin interest coverage ({val.interest_coverage:.1f}×) — debt service risk"
            )

        if val.piotroski is not None:
            checked += 1
            p = val.piotroski
            if p.score >= 7:
                points += 1
                pos.append(f"Piotroski F-Score {p.score}/9 — financially strong")
            elif p.score <= 3:
                neg.append(f"Piotroski F-Score {p.score}/9 — financial deterioration signals")
                risks.append("Multiple Piotroski criteria failing — review balance sheet carefully")

    # Sortino as quality of returns (asymmetry)
    if snap.ratios.sortino is not None and snap.ratios.sortino > 1.5:
        checked += 1
        points += 1
        pos.append(f"Sortino ratio {snap.ratios.sortino:.2f} — strong downside-adjusted returns")

    signal = _score_to_signal(points, checked)
    return signal, pos, neg, risks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_to_signal(points: int, checked: int) -> str:
    if checked == 0:
        return "yellow"
    ratio = points / checked
    if ratio >= 0.6:
        return "green"
    if ratio >= 0.3:
        return "yellow"
    return "red"


def _fmt(val: float | None, decimals: int = 1) -> str:
    return f"{val:.{decimals}f}" if val is not None else "N/A"
