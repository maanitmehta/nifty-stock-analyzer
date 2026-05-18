"""Valuation metrics and Piotroski F-score computation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ..data.fundamentals import FundamentalData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PiotroskiCriteria:
    """Individual pass/fail for each of the 9 Piotroski criteria."""

    # Profitability
    roa_positive: bool | None = None         # F1: ROA > 0
    ocf_positive: bool | None = None         # F2: Operating Cash Flow > 0
    roa_improving: bool | None = None        # F3: ROA higher than prior year
    accruals_low: bool | None = None         # F4: OCF/Assets > ROA (cash quality)

    # Leverage / Liquidity
    leverage_falling: bool | None = None     # F5: LT Debt/Assets lower than prior year
    liquidity_improving: bool | None = None  # F6: Current ratio higher than prior year
    no_dilution: bool | None = None          # F7: Shares outstanding not increased

    # Operating efficiency
    gross_margin_improving: bool | None = None   # F8: Gross margin higher than prior year
    asset_turnover_improving: bool | None = None # F9: Asset turnover higher than prior year


@dataclass
class PiotroskiResult:
    score: int                              # 0–9
    criteria: PiotroskiCriteria
    criteria_available: int                 # how many of the 9 could be computed
    label: str                              # "Weak", "Neutral", "Strong"


@dataclass
class ValuationSnapshot:
    # Multiples
    pe_trailing: float | None
    pe_forward: float | None
    pb: float | None
    ev_ebitda: float | None
    ps: float | None

    # Quality
    roe: float | None
    roa: float | None
    debt_to_equity: float | None
    current_ratio: float | None
    gross_margin: float | None
    operating_margin: float | None
    net_margin: float | None
    interest_coverage: float | None

    # Growth
    revenue_growth_yoy: float | None
    earnings_growth_yoy: float | None

    # Piotroski
    piotroski: PiotroskiResult | None

    # Signal summary
    valuation_signal: str    # "Cheap", "Fair", "Expensive", "N/A"
    quality_signal: str      # "High", "Medium", "Low", "N/A"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_valuation(fund: FundamentalData) -> ValuationSnapshot:
    """Derive all valuation metrics and Piotroski score from FundamentalData.

    Fields that cannot be computed (missing yfinance data) are set to None.
    The function never raises — it degrades gracefully.
    """
    piotroski = _piotroski(fund)

    return ValuationSnapshot(
        pe_trailing=fund.pe_trailing,
        pe_forward=fund.pe_forward,
        pb=fund.pb,
        ev_ebitda=fund.ev_ebitda,
        ps=fund.ps,
        roe=fund.roe,
        roa=fund.roa,
        debt_to_equity=fund.debt_to_equity,
        current_ratio=fund.current_ratio,
        gross_margin=fund.gross_margin,
        operating_margin=fund.operating_margin,
        net_margin=fund.net_margin,
        interest_coverage=fund.interest_coverage,
        revenue_growth_yoy=fund.revenue_growth_yoy,
        earnings_growth_yoy=fund.earnings_growth_yoy,
        piotroski=piotroski,
        valuation_signal=_valuation_signal(fund),
        quality_signal=_quality_signal(fund, piotroski),
    )


# ---------------------------------------------------------------------------
# Piotroski F-score
# ---------------------------------------------------------------------------


def _piotroski(fund: FundamentalData) -> PiotroskiResult | None:
    """Compute the Piotroski F-score from financial statement DataFrames.

    Returns None if fewer than 4 of the 9 criteria can be computed.
    """
    bs = fund.balance_sheet
    inc = fund.financials
    cf = fund.cashflow

    if bs.empty and inc.empty and cf.empty:
        return None

    c = PiotroskiCriteria()

    try:
        # We need at least two years of data (columns sorted newest-first by yfinance)
        ta_now = _get(bs, "Total Assets", col=0)
        ta_prev = _get(bs, "Total Assets", col=1)

        ni_now = _get(inc, "Net Income", col=0)
        ni_prev = _get(inc, "Net Income", col=1)

        ocf_now = _get(cf, "Operating Cash Flow", "Total Cash From Operating Activities", col=0)
        rev_now = _get(inc, "Total Revenue", col=0)
        rev_prev = _get(inc, "Total Revenue", col=1)
        gp_now = _get(inc, "Gross Profit", col=0)
        gp_prev = _get(inc, "Gross Profit", col=1)

        ltd_now = _get(bs, "Long Term Debt", col=0)
        ltd_prev = _get(bs, "Long Term Debt", col=1)

        cur_assets_now = _get(bs, "Current Assets", "Total Current Assets", col=0)
        cur_assets_prev = _get(bs, "Current Assets", "Total Current Assets", col=1)
        cur_liab_now = _get(bs, "Current Liabilities", "Total Current Liabilities", col=0)
        cur_liab_prev = _get(bs, "Current Liabilities", "Total Current Liabilities", col=1)

        shares_now = _get(bs, "Ordinary Shares Number", "Common Stock", col=0)
        shares_prev = _get(bs, "Ordinary Shares Number", "Common Stock", col=1)

        # ── F1: ROA > 0 ────────────────────────────────────────────────
        if ni_now is not None and ta_now and ta_now != 0:
            roa_now = ni_now / ta_now
            c.roa_positive = roa_now > 0

        # ── F2: OCF > 0 ────────────────────────────────────────────────
        if ocf_now is not None:
            c.ocf_positive = ocf_now > 0

        # ── F3: ROA improving ──────────────────────────────────────────
        if (ni_now is not None and ta_now and ta_now != 0
                and ni_prev is not None and ta_prev and ta_prev != 0):
            roa_now_v = ni_now / ta_now
            roa_prev_v = ni_prev / ta_prev
            c.roa_improving = roa_now_v > roa_prev_v

        # ── F4: Accruals (OCF/TA > ROA) ────────────────────────────────
        if ocf_now is not None and ta_now and ta_now != 0 and ni_now is not None:
            roa_now_v2 = ni_now / ta_now
            c.accruals_low = (ocf_now / ta_now) > roa_now_v2

        # ── F5: Leverage falling ────────────────────────────────────────
        if (ltd_now is not None and ta_now and ta_now != 0
                and ltd_prev is not None and ta_prev and ta_prev != 0):
            lev_now = ltd_now / ta_now
            lev_prev = ltd_prev / ta_prev
            c.leverage_falling = lev_now < lev_prev

        # ── F6: Liquidity improving ─────────────────────────────────────
        if (cur_assets_now is not None and cur_liab_now and cur_liab_now != 0
                and cur_assets_prev is not None and cur_liab_prev and cur_liab_prev != 0):
            cr_now = cur_assets_now / cur_liab_now
            cr_prev = cur_assets_prev / cur_liab_prev
            c.liquidity_improving = cr_now > cr_prev

        # ── F7: No dilution ─────────────────────────────────────────────
        if shares_now is not None and shares_prev is not None:
            c.no_dilution = shares_now <= shares_prev * 1.001  # allow tiny rounding

        # ── F8: Gross margin improving ──────────────────────────────────
        if (gp_now is not None and rev_now and rev_now != 0
                and gp_prev is not None and rev_prev and rev_prev != 0):
            gm_now = gp_now / rev_now
            gm_prev = gp_prev / rev_prev
            c.gross_margin_improving = gm_now > gm_prev

        # ── F9: Asset turnover improving ────────────────────────────────
        if (rev_now is not None and ta_now and ta_now != 0
                and rev_prev is not None and ta_prev and ta_prev != 0):
            at_now = rev_now / ta_now
            at_prev = rev_prev / ta_prev
            c.asset_turnover_improving = at_now > at_prev

    except Exception as exc:
        logger.warning("Piotroski computation error: %s", exc)

    criteria_values = [
        c.roa_positive, c.ocf_positive, c.roa_improving, c.accruals_low,
        c.leverage_falling, c.liquidity_improving, c.no_dilution,
        c.gross_margin_improving, c.asset_turnover_improving,
    ]
    available = sum(1 for v in criteria_values if v is not None)

    if available < 4:
        return None

    score = sum(1 for v in criteria_values if v is True)
    label = "Strong" if score >= 7 else ("Neutral" if score >= 4 else "Weak")

    return PiotroskiResult(score=score, criteria=c, criteria_available=available, label=label)


# ---------------------------------------------------------------------------
# Signal derivation
# ---------------------------------------------------------------------------


def _valuation_signal(fund: FundamentalData) -> str:
    """Rough valuation signal based on P/E and P/B vs broad NSE norms."""
    pe = fund.pe_trailing
    pb = fund.pb

    cheap_flags = 0
    expensive_flags = 0
    checked = 0

    if pe is not None and pe > 0:
        checked += 1
        if pe < 15:
            cheap_flags += 1
        elif pe > 35:
            expensive_flags += 1

    if pb is not None and pb > 0:
        checked += 1
        if pb < 2:
            cheap_flags += 1
        elif pb > 6:
            expensive_flags += 1

    if checked == 0:
        return "N/A"
    if expensive_flags >= 1 and cheap_flags == 0:
        return "Expensive"
    if cheap_flags >= 1 and expensive_flags == 0:
        return "Cheap"
    return "Fair"


def _quality_signal(fund: FundamentalData, piotroski: PiotroskiResult | None) -> str:
    """Quality signal combining ROE, D/E, and Piotroski."""
    if piotroski is not None:
        if piotroski.score >= 7:
            return "High"
        if piotroski.score <= 3:
            return "Low"

    roe = fund.roe
    de = fund.debt_to_equity

    high_flags = 0
    low_flags = 0

    if roe is not None:
        if roe > 0.15:
            high_flags += 1
        elif roe < 0.05:
            low_flags += 1

    if de is not None:
        if de < 0.5:
            high_flags += 1
        elif de > 2.0:
            low_flags += 1

    if high_flags > low_flags:
        return "High"
    if low_flags > high_flags:
        return "Low"
    if high_flags == 0 and low_flags == 0:
        return "N/A"
    return "Medium"


# ---------------------------------------------------------------------------
# Statement parsing helpers
# ---------------------------------------------------------------------------


def _get(df: pd.DataFrame, *row_keys: str, col: int = 0) -> float | None:
    """Safely extract a scalar from a yfinance financial statement DataFrame."""
    if df is None or df.empty:
        return None
    for key in row_keys:
        if key in df.index:
            try:
                cols = df.columns.tolist()
                if col >= len(cols):
                    return None
                val = df.loc[key, cols[col]]
                if pd.isna(val):
                    return None
                return float(val)
            except Exception:
                continue
    return None
