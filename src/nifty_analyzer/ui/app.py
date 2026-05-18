"""NSE Stock Analyzer — Streamlit dashboard.

Run with:
    streamlit run src/nifty_analyzer/ui/app.py
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st

from nifty_analyzer.data.fetcher import PriceDataFetcher
from nifty_analyzer.data.fundamentals import FundamentalFetcher
from nifty_analyzer.features.decision_card import DecisionCard, generate_decision_card
from nifty_analyzer.features.factor_model import (
    FactorData,
    FF3Result,
    construct_india_factors,
    ff3_regression,
    interpret_hml,
    interpret_smb,
    rolling_ff3,
)
from nifty_analyzer.features.snapshot import RiskSnapshot, compute_snapshot
from nifty_analyzer.features.technicals import TechnicalSnapshot, compute_technicals
from nifty_analyzer.features.valuation import ValuationSnapshot, compute_valuation
from nifty_analyzer.ui.charts import (
    candlestick_chart,
    drawdown_chart,
    equity_curve_chart,
    rolling_capm_chart,
    volume_chart,
)
from nifty_analyzer.universe import load_universe, search_stocks

# ── Page config must be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="NSE Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def _pct(val: float, decimals: int = 2) -> str:
    if math.isnan(val):
        return "N/A"
    return f"{val * 100:+.{decimals}f}%"


def _fmt(val: float, decimals: int = 2, suffix: str = "") -> str:
    if val is None or math.isnan(val):
        return "N/A"
    return f"{val:.{decimals}f}{suffix}"


def _signal_badge(label: str, color: str) -> str:
    """Return an HTML span badge."""
    colors = {"green": "#10B981", "red": "#EF4444", "yellow": "#F59E0B", "gray": "#94A3B8"}
    c = colors.get(color, "#94A3B8")
    return f'<span style="background:{c};color:white;padding:2px 8px;border-radius:4px;font-size:0.8rem">{label}</span>'


@st.cache_data(ttl=3600, show_spinner=False)
def _load_universe_cached(filter_name: str) -> pd.DataFrame:
    return load_universe(filter_name)  # type: ignore[arg-type]


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_prices(ticker: str, lookback_years: int) -> pd.DataFrame:
    return PriceDataFetcher().fetch(ticker, lookback_years)  # type: ignore[arg-type]


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_benchmark(lookback_years: int) -> pd.DataFrame:
    return PriceDataFetcher().fetch_benchmark(lookback_years)  # type: ignore[arg-type]


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_fundamentals(ticker: str):
    return FundamentalFetcher().fetch(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_factors(lookback_years: int):
    try:
        return construct_india_factors(lookback_years)
    except Exception:
        return None


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 NSE Analyzer")
    st.markdown("---")

    # Universe selector — Nifty 50 and Nifty 500 bundled; All NSE needs download
    _pkg_data = Path(__file__).parent / "data_files"
    _nse_all_path = _pkg_data.parent.parent.parent.parent.parent / "data" / "universe" / "nse_all.csv"

    universe_options = ["Nifty 50", "Nifty 500"]
    if _nse_all_path.exists():
        universe_options.append("All NSE Stocks (~1800)")

    universe_label = st.selectbox("Universe", universe_options)
    universe_filter = {"Nifty 50": "nifty50", "Nifty 500": "nifty500", "All NSE Stocks (~1800)": "nse_all"}[universe_label]

    if "All NSE" not in universe_label:
        st.caption(f"{'500' if '500' in universe_label else '50'} stocks · "
                   "Run `make universe` to unlock all ~1800 NSE stocks.")

    # Load universe
    try:
        df_universe = _load_universe_cached(universe_filter)
    except FileNotFoundError:
        df_universe = _load_universe_cached("nifty50")
        universe_filter = "nifty50"

    # Search box → filtered dropdown
    search = st.text_input("🔍 Search stock", placeholder="e.g. Infosys, RELIANCE…")
    if search.strip():
        filtered = search_stocks(search, filter=universe_filter)  # type: ignore[arg-type]
    else:
        filtered = df_universe

    options = [
        f"{row['nse_symbol']}  —  {row['company_name']}"
        for _, row in filtered.iterrows()
    ]
    if not options:
        st.warning("No stocks matched your search.")
        st.stop()

    selected_option = st.selectbox("Select stock", options)
    nse_symbol = selected_option.split("  —  ")[0].strip()
    ticker = f"{nse_symbol}.NS"

    st.markdown("---")
    lookback_years = st.select_slider("Lookback", options=[1, 3, 5], value=3)

    ma_overlays = st.multiselect(
        "Chart overlays",
        options=["SMA 20", "SMA 50", "SMA 200", "EMA 12", "EMA 26", "Bollinger Upper", "Bollinger Lower"],
        default=["SMA 50", "SMA 200"],
    )

    chart_range = st.selectbox("Chart window", ["3M", "6M", "1Y", "All"], index=2)


# ── Data loading ─────────────────────────────────────────────────────────────

with st.spinner(f"Loading data for {nse_symbol}…"):
    try:
        stock_prices = _fetch_prices(ticker, lookback_years)
        market_prices = _fetch_benchmark(lookback_years)
    except Exception as exc:
        st.error(f"**Data error:** {exc}")
        st.info("Check that the ticker exists on NSE and yfinance can reach it.")
        st.stop()

# Trim to chart window
_WINDOW_DAYS = {"3M": 63, "6M": 126, "1Y": 252, "All": 99999}
n_days = _WINDOW_DAYS[chart_range]
chart_prices = stock_prices.iloc[-n_days:]

# Compute metrics
snapshot: RiskSnapshot = compute_snapshot(ticker, stock_prices, market_prices, lookback_years)
tech: TechnicalSnapshot = compute_technicals(chart_prices)
tech_full: TechnicalSnapshot = compute_technicals(stock_prices)

# Factor model (non-blocking — fetch happens in background via cache)
factors: FactorData | None = _fetch_factors(lookback_years)
ff3: FF3Result | None = None
roll_ff3_df = None
if factors is not None:
    try:
        from nifty_analyzer.features.returns import log_returns as _lr
        stock_ret_full = _lr(stock_prices["Close"])
        ff3 = ff3_regression(stock_ret_full, factors)
        roll_ff3_df = rolling_ff3(stock_ret_full, factors)
    except Exception:
        pass

# Valuation (lazy — fetched inside the tab, but we need val for decision card)
_val_cache: ValuationSnapshot | None = None

def _get_val() -> ValuationSnapshot | None:
    global _val_cache
    if _val_cache is None:
        try:
            fund = _fetch_fundamentals(ticker)
            _val_cache = compute_valuation(fund)
        except Exception:
            pass
    return _val_cache


# ── Header ───────────────────────────────────────────────────────────────────

company_name = selected_option.split("  —  ")[1].strip()
current_price = float(stock_prices["Close"].iloc[-1])
prev_price = float(stock_prices["Close"].iloc[-2])
day_chg = (current_price - prev_price) / prev_price

# Try to get sector from universe
try:
    meta = df_universe[df_universe["nse_symbol"] == nse_symbol].iloc[0].to_dict()
    sector = meta.get("sector", "")
    industry = meta.get("industry", "")
except (IndexError, KeyError):
    sector = industry = ""

st.markdown(f"## {company_name} `{nse_symbol}`")
col_price, col_chg, col_sector = st.columns([2, 2, 4])
with col_price:
    st.metric("Current Price (₹)", f"{current_price:,.2f}", delta=f"{day_chg*100:+.2f}% 1D")
with col_chg:
    ret_1y_label = _pct(snapshot.annualized_return) + " ann."
    st.metric(f"Ann. Return ({lookback_years}Y)", ret_1y_label)
with col_sector:
    if sector:
        st.markdown(f"**Sector:** {sector}  \n**Industry:** {industry}")

st.markdown("---")


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_risk, tab_technicals, tab_valuation, tab_factor, tab_guide = st.tabs(
    ["📊 Overview", "⚖️ Risk Metrics", "📉 Technicals", "🏷️ Valuation", "🔬 Factor Model", "📖 How to Use"]
)


# ── TAB 1: Overview ──────────────────────────────────────────────────────────

with tab_overview:
    # Quick signal scorecard
    def _rsi_signal() -> tuple[str, str]:
        rsi = tech_full.rsi.current
        if math.isnan(rsi):
            return "N/A", "gray"
        if rsi > 70:
            return "Overbought", "red"
        if rsi < 30:
            return "Oversold", "green"
        return "Neutral", "yellow"

    def _trend_signal() -> tuple[str, str]:
        ma = tech_full.ma
        if ma.golden_cross:
            return "Golden Cross 🌟", "green"
        if ma.death_cross:
            return "Death Cross ⚠️", "red"
        if ma.price_above_sma_200:
            return "Above 200 SMA", "green"
        return "Below 200 SMA", "red"

    def _macd_signal() -> tuple[str, str]:
        if tech_full.macd.is_bullish_crossover:
            return "Bullish Cross", "green"
        if tech_full.macd.is_bearish_crossover:
            return "Bearish Cross", "red"
        if tech_full.macd.current_histogram > 0:
            return "Bullish", "green"
        return "Bearish", "red"

    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, (label, (sig, color)) in zip(
        [sc1, sc2, sc3, sc4],
        [
            ("Momentum (RSI)", _rsi_signal()),
            ("Trend (MA)", _trend_signal()),
            ("MACD", _macd_signal()),
            ("vs 52W High", (f"{tech_full.range_52w.pct_from_high * 100:+.1f}%",
                             "green" if tech_full.range_52w.pct_from_high > -0.05 else "red")),
        ],
    ):
        with col:
            st.markdown(f"**{label}**")
            st.markdown(_signal_badge(sig, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Decision Card ────────────────────────────────────────────────────
    st.subheader("Decision Summary")
    card: DecisionCard = generate_decision_card(ticker, snapshot, tech_full, _get_val())

    _LABEL_COLOR = {
        "Consider": "#10B981", "Watchlist": "#34D399",
        "Neutral": "#F59E0B", "Caution": "#FB923C", "Avoid": "#EF4444",
    }
    card_color = _LABEL_COLOR.get(card.label, "#94A3B8")
    st.markdown(
        f'<div style="background:{card_color}22;border-left:4px solid {card_color};'
        f'padding:12px 16px;border-radius:6px;margin-bottom:12px">'
        f'<span style="font-size:1.4rem;font-weight:700;color:{card_color}">'
        f'{card.label}</span>'
        f'<span style="color:#94A3B8;margin-left:12px">Composite score: {card.score}/100</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    dc1, dc2, dc3, dc4 = st.columns(4)
    _SIG_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
    dc1.markdown(f"**Risk**  \n{_SIG_EMOJI[card.risk_signal]} {card.risk_signal.title()}")
    dc2.markdown(f"**Valuation**  \n{_SIG_EMOJI[card.valuation_signal]} {card.valuation_signal.title()}")
    dc3.markdown(f"**Momentum**  \n{_SIG_EMOJI[card.momentum_signal]} {card.momentum_signal.title()}")
    dc4.markdown(f"**Quality**  \n{_SIG_EMOJI[card.quality_signal]} {card.quality_signal.title()}")

    if card.positives or card.negatives:
        with st.expander("📋 Key findings", expanded=False):
            if card.positives:
                st.markdown("**Positives**")
                for p in card.positives:
                    st.markdown(f"✅ {p}")
            if card.negatives:
                st.markdown("**Negatives**")
                for n in card.negatives:
                    st.markdown(f"❌ {n}")
            if card.key_risks:
                st.markdown("**Key risks**")
                for r in card.key_risks:
                    st.markdown(f"⚠️ {r}")

    st.markdown("---")
    st.plotly_chart(equity_curve_chart(snapshot, market_prices), use_container_width=True)


# ── TAB 2: Risk Metrics ───────────────────────────────────────────────────────

with tab_risk:
    st.subheader(f"Risk Metrics — {lookback_years}Y Lookback")

    capm = snapshot.capm
    ratios = snapshot.ratios

    # --- Metric cards row 1 ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Alpha (ann.)", _pct(capm.alpha_annualized) if capm else "N/A",
              help="Excess return vs Nifty 50 after adjusting for market risk. "
                   "+5% means the stock beat the index by 5% per year on a risk-adjusted basis. "
                   "Only trust it if the p-value below is <0.05.")
    m2.metric("Beta", _fmt(capm.beta) if capm else "N/A",
              help="How much the stock moves relative to Nifty 50. "
                   "β=1.5 → if Nifty falls 10%, expect ~15% drop. "
                   "β<1 = defensive, β>1 = aggressive/cyclical.")
    m3.metric("R²", _fmt(capm.r_squared) if capm else "N/A",
              help="How much of this stock's movement is explained by the market. "
                   "R²=0.8 means 80% is market-driven, 20% is stock-specific. "
                   "Low R² → alpha/beta are less reliable signals.")
    m4.metric("Alpha p-value", _fmt(capm.alpha_pvalue, 3) if capm else "N/A",
              help="Is the alpha statistically real or just noise? "
                   "<0.05 = significant (trust the alpha). "
                   ">0.10 = could easily be luck — don't act on it alone.")

    # --- Metric cards row 2 ---
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Sharpe Ratio", _fmt(ratios.sharpe),
              help="Return per unit of total risk (volatility). "
                   ">1.0 = good, >2.0 = excellent. "
                   "Compare two stocks: the higher Sharpe earns more per unit of risk taken.")
    m6.metric("Sortino Ratio", _fmt(ratios.sortino),
              help="Like Sharpe but only penalises downside moves, not upside. "
                   "Preferred for stocks with asymmetric returns (large wins, small losses). "
                   "Sortino > Sharpe is normal and healthy.")
    m7.metric("Calmar Ratio", _fmt(ratios.calmar),
              help="Annualised return divided by the worst drawdown. "
                   "Answers: 'How much pain did investors endure to earn this return?' "
                   ">0.5 = reasonable, >1.0 = strong capital efficiency.")
    m8.metric("Max Drawdown", _pct(snapshot.drawdown.max_drawdown),
              help="The worst peak-to-trough loss in the lookback period. "
                   "Ask yourself: could you have held through this without panic-selling? "
                   "If not, position-size accordingly.")

    # --- Metric cards row 3 ---
    m9, m10, m11, m12 = st.columns(4)
    m9.metric("Ann. Return", _pct(snapshot.annualized_return))
    m10.metric("Ann. Volatility", _pct(snapshot.annualized_std))
    m11.metric("VaR 95% (1D)", _pct(snapshot.var_95))
    m12.metric("VaR 99% (1D)", _pct(snapshot.var_99))

    st.markdown("---")
    st.plotly_chart(drawdown_chart(snapshot), use_container_width=True)

    if not snapshot.rolling_capm_df.empty:
        st.plotly_chart(rolling_capm_chart(snapshot), use_container_width=True)
    else:
        st.info("Not enough data for rolling CAPM chart (need >252 trading days).")


# ── TAB 3: Technicals ─────────────────────────────────────────────────────────

with tab_technicals:
    st.subheader("Technical Indicators")

    # Latest values summary row
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("RSI (14)", _fmt(tech.rsi.current),
              delta="Overbought" if tech.rsi.is_overbought else ("Oversold" if tech.rsi.is_oversold else "Neutral"))
    t2.metric("MACD", _fmt(tech.macd.current_macd, 4))
    t3.metric("Signal", _fmt(tech.macd.current_signal, 4))
    t4.metric("ATR (14)", f"₹{tech.atr_current:,.1f}" if not math.isnan(tech.atr_current) else "N/A")
    t5.metric("Vol Ratio (20D)", _fmt(tech.volume.ratio, 2) + "×",
              delta="Above avg" if tech.volume.is_above_average else "Below avg")

    st.markdown("---")

    # MA summary
    mac = tech.ma
    ma1, ma2, ma3, ma4 = st.columns(4)
    ma1.metric("vs SMA 50", _pct(mac.pct_from_sma_50))
    ma2.metric("vs SMA 200", _pct(mac.pct_from_sma_200))
    ma3.metric("52W High", _pct(tech.range_52w.pct_from_high))
    ma4.metric("52W Low (above)", _pct(tech.range_52w.pct_from_low))

    # Candlestick + RSI + MACD chart
    st.plotly_chart(candlestick_chart(chart_prices, tech, ma_overlays), use_container_width=True)

    # Volume chart
    st.plotly_chart(volume_chart(chart_prices, tech), use_container_width=True)


# ── TAB 4: Valuation ──────────────────────────────────────────────────────────

with tab_valuation:
    with st.spinner("Loading fundamental data…"):
        try:
            fund = _fetch_fundamentals(ticker)
            val: ValuationSnapshot = compute_valuation(fund)
        except Exception as exc:
            st.error(f"Could not load fundamental data: {exc}")
            val = None  # type: ignore[assignment]

    if val is None:
        st.info("Fundamental data unavailable for this ticker.")
    else:
        # ── Signal badges ────────────────────────────────────────────────
        b1, b2, b3 = st.columns(3)
        _VAL_COLOR = {"Cheap": "green", "Fair": "yellow", "Expensive": "red", "N/A": "gray"}
        _QUAL_COLOR = {"High": "green", "Medium": "yellow", "Low": "red", "N/A": "gray"}
        with b1:
            st.markdown("**Valuation**")
            st.markdown(_signal_badge(val.valuation_signal, _VAL_COLOR[val.valuation_signal]),
                        unsafe_allow_html=True)
        with b2:
            st.markdown("**Quality**")
            st.markdown(_signal_badge(val.quality_signal, _QUAL_COLOR[val.quality_signal]),
                        unsafe_allow_html=True)
        with b3:
            if val.piotroski:
                st.markdown("**Piotroski F-Score**")
                p_color = "green" if val.piotroski.score >= 7 else ("red" if val.piotroski.score <= 3 else "yellow")
                st.markdown(_signal_badge(f"{val.piotroski.score}/9 — {val.piotroski.label}", p_color),
                            unsafe_allow_html=True)

        st.markdown("---")

        # ── Valuation multiples ──────────────────────────────────────────
        st.subheader("Valuation Multiples")
        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("P/E (Trailing)", _fmt(val.pe_trailing, 1) + "×" if val.pe_trailing else "N/A",
                  help="Price / Trailing 12-month EPS. Lower = cheaper relative to earnings.")
        v2.metric("P/E (Forward)", _fmt(val.pe_forward, 1) + "×" if val.pe_forward else "N/A",
                  help="Price / Analyst consensus forward EPS estimate.")
        v3.metric("P/B", _fmt(val.pb, 2) + "×" if val.pb else "N/A",
                  help="Price / Book Value per share. <1 may indicate undervaluation.")
        v4.metric("EV/EBITDA", _fmt(val.ev_ebitda, 1) + "×" if val.ev_ebitda else "N/A",
                  help="Enterprise Value / EBITDA. Capital-structure-neutral valuation.")
        v5.metric("P/S", _fmt(val.ps, 2) + "×" if val.ps else "N/A",
                  help="Price / Sales. Useful for loss-making or early-stage companies.")

        st.markdown("---")

        # ── Profitability & quality ──────────────────────────────────────
        st.subheader("Profitability & Financial Health")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("ROE", _pct(val.roe) if val.roe is not None else "N/A",
                  help="Return on Equity. >15% is generally strong for Indian large-caps.")
        q2.metric("ROA", _pct(val.roa) if val.roa is not None else "N/A",
                  help="Return on Assets. Measures capital efficiency regardless of leverage.")
        q3.metric("Gross Margin", _pct(val.gross_margin) if val.gross_margin is not None else "N/A",
                  help="Gross Profit / Revenue. Higher and stable = pricing power.")
        q4.metric("Operating Margin", _pct(val.operating_margin) if val.operating_margin is not None else "N/A",
                  help="Operating Profit / Revenue. Measures operational efficiency.")

        q5, q6, q7, q8 = st.columns(4)
        q5.metric("Net Margin", _pct(val.net_margin) if val.net_margin is not None else "N/A",
                  help="Net Profit / Revenue (bottom-line profitability).")
        q6.metric("Debt / Equity", _fmt(val.debt_to_equity) if val.debt_to_equity is not None else "N/A",
                  help="Total Debt / Shareholders' Equity. <1 = conservative balance sheet.")
        q7.metric("Current Ratio", _fmt(val.current_ratio) if val.current_ratio is not None else "N/A",
                  help="Current Assets / Current Liabilities. >1.5 = healthy short-term liquidity.")
        q8.metric("Interest Coverage", _fmt(val.interest_coverage) if val.interest_coverage is not None else "N/A",
                  help="EBIT / Interest Expense. >3× = comfortable debt servicing. <1.5× = warning.")

        st.markdown("---")

        # ── Growth ──────────────────────────────────────────────────────
        st.subheader("Growth")
        g1, g2 = st.columns(2)
        g1.metric("Revenue Growth (YoY)",
                  _pct(val.revenue_growth_yoy) if val.revenue_growth_yoy is not None else "N/A",
                  help="Year-over-year revenue growth rate (TTM vs prior year).")
        g2.metric("Earnings Growth (YoY)",
                  _pct(val.earnings_growth_yoy) if val.earnings_growth_yoy is not None else "N/A",
                  help="Year-over-year EPS growth rate.")

        # ── Piotroski breakdown ──────────────────────────────────────────
        if val.piotroski:
            st.markdown("---")
            st.subheader(f"Piotroski F-Score: {val.piotroski.score}/9 — {val.piotroski.label}")
            st.caption(
                f"Computed from {val.piotroski.criteria_available}/9 criteria "
                "(criteria with missing data are excluded from the score)."
            )

            c = val.piotroski.criteria

            def _bool_badge(val: bool | None, label: str) -> str:
                if val is None:
                    return f"⬜ {label} *(data unavailable)*"
                return f"{'✅' if val else '❌'} {label}"

            col_prof, col_lev, col_eff = st.columns(3)
            with col_prof:
                st.markdown("**Profitability**")
                st.markdown(_bool_badge(c.roa_positive, "ROA > 0"))
                st.markdown(_bool_badge(c.ocf_positive, "Operating Cash Flow > 0"))
                st.markdown(_bool_badge(c.roa_improving, "ROA improving YoY"))
                st.markdown(_bool_badge(c.accruals_low, "Cash earnings quality (OCF > ROA)"))
            with col_lev:
                st.markdown("**Leverage & Liquidity**")
                st.markdown(_bool_badge(c.leverage_falling, "Long-term debt falling"))
                st.markdown(_bool_badge(c.liquidity_improving, "Current ratio improving"))
                st.markdown(_bool_badge(c.no_dilution, "No share dilution"))
            with col_eff:
                st.markdown("**Operating Efficiency**")
                st.markdown(_bool_badge(c.gross_margin_improving, "Gross margin improving"))
                st.markdown(_bool_badge(c.asset_turnover_improving, "Asset turnover improving"))
        else:
            st.info(
                "Piotroski F-Score requires at least 2 years of financial statement data. "
                "Not enough data available from yfinance for this stock."
            )

        if fund.fetched_at:
            st.caption(f"Fundamental data fetched at: {fund.fetched_at[:19]} UTC")


# ── TAB 5: Factor Model ───────────────────────────────────────────────────────

with tab_factor:
    st.subheader("Factor Model — India Fama-French Proxy")
    st.caption(
        "Market factor: Nifty 50 (^NSEI). "
        "SMB proxy: Nifty Midcap 50 (^NSEMDCP50) − Nifty 50. "
        "HML: not available (requires cross-sectional book-value data). "
        "Model falls back to 2-factor (Market + SMB) when midcap index is unavailable."
    )

    if factors is None:
        st.warning("Factor data could not be loaded. Check your internet connection.")
    elif ff3 is None:
        st.warning("Factor regression failed — insufficient overlapping observations.")
    else:
        # ── Factor loadings ───────────────────────────────────────────────
        st.markdown(f"**Factors used:** {', '.join(ff3.factors_used)}")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Alpha (ann.)", _pct(ff3.alpha_annualized),
                  help="Excess return unexplained by factor exposures. "
                       f"p={ff3.alpha_pvalue:.3f}")
        f2.metric("Market Beta", _fmt(ff3.mkt_beta),
                  help=f"Sensitivity to Nifty 50. p={ff3.mkt_pvalue:.3f}")
        f3.metric("SMB Loading",
                  _fmt(ff3.smb_loading) if ff3.smb_loading is not None else "N/A",
                  help="Size tilt. Positive = small/midcap behaviour. "
                       + (f"p={ff3.smb_pvalue:.3f}" if ff3.smb_pvalue else ""))
        f4.metric("R²", _fmt(ff3.r_squared),
                  help="Fraction of return variance explained by the factors.")

        # Interpretation labels
        i1, i2, i3 = st.columns(3)
        with i1:
            st.markdown("**Size tilt**")
            st.markdown(_signal_badge(interpret_smb(ff3.smb_loading), "gray"), unsafe_allow_html=True)
        with i2:
            st.markdown("**Value/Growth tilt**")
            st.markdown(_signal_badge(interpret_hml(ff3.hml_loading), "gray"), unsafe_allow_html=True)
        with i3:
            alpha_sig = "green" if ff3.alpha_pvalue < 0.05 and ff3.alpha_annualized > 0 else \
                        "red" if ff3.alpha_pvalue < 0.05 and ff3.alpha_annualized < 0 else "gray"
            alpha_label = "Significant α" if ff3.alpha_pvalue < 0.05 else "α not significant"
            st.markdown("**Alpha significance**")
            st.markdown(_signal_badge(alpha_label, alpha_sig), unsafe_allow_html=True)

        st.markdown("---")

        # ── Rolling factor loadings chart ─────────────────────────────────
        if roll_ff3_df is not None and not roll_ff3_df.empty:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            has_smb = "smb_loading" in roll_ff3_df.columns
            rows = 3 if has_smb else 2
            titles = ["Rolling Alpha (annualised %)", "Rolling Market Beta"]
            if has_smb:
                titles.append("Rolling SMB Loading")

            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                                vertical_spacing=0.06, subplot_titles=titles)

            fig.add_trace(go.Scatter(
                x=roll_ff3_df.index,
                y=roll_ff3_df["alpha_annualized"] * 100,
                name="Alpha %", line=dict(color="#10B981", width=1.5)), row=1, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

            fig.add_trace(go.Scatter(
                x=roll_ff3_df.index,
                y=roll_ff3_df["mkt_beta"],
                name="Mkt Beta", line=dict(color="#F59E0B", width=1.5)), row=2, col=1)
            fig.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1)

            if has_smb:
                fig.add_trace(go.Scatter(
                    x=roll_ff3_df.index,
                    y=roll_ff3_df["smb_loading"],
                    name="SMB", line=dict(color="#8B5CF6", width=1.5)), row=3, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

            fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0),
                              showlegend=False, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Rolling factor chart requires >252 trading days of data.")

        with st.expander("How to interpret factor loadings"):
            st.markdown("""
**Market Beta:** Same as CAPM beta — sensitivity to Nifty 50 moves. In a multi-factor
model, beta is "cleaner" because SMB and HML have been controlled for.

**SMB Loading (positive):** The stock moves more like a mid/small-cap company than a
large-cap, even if it is technically large. This could reflect cyclicality, lower
liquidity, or business exposure to smaller-company sectors.

**SMB Loading (negative):** The stock exhibits large-cap-like behaviour — low volatility
relative to size, more stable earnings, stronger institutional following.

**Alpha in this model:** Any return unexplained by market, size, and value exposures.
A positive, statistically significant multi-factor alpha is a stronger quality signal
than CAPM alpha alone, because it controls for tilts that could otherwise be mistaken
for stock-picking skill.

**R² comparison:** If this model's R² is meaningfully higher than the CAPM R² (see
Risk tab), the stock has material factor tilts worth accounting for.
            """)


# ── TAB 6: How to Use ─────────────────────────────────────────────────────────

with tab_guide:
    st.markdown("## How to Use This Tool")
    st.markdown(
        "This guide explains what each metric means, what a good or bad reading looks like, "
        "and how to combine them into a decision. All metrics are descriptive — they summarise "
        "historical data. They do not predict the future."
    )

    # ── Decision Framework ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Decision Framework")
    st.info(
        "**Suggested workflow:**\n\n"
        "1. **Screen by fundamentals** (valuation tab, coming soon) — avoid overpaying.\n"
        "2. **Check the risk profile** (Risk tab) — is the risk/reward acceptable?\n"
        "3. **Use technicals for timing** (Technicals tab) — don't buy into extreme weakness or overbought momentum.\n"
        "4. **Size the position** using Max Drawdown and VaR — if you can't stomach the worst historical loss, reduce size.\n"
        "5. **Review alpha p-value** — only credit outperformance if it is statistically significant."
    )

    col_buy, col_avoid = st.columns(2)
    with col_buy:
        st.success(
            "**Signals that support a position**\n\n"
            "- Alpha > 0 and p-value < 0.05\n"
            "- Sharpe Ratio > 1.0\n"
            "- RSI 40–60 (neutral momentum — not chasing)\n"
            "- Price above 200 SMA (established uptrend)\n"
            "- MACD bullish crossover with rising histogram\n"
            "- Volume above 20-day average on up days\n"
            "- Beta appropriate for your portfolio risk budget"
        )
    with col_avoid:
        st.error(
            "**Red flags to watch for**\n\n"
            "- Alpha > 0 but p-value > 0.10 (likely noise)\n"
            "- RSI > 75 (overbought — risk of pullback)\n"
            "- Price far below 200 SMA (broken trend)\n"
            "- Death cross (50 SMA < 200 SMA) recently formed\n"
            "- Max Drawdown > 50% (extreme historical pain)\n"
            "- Sharpe < 0 (you're being paid nothing for the risk)\n"
            "- R² < 0.2 (highly idiosyncratic — alpha is unreliable)"
        )

    # ── Risk Metrics Explained ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚖️ Risk Metrics Explained")

    with st.expander("Alpha — Did the stock beat the market on a risk-adjusted basis?"):
        st.markdown("""
**What it is:** Jensen's Alpha measures how much a stock returned *above* what you would expect
given its market exposure (beta). It is computed from an OLS regression of the stock's excess
returns (above the RBI T-bill rate) on the Nifty 50's excess returns.

**Formula:** `Alpha = Stock Return − [Rf + Beta × (Market Return − Rf)]`

**How to read it:**
| Alpha | Interpretation |
|---|---|
| > +3% | Strong outperformer — stock adds value beyond market exposure |
| 0% to +3% | Modest outperformance — check p-value before acting |
| ~0% | Fairly priced for its risk; returns explained by market |
| < 0% | Underperformer — paying for risk without reward |

**Critical check:** Always look at the **p-value alongside alpha**. An alpha of +8% with a
p-value of 0.40 is statistically indistinguishable from zero — it could be random variation
over the lookback period. Only act on alpha when p-value < 0.05.

**Lookback matters:** Alpha over 1Y can be driven by a single event. 3Y is more reliable.
        """)

    with st.expander("Beta — How much does this stock amplify the market?"):
        st.markdown("""
**What it is:** Beta measures sensitivity to Nifty 50 moves. It is the slope of the regression
line between the stock's daily returns and the market's daily returns.

**How to read it:**
| Beta | Interpretation |
|---|---|
| < 0 | Inverse to market (rare — e.g. some commodities) |
| 0 to 0.5 | Very defensive — moves little with the market |
| 0.5 to 1.0 | Defensive — partially follows market |
| ~1.0 | Moves in line with Nifty 50 |
| 1.0 to 1.5 | Aggressive — amplifies market up and down |
| > 1.5 | High-beta — consider carefully in volatile markets |

**Portfolio use:** If your portfolio target beta is 1.0, adding a β=1.5 stock increases
your market exposure. To stay neutral, you'd need to pair it with a β=0.5 stock or reduce
overall position size.

**Sector patterns:** FMCG and utilities tend to have β < 1. IT, metals, and cyclicals
often have β > 1.
        """)

    with st.expander("Sharpe Ratio — Am I being rewarded enough for the risk I'm taking?"):
        st.markdown("""
**What it is:** The Sharpe Ratio divides the excess return (above the risk-free rate) by the
annualised standard deviation. It is the single most widely used risk-adjusted performance metric.

**Formula:** `Sharpe = (Annualised Return − Rf) / Annualised Volatility`

**Benchmarks (rough guide for equity):**
| Sharpe | Interpretation |
|---|---|
| < 0 | Negative real return after risk-free rate — poor |
| 0 to 0.5 | Weak — not well compensated for volatility |
| 0.5 to 1.0 | Acceptable |
| 1.0 to 2.0 | Good |
| > 2.0 | Excellent (rare to sustain over long periods) |

**Limitation:** Sharpe penalises upside volatility equally with downside.
A stock that occasionally delivers large positive surprises will look worse on Sharpe
than on Sortino. Use both together.
        """)

    with st.expander("Sortino Ratio — A fairer measure for asymmetric returns"):
        st.markdown("""
**What it is:** The Sortino Ratio is identical to Sharpe but uses only *downside* standard
deviation (returns below zero) in the denominator. It does not penalise a stock for having
high positive returns.

**Formula:** `Sortino = (Annualised Return − Rf) / Annualised Downside Volatility`

**When to prefer Sortino over Sharpe:**
- When a stock has a skewed return distribution (e.g. many small gains, rare large losses)
- When evaluating momentum or growth stocks that can have high upside variance
- For comparing stocks with similar Sharpe but different drawdown behaviour

**Rule of thumb:** If Sortino >> Sharpe, the stock's volatility is mostly to the upside —
that is a positive characteristic. If Sortino ≈ Sharpe, volatility is symmetric.
        """)

    with st.expander("Calmar Ratio — Return relative to worst-case loss"):
        st.markdown("""
**What it is:** Calmar divides the annualised return by the absolute value of the maximum
drawdown. It answers: *"How much return did investors earn per unit of peak-to-trough pain?"*

**Formula:** `Calmar = Annualised Return / |Max Drawdown|`

**Interpretation:**
| Calmar | Interpretation |
|---|---|
| < 0.25 | Poor — large drawdowns not compensated by returns |
| 0.25 to 0.5 | Below average |
| 0.5 to 1.0 | Reasonable |
| > 1.0 | Strong — efficient use of drawdown budget |

**Use case:** Calmar is especially useful when comparing two stocks with similar
Sharpe ratios but different drawdown profiles. The one with the higher Calmar
recovers faster and rewards investors more per unit of pain endured.
        """)

    with st.expander("Max Drawdown — What is the worst this stock has done?"):
        st.markdown("""
**What it is:** The largest peak-to-trough decline in the stock's price over the lookback
period, expressed as a percentage. It represents the worst loss an investor would have
experienced if they bought at the peak and sold at the trough.

**How to use it for position sizing:**
The key question is: *"If I held this stock and it fell X%, would I panic and sell?"*

If your answer is yes at a drawdown smaller than the historical max, you are probably
over-positioned. A simple rule:

> **Max position size ≈ Your personal pain threshold / Historical max drawdown**

Example: If you can tolerate a 10% portfolio impact, and the stock's max drawdown is 40%,
your position should not exceed 25% of the portfolio (10% / 40% = 25%).

**Context matters:** A -40% drawdown during the 2020 COVID crash is very different from
a -40% drawdown during a calm market — check the dates on the drawdown chart.
        """)

    with st.expander("VaR (Value at Risk) — Expected daily tail loss"):
        st.markdown("""
**What it is:** Historical simulation VaR estimates the daily loss threshold that is
exceeded on only X% of trading days.

- **VaR 95%** = the loss exceeded on only 5% of days (~1 in 20 trading days)
- **VaR 99%** = the loss exceeded on only 1% of days (~1 in 100 trading days)

**Example:** VaR 95% = 2.1% means on 19 out of 20 days, the stock lost less than 2.1%.
On roughly 1 day per month, losses exceeded 2.1%.

**Limitations:**
- Historical VaR assumes past return distributions repeat. Tail events (crashes) may be
  underestimated if the lookback period did not include a major drawdown.
- It is a one-day metric — do not extrapolate directly to weekly or monthly risk.

**Use it alongside Max Drawdown:** VaR tells you what to expect on a typical bad day;
Max Drawdown tells you the worst sustained period.
        """)

    # ── Technical Indicators Explained ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📉 Technical Indicators Explained")

    with st.expander("RSI — Is the stock overbought or oversold?"):
        st.markdown("""
**What it is:** The Relative Strength Index (RSI) measures the speed and magnitude of recent
price changes on a 0–100 scale. It is computed over a 14-day default window.

**How to read it:**
| RSI | Signal |
|---|---|
| > 70 | **Overbought** — price has risen sharply; pullback risk is elevated |
| 50–70 | Bullish momentum, but not extreme |
| 30–50 | Bearish or neutral momentum |
| < 30 | **Oversold** — price has fallen sharply; potential bounce or value opportunity |

**Important caveats:**
- RSI can stay overbought (>70) for weeks in a strong trend. Never sell a strong stock
  purely because RSI is 75.
- RSI is most reliable as a *reversal signal* in sideways/ranging markets.
- In strong uptrends, RSI rarely falls below 40. In downtrends, it rarely rises above 60.
  Adjust your interpretation accordingly.
- **RSI divergence** is a powerful signal: if price makes a new high but RSI makes a lower
  high, momentum is weakening — a potential reversal warning.
        """)

    with st.expander("MACD — Momentum direction and strength"):
        st.markdown("""
**What it is:** Moving Average Convergence Divergence (MACD) uses the difference between a
12-day and 26-day exponential moving average (the MACD line) and a 9-day EMA of that
difference (the Signal line).

**Components:**
- **MACD Line** = EMA(12) − EMA(26) — captures momentum direction
- **Signal Line** = EMA(9) of MACD — smoothed confirmation
- **Histogram** = MACD − Signal — visualises the gap between them

**Signals:**
| Event | Interpretation |
|---|---|
| MACD crosses above Signal | **Bullish crossover** — momentum turning positive |
| MACD crosses below Signal | **Bearish crossover** — momentum turning negative |
| Histogram growing above zero | Bullish momentum accelerating |
| Histogram shrinking above zero | Bullish momentum weakening |
| Histogram growing below zero | Bearish momentum accelerating |

**Best used with:** Moving averages (confirm MACD signal is in the direction of the trend)
and volume (a crossover on high volume is more reliable).
        """)

    with st.expander("Moving Averages — Trend direction and key support/resistance"):
        st.markdown("""
**What they are:** Simple (SMA) and Exponential (EMA) moving averages smooth daily price
noise to reveal the underlying trend direction.

**Key levels:**
| Average | Common use |
|---|---|
| SMA 20 | Short-term trend; active traders |
| SMA 50 | Medium-term trend; swing traders |
| SMA 200 | Long-term trend; investors |
| EMA 12/26 | Used in MACD calculation; faster-reacting |

**Price vs moving average:**
- **Price above 200 SMA** → stock is in a long-term uptrend. Pullbacks to the 200 SMA
  are often buying opportunities.
- **Price below 200 SMA** → stock is in a downtrend. Rallies to the 200 SMA are often
  resistance and potential exit points.

**Golden Cross / Death Cross:**
- **Golden Cross**: SMA 50 crosses above SMA 200 — a long-term bullish signal, often
  attracts institutional buying.
- **Death Cross**: SMA 50 crosses below SMA 200 — a long-term bearish signal, may trigger
  institutional selling.
- Both are *lagging* signals — they confirm a trend already in place, they do not predict it.
        """)

    with st.expander("Bollinger Bands — Volatility and price extremes"):
        st.markdown("""
**What they are:** Bollinger Bands place two bands at ±2 standard deviations around a
20-day simple moving average. They expand when volatility is high and contract when it is low.

**Key signals:**
| Situation | Interpretation |
|---|---|
| Price touches upper band | Price is 2σ above average — elevated, potential reversal |
| Price touches lower band | Price is 2σ below average — depressed, potential bounce |
| Bands very narrow (squeeze) | Volatility is unusually low; a large move is likely coming |
| Bands very wide | Volatility is elevated; trend may be exhausting |

**%B (pct_b):**
- %B > 1.0 → price is above the upper band
- %B = 0.5 → price is at the middle band (20 SMA)
- %B < 0.0 → price is below the lower band

**Important:** Touching a band is not a buy/sell signal by itself. In a strong trend,
price can "walk the band" for extended periods. Confirm with RSI or volume.
        """)

    with st.expander("ATR — How much does this stock move in a day?"):
        st.markdown("""
**What it is:** Average True Range (ATR) measures the average daily price range
(high minus low, adjusted for gaps) over 14 days. It is expressed in rupees, not percentage.

**Use cases:**
- **Stop-loss placement:** A common rule is to set a stop 1.5–2× ATR below your entry.
  This avoids being stopped out by normal daily noise.
- **Position sizing:** Normalise position sizes across stocks by their ATR.
  If you want to risk ₹10,000 per trade and ATR is ₹50, position = 10,000 / 50 = 200 shares.
- **Volatility regime:** A rising ATR means volatility is increasing; a falling ATR means
  the market is calming. Neither is inherently good or bad, but it changes risk management.

**Example:** ATR = ₹85 on a ₹1,000 stock means the stock typically moves ₹85 (8.5%) per day.
A stop 2× ATR below entry = ₹170 below entry price.
        """)

    with st.expander("Volume — Is price action supported by conviction?"):
        st.markdown("""
**What it is:** Volume measures the number of shares traded. The **Volume Ratio** shown
compares today's volume to the 20-day average.

**Why it matters:** Price moves on high volume are more significant than price moves on
low volume. Volume is the "conviction" behind a price change.

**Key patterns:**
| Pattern | Interpretation |
|---|---|
| Price up + volume > average | Strong buying conviction — bullish |
| Price up + volume < average | Weak move — may not sustain |
| Price down + volume > average | Strong selling pressure — bearish |
| Price down + volume < average | Lack of conviction in the sell-off — may recover |
| Steady price + volume spike | Accumulation or distribution by large players |

**On-Balance Volume (OBV):** A running total that adds volume on up days and subtracts
it on down days. A rising OBV with a flat price suggests hidden accumulation.
A falling OBV with a flat price suggests distribution — a potential warning before a sell-off.
        """)

    # ── Glossary ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔤 Quick Glossary")
    st.markdown("""
| Term | Definition |
|---|---|
| **Rf (Risk-Free Rate)** | RBI 91-day T-bill rate (~6.5%). The return you can earn with zero risk. Used as the baseline for all excess return calculations. |
| **Excess Return** | Return above the risk-free rate. The reward for taking market risk. |
| **Annualised** | A figure scaled to represent a 1-year equivalent, regardless of the actual lookback period. |
| **Lookback Period** | The historical window used to compute all metrics. 1Y is recent but noisy; 3Y is the recommended default; 5Y smooths over a full market cycle. |
| **OLS** | Ordinary Least Squares — the statistical method used to fit the CAPM regression line. |
| **52W High/Low** | The highest and lowest closing price in the last 252 trading days (approximately 1 year). |
| **EMA vs SMA** | EMA gives more weight to recent prices (faster-reacting); SMA weights all periods equally (smoother). |
    """)

    st.caption(
        "All metrics are computed from historical price data sourced from Yahoo Finance via yfinance. "
        "This tool is for informational and research purposes only. Nothing here constitutes "
        "financial advice or a solicitation to buy or sell any security."
    )
