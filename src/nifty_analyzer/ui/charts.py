"""Plotly chart builders for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..features.snapshot import RiskSnapshot
from ..features.technicals import TechnicalSnapshot


def equity_curve_chart(snapshot: RiskSnapshot, market_prices: pd.DataFrame) -> go.Figure:
    """Indexed equity curve: stock vs Nifty 50 benchmark."""
    stock = (snapshot.prices / float(snapshot.prices.iloc[0]) * 100).rename("Stock")
    mkt_close = market_prices["Close"]
    mkt_close = mkt_close[mkt_close.index >= snapshot.prices.index[0]]
    market = (mkt_close / float(mkt_close.iloc[0]) * 100).rename("Nifty 50")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock.index, y=stock, name="Stock", line=dict(color="#2563EB", width=2)))
    fig.add_trace(go.Scatter(x=market.index, y=market, name="Nifty 50", line=dict(color="#94A3B8", width=1.5, dash="dot")))
    fig.update_layout(
        title="Indexed Price (Base = 100)",
        xaxis_title=None, yaxis_title="Index",
        legend=dict(orientation="h", y=1.02),
        height=350, margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )
    return fig


def drawdown_chart(snapshot: RiskSnapshot) -> go.Figure:
    """Drawdown waterfall chart."""
    dd = snapshot.drawdown.drawdown_series * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd,
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="#EF4444", width=1),
        fillcolor="rgba(239,68,68,0.2)",
    ))
    fig.update_layout(
        title="Drawdown (%)",
        xaxis_title=None, yaxis_title="%",
        height=220, margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )
    return fig


def rolling_capm_chart(snapshot: RiskSnapshot) -> go.Figure:
    """Rolling 12-month alpha and beta."""
    df = snapshot.rolling_capm_df
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Rolling Alpha (annualized)", "Rolling Beta"))

    fig.add_trace(go.Scatter(x=df.index, y=(df["alpha_annualized"] * 100),
                             name="Alpha %", line=dict(color="#10B981")), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["beta"],
                             name="Beta", line=dict(color="#F59E0B")), row=2, col=1)
    fig.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0),
                      showlegend=False, hovermode="x unified")
    return fig


def candlestick_chart(ohlcv: pd.DataFrame, tech: TechnicalSnapshot, ma_overlays: list[str]) -> go.Figure:
    """Candlestick with optional MA overlays, RSI and MACD subplots."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.22, 0.23],
        vertical_spacing=0.04,
        subplot_titles=("Price", "RSI (14)", "MACD (12, 26, 9)"),
    )

    # --- Candlestick ---
    fig.add_trace(go.Candlestick(
        x=ohlcv.index,
        open=ohlcv["Open"], high=ohlcv["High"],
        low=ohlcv["Low"], close=ohlcv["Close"],
        name="Price", increasing_line_color="#10B981", decreasing_line_color="#EF4444",
    ), row=1, col=1)

    _MA_COLORS = {  # noqa: N806
        "SMA 20": ("#60A5FA", tech.ma.sma_20),
        "SMA 50": ("#F59E0B", tech.ma.sma_50),
        "SMA 200": ("#8B5CF6", tech.ma.sma_200),
        "EMA 12": ("#34D399", tech.ma.ema_12),
        "EMA 26": ("#FB923C", tech.ma.ema_26),
        "Bollinger Upper": ("#94A3B8", tech.bollinger.upper),
        "Bollinger Lower": ("#94A3B8", tech.bollinger.lower),
    }
    for label in ma_overlays:
        if label in _MA_COLORS:
            color, series = _MA_COLORS[label]
            dash = "dot" if "Bollinger" in label else "solid"
            fig.add_trace(go.Scatter(x=series.index, y=series, name=label,
                                     line=dict(color=color, width=1.2, dash=dash)), row=1, col=1)

    # --- RSI ---
    fig.add_trace(go.Scatter(x=tech.rsi.series.index, y=tech.rsi.series,
                             name="RSI", line=dict(color="#6366F1", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#10B981", row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.07)", layer="below", row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(16,185,129,0.07)", layer="below", row=2, col=1)

    # --- MACD ---
    colors = ["#10B981" if v >= 0 else "#EF4444" for v in tech.macd.histogram.fillna(0)]
    fig.add_trace(go.Bar(x=tech.macd.histogram.index, y=tech.macd.histogram,
                         name="Histogram", marker_color=colors, opacity=0.6), row=3, col=1)
    fig.add_trace(go.Scatter(x=tech.macd.macd_line.index, y=tech.macd.macd_line,
                             name="MACD", line=dict(color="#2563EB", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=tech.macd.signal_line.index, y=tech.macd.signal_line,
                             name="Signal", line=dict(color="#F59E0B", width=1.2, dash="dot")), row=3, col=1)

    fig.update_layout(height=700, margin=dict(l=0, r=0, t=40, b=0),
                      hovermode="x unified", showlegend=True,
                      xaxis_rangeslider_visible=False)
    return fig


def volume_chart(ohlcv: pd.DataFrame, tech: TechnicalSnapshot) -> go.Figure:
    """Volume bars with 20-day average line."""
    fig = go.Figure()
    colors = ["#10B981" if c >= o else "#EF4444"
              for c, o in zip(ohlcv["Close"], ohlcv["Open"])]
    fig.add_trace(go.Bar(x=ohlcv.index, y=ohlcv["Volume"], name="Volume",
                         marker_color=colors, opacity=0.7))
    avg = ohlcv["Volume"].rolling(20).mean()
    fig.add_trace(go.Scatter(x=avg.index, y=avg, name="20D Avg",
                             line=dict(color="#F59E0B", width=1.5, dash="dot")))
    fig.update_layout(title="Volume", height=200, margin=dict(l=0, r=0, t=30, b=0),
                      hovermode="x unified", showlegend=True)
    return fig
