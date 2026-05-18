# NSE Stock Analyzer

A stock analysis dashboard for NSE-listed equities, built for semi-professional investors. Select any stock and get a unified view of risk metrics, technical indicators, valuation multiples, and factor exposures — plus a plain-English guide explaining how to use each metric to make decisions.

![Python](https://img.shields.io/badge/python-3.9+-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.35+-red) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

| Tab | What you get |
|---|---|
| **Overview** | Price chart vs Nifty 50, signal scorecard, composite decision card (0–100 score with Avoid → Consider verdict) |
| **Risk Metrics** | Alpha, Beta, R², Sharpe, Sortino, Calmar, Max Drawdown, VaR 95%/99%, rolling alpha/beta charts |
| **Technicals** | Candlestick + RSI + MACD subplots, SMA/EMA overlays, Bollinger Bands, ATR, OBV, volume ratio, 52-week range |
| **Valuation** | P/E, P/B, EV/EBITDA, P/S, ROE, ROA, Debt/Equity, Interest Coverage, Piotroski F-Score (9-criteria breakdown) |
| **Factor Model** | India Fama-French proxy regression (Market + SMB), rolling factor loadings, size/value tilt interpretation |
| **How to Use** | Plain-English guide for every metric — what it means, how to read it, decision frameworks, glossary |

---

## Quickstart

```bash
# Clone
git clone https://github.com/maanitmehta/nifty-stock-analyzer.git
cd nifty-stock-analyzer

# Set up environment
make install-dev

# (Optional) Download full NSE universe (~1800 stocks, otherwise uses Nifty 50)
make universe

# Run the dashboard
make run
```

Open **http://localhost:8501** in your browser.

---

## Installation

Requires Python 3.9+.

```bash
# Runtime only
make install

# With dev tools (pytest, ruff, black, mypy)
make install-dev
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Usage

### Run the dashboard
```bash
make run
# or
.venv/bin/streamlit run src/nifty_analyzer/ui/app.py
```

### Expand to all NSE stocks
By default the stock selector shows Nifty 50. To unlock all ~1800 NSE-listed equities:
```bash
make universe
```
This downloads the NSE equity list and saves it to `data/universe/nse_all.csv`. Re-run weekly to stay current.

### Run tests
```bash
make test
```

### Lint and format
```bash
make lint   # ruff
make fmt    # black
```

---

## Project Structure

```
nifty-stock-analyzer/
├── src/nifty_analyzer/
│   ├── config.py              # Settings (cache TTL, risk-free rate, paths)
│   ├── universe.py            # Stock universe loader (Nifty 50 / Nifty 500 / all NSE)
│   ├── data/
│   │   ├── fetcher.py         # OHLCV price pipeline via yfinance (parquet cache)
│   │   └── fundamentals.py    # Fundamental data fetcher (P/E, P/B, financials)
│   ├── features/
│   │   ├── returns.py         # Log returns, annualised return/vol, VaR, rolling vol
│   │   ├── risk.py            # Max drawdown, drawdown series, duration
│   │   ├── capm.py            # CAPM regression, rolling alpha/beta, Sharpe, Sortino, Calmar
│   │   ├── technicals.py      # RSI, MACD, MAs, Bollinger, ATR, OBV, volume, 52W range
│   │   ├── valuation.py       # Valuation multiples + Piotroski F-score
│   │   ├── factor_model.py    # India FF3 proxy, rolling factor loadings
│   │   ├── snapshot.py        # Risk metric orchestrator
│   │   └── decision_card.py   # Signal aggregator → composite score + verdict
│   └── ui/
│       ├── app.py             # Streamlit dashboard
│       └── charts.py          # Plotly chart builders
├── data/
│   └── universe/
│       └── nifty50.csv        # Curated Nifty 50 with sector/industry metadata
├── scripts/
│   └── download_universe.py   # Fetches full NSE equity list from NSE archives
├── tests/                     # 179 unit tests, 56% coverage
├── Dockerfile
├── Makefile
└── pyproject.toml
```

---

## Data Sources

| Data | Source | Refresh |
|---|---|---|
| OHLCV prices | Yahoo Finance via `yfinance` | Cached 24h |
| Fundamental data | Yahoo Finance `.info` + financial statements | Cached 7 days |
| NSE equity list | NSE India archives (`EQUITY_L.csv`) | Manual (`make universe`) |
| Risk-free rate | RBI 91-day T-bill (~6.5% annualised, configurable) | Static |

---

## Configuration

Copy `.env.example` to `.env` to override defaults:

```bash
RISK_FREE_RATE=0.065        # Annualised risk-free rate
CACHE_MAX_AGE_HOURS=24      # Price cache TTL
```

---

## Docker

```bash
make docker-build
make docker-run   # runs on http://localhost:8501
```

The `data/cache` directory is mounted as a volume so the price cache persists across container restarts.

---

## Deployment

**Streamlit Cloud (free):**
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Point to `src/nifty_analyzer/ui/app.py`
4. Deploy — Streamlit Cloud reads `requirements.txt` automatically

**Self-hosted:**
```bash
docker build -t nifty-analyzer .
docker run -p 8501:8501 -v $(pwd)/data/cache:/app/data/cache nifty-analyzer
```

---

## Disclaimer

This tool is for informational and research purposes only. Nothing here constitutes financial advice or a solicitation to buy or sell any security. All metrics are computed from historical data and do not predict future returns.
