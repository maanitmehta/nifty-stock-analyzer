"""Universe management: load stock lists from curated CSVs or the live NSE equity file."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from .config import settings

logger = logging.getLogger(__name__)

UniverseFilter = Literal["nifty50", "nifty500", "nse_all"]

# Columns guaranteed in our curated CSVs (nifty50.csv / nifty500.csv)
_CURATED_REQUIRED = {"nse_symbol", "company_name", "sector"}

# Column mapping from the raw NSE equity list CSV to our internal schema
_NSE_RAW_COL_MAP = {
    "SYMBOL": "nse_symbol",
    "NAME OF COMPANY": "company_name",
    "SERIES": "series",
    "ISIN NUMBER": "isin",
    "FACE VALUE": "face_value",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_universe(filter: UniverseFilter = "nifty50") -> pd.DataFrame:
    """Return a DataFrame of stocks for the given universe filter.

    Filters:
        nifty50  — 50-stock curated CSV with sector / industry metadata.
        nifty500 — 500-stock curated CSV (must be placed at data/universe/nifty500.csv).
        nse_all  — Full NSE equity list loaded from data/universe/nse_all.csv.
                   Run `scripts/download_universe.py` to generate this file first.
    """
    if filter in ("nifty50", "nifty500"):
        return _load_curated(filter)
    if filter == "nse_all":
        return _load_nse_all()
    raise ValueError(f"Unknown filter: {filter!r}. Choose nifty50, nifty500, or nse_all.")


def get_ticker_list(filter: UniverseFilter = "nifty50") -> list[str]:
    """Return yfinance-compatible tickers (e.g. 'RELIANCE.NS') for the given filter."""
    df = load_universe(filter)
    return [f"{sym}.NS" for sym in df["nse_symbol"].tolist()]


def get_metadata(nse_symbol: str) -> dict[str, object]:
    """Return metadata dict for a single stock, searching all available universe files."""
    for filter_ in ("nifty50", "nifty500", "nse_all"):
        path = _universe_path(filter_)
        if not path.exists():
            continue
        try:
            df = load_universe(filter_)  # type: ignore[arg-type]
        except Exception:
            continue
        row = df[df["nse_symbol"] == nse_symbol]
        if not row.empty:
            return row.iloc[0].to_dict()
    raise KeyError(
        f"Stock {nse_symbol!r} not found in any loaded universe. "
        "If it is an NSE stock outside Nifty 50/500, run scripts/download_universe.py first."
    )


def search_stocks(query: str, filter: UniverseFilter = "nse_all") -> pd.DataFrame:
    """Case-insensitive substring search across nse_symbol and company_name."""
    df = load_universe(filter)
    q = query.strip().lower()
    mask = df["nse_symbol"].str.lower().str.contains(q, na=False) | df[
        "company_name"
    ].str.lower().str.contains(q, na=False)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _universe_path(filter: str) -> Path:
    return settings.universe_dir / f"{filter}.csv"


def _load_curated(filter: str) -> pd.DataFrame:
    path = _universe_path(filter)
    if not path.exists():
        raise FileNotFoundError(
            f"Universe file not found: {path}\n"
            f"Place the {filter}.csv file in {settings.universe_dir}."
        )
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = _CURATED_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Universe CSV {path.name} is missing columns: {missing}")
    return df


def _load_nse_all() -> pd.DataFrame:
    path = _universe_path("nse_all")
    if not path.exists():
        raise FileNotFoundError(
            f"Full NSE universe file not found: {path}\n"
            "Run:  python scripts/download_universe.py"
        )
    df = pd.read_csv(path, dtype=str).fillna("")
    if "nse_symbol" not in df.columns:
        raise ValueError(
            f"{path.name} is missing 'nse_symbol' column. "
            "Re-run scripts/download_universe.py to regenerate it."
        )
    # Keep only EQ series (exclude bonds, ETFs, etc.) when the column is present
    if "series" in df.columns:
        df = df[df["series"] == "EQ"].copy()
    return df.reset_index(drop=True)
