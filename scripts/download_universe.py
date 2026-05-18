#!/usr/bin/env python3
"""Download the full NSE equity list and save it as data/universe/nse_all.csv.

Usage:
    python scripts/download_universe.py [--force]

The NSE archives publish a fresh EQUITY_L.csv daily. This script fetches it,
cleans the columns, filters to EQ series (excludes ETFs / bonds / SME), and
writes nse_all.csv into data/universe/.

Run this once before using the 'nse_all' universe filter, then re-run
periodically (weekly is enough — the universe changes slowly).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# Resolve project root regardless of where the script is called from
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nifty_analyzer.config import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NSE_EQUITY_URL = settings.nse_equity_list_url
OUTPUT_PATH = settings.universe_dir / "nse_all.csv"

# NSE requires a browser-like User-Agent; requests without it get blocked
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def fetch_nse_equity_list(url: str, timeout: int = 30) -> pd.DataFrame:
    logger.info("Fetching NSE equity list from %s", url)

    session = requests.Session()
    # NSE uses cookies set by the homepage; warm the session first
    try:
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=timeout)
        time.sleep(1)
    except requests.RequestException as e:
        logger.warning("Could not warm NSE session (continuing anyway): %s", e)

    resp = session.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()

    raw = pd.read_csv(StringIO(resp.text))
    logger.info("Downloaded %d rows, %d columns", len(raw), len(raw.columns))
    return raw


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw NSE CSV to internal schema."""
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    col_map = {
        "SYMBOL": "nse_symbol",
        "NAME OF COMPANY": "company_name",
        "SERIES": "series",
        "DATE OF LISTING": "listing_date",
        "PAID UP VALUE": "paid_up_value",
        "MARKET LOT": "market_lot",
        "ISIN NUMBER": "isin",
        "FACE VALUE": "face_value",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Keep only main-board equity (EQ series); drop SME, ETF, etc.
    if "series" in df.columns:
        before = len(df)
        df = df[df["series"] == "EQ"].copy()
        logger.info("Filtered to EQ series: %d → %d rows", before, len(df))

    df = df.reset_index(drop=True)
    return df


def main(force: bool = False) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        age_days = (time.time() - OUTPUT_PATH.stat().st_mtime) / 86400
        if age_days < 7:
            logger.info(
                "nse_all.csv is %.1f days old (< 7). Use --force to re-download.", age_days
            )
            return

    try:
        raw = fetch_nse_equity_list(NSE_EQUITY_URL)
    except requests.HTTPError as e:
        logger.error("HTTP error fetching NSE equity list: %s", e)
        sys.exit(1)
    except requests.RequestException as e:
        logger.error("Network error: %s", e)
        sys.exit(1)

    df = clean(raw)

    if df.empty:
        logger.error("Cleaned DataFrame is empty — aborting write.")
        sys.exit(1)

    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved %d stocks to %s", len(df), OUTPUT_PATH)

    # Print a quick summary
    print(f"\nNSE All-stocks universe: {len(df):,} EQ-series stocks")
    print(df[["nse_symbol", "company_name"]].head(10).to_string(index=False))
    print("...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if file is fresh")
    args = parser.parse_args()
    main(force=args.force)
