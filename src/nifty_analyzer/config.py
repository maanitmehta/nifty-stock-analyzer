from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = three levels up from this file (src/nifty_analyzer/config.py)
_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths — use home dir so it works both locally and on Streamlit Cloud
    cache_dir: Path = Path.home() / ".nifty_analyzer" / "cache"
    universe_dir: Path = _ROOT / "data" / "universe"

    # Cache behaviour
    cache_max_age_hours: int = 24

    # Finance constants
    risk_free_rate: float = 0.065  # RBI 91-day T-bill ~6.5% annualized
    benchmark_ticker: str = "^NSEI"

    # Default lookback window in years (1, 3, or 5)
    default_lookback_years: int = 3

    # NSE equity list URL (full exchange universe)
    nse_equity_list_url: str = (
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    )


settings = Settings()
