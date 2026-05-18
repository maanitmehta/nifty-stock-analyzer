"""Parquet-backed cache for price and fundamental data."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from ..config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Read/write parquet files under a named subdirectory of the cache root.

    Keys are arbitrary strings (typically ticker + lookback). Special characters
    that are invalid in filenames are sanitised automatically.
    """

    def __init__(self, subdir: str = "prices") -> None:
        self._dir = settings.cache_dir / subdir
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, key: str) -> pd.DataFrame | None:
        """Return cached DataFrame, or None if the key is not in cache."""
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Corrupt cache file %s, will re-fetch: %s", path, exc)
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, df: pd.DataFrame) -> None:
        """Persist a DataFrame to cache."""
        df.to_parquet(self._path(key), index=True)
        logger.debug("Cached %d rows → %s", len(df), self._path(key).name)

    def is_stale(self, key: str, max_age_hours: int | None = None) -> bool:
        """Return True if the cached file is absent or older than max_age_hours."""
        path = self._path(key)
        if not path.exists():
            return True
        threshold = max_age_hours if max_age_hours is not None else settings.cache_max_age_hours
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        return age_hours > threshold

    def invalidate(self, key: str) -> None:
        """Delete a cached file if it exists."""
        self._path(key).unlink(missing_ok=True)

    def invalidate_all(self) -> None:
        """Delete every file in this cache subdirectory."""
        for f in self._dir.glob("*.parquet"):
            f.unlink()
        logger.info("Invalidated all files in %s", self._dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, key: str) -> Path:
        # Replace characters that are problematic in filenames
        safe = key.replace("^", "IDX_").replace("/", "_").replace("\\", "_").replace(":", "_")
        return self._dir / f"{safe}.parquet"
