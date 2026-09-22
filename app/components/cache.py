"""Shared caching helpers for the Streamlit UI.

Wraps the existing ``python_stocks`` fetchers with Streamlit's cache
decorators so that:
- ARK CSV downloads are cached for 1 hour on disk + in memory.
- yfinance downloads are cached for 5 minutes (short period) or 1 hour.
- HF pipeline (if used later) is cached for the process lifetime.

The module imports Streamlit lazily so it remains importable in tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import streamlit as st

from python_stocks import ARKDataFetcher, StockDataFetcher

ARK_TTL_SECONDS = 60 * 60  # 1 hour
PRICE_TTL_SHORT = 60 * 5    # 5 min for intraday
PRICE_TTL_LONG = 60 * 60    # 1 hour otherwise

_SHORT_PERIODS = {"1d", "5d", "1mo"}


def data_dir() -> Path:
    """Resolve the data directory from secrets/env or fall back to ./data."""
    import os

    candidates = []
    try:
        candidates.append(st.secrets.get("DATA_DIR"))
    except (FileNotFoundError, KeyError):
        # No .streamlit/secrets.toml, or it has no DATA_DIR key.
        pass
    candidates.append(os.environ.get("PYTHON_STOCKS_DATA_DIR"))
    for c in candidates:
        if c:
            p = Path(c)
            p.mkdir(parents=True, exist_ok=True)
            return p
    p = Path("data")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ttl_for_period(period: str) -> int:
    return PRICE_TTL_SHORT if period in _SHORT_PERIODS else PRICE_TTL_LONG


@st.cache_data(ttl=ARK_TTL_SECONDS, show_spinner="Fetching ARK holdings…")
def fetch_ark(ticker: str, force_refresh: bool) -> Any:
    """Cached wrapper around :meth:`ARKDataFetcher.get_holding`."""
    fetcher = ARKDataFetcher(data_dir=data_dir())
    try:
        return fetcher.get_holding(ticker, force_refresh=force_refresh)
    finally:
        fetcher.close()


@st.cache_data(ttl=PRICE_TTL_LONG, show_spinner="Fetching price data…")
def fetch_price(ticker: str, period: str, interval: str, force_refresh: bool = False) -> Any:
    """Cached wrapper around :meth:`StockDataFetcher.get_price`.

    Streamlit's ``@st.cache_data`` TTL is fixed at decoration time, so
    we use a single one-hour TTL regardless of period. ``force_refresh``
    is part of the cache key, so flipping it lets the page bypass any
    cached ``None`` from a previous failed fetch.
    """
    fetcher = StockDataFetcher(data_dir=data_dir())
    try:
        return fetcher.get_price(ticker, period=period, interval=interval)
    finally:
        fetcher.close()


def ttl_for(period: str) -> int:
    """Expose the TTL selection so pages can ``st.cache_data(ttl=…)``."""
    return _ttl_for_period(period)


def clear_caches() -> Callable[[], None]:
    """Return a callable that clears all Streamlit caches."""
    return st.cache_data.clear