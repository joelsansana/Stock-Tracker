"""Shared caching helpers for the Streamlit UI.

Wraps the existing ``python_stocks`` fetchers with Streamlit's cache
decorators so that:
- ARK CSV downloads are cached for 1 hour on disk + in memory.
- yfinance downloads are cached for 5 minutes (short period) or 1 hour.
- HF pipeline (if used later) is cached for the process lifetime.

The module imports Streamlit lazily so it remains importable in tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import streamlit as st

# Import from submodules rather than the package root. Streamlit Cloud
# has shown a tendency to cache the package's __init__.py bytecode
# across rebuilds, which can leave freshly-added top-level exports
# invisible for a deploy cycle.
from python_stocks.ark_fetcher import ARKDataFetcher
from python_stocks.stock_data import StockDataFetcher

try:
    from python_stocks.stock_data import StockFetchError

    _STOCK_FETCH_ERROR_SOURCE = "python_stocks.stock_data.StockFetchError"
except ImportError:
    # Streamlit Cloud may also be running a deploy where the *submodule*
    # itself is stale (the previous commit introduced this class). Use a
    # local fallback so this module still loads; UI code that cares
    # about the rich attributes will fall back to ``str(exc)``.
    class StockFetchError(Exception):  # type: ignore[no-redef]
        """Fallback used when the deployed ``python_stocks`` is stale."""

        def __init__(self, ticker: str = "?", attempts: int = 0,
                     last_exc: Optional[BaseException] = None, empty: bool = False) -> None:
            msg = (
                f"yfinance fetch failed for {ticker!r} "
                f"(attempts={attempts}, empty={empty}, "
                f"last_error={type(last_exc).__name__ if last_exc else '—'})"
            )
            super().__init__(msg)
            self.ticker = ticker
            self.attempts = attempts
            self.last_exc = last_exc
            self.empty = empty

    _STOCK_FETCH_ERROR_SOURCE = "local fallback (deployed python_stocks is stale)"

logger = logging.getLogger(__name__)

ARK_TTL_SECONDS = 60 * 60  # 1 hour
PRICE_TTL_SHORT = 60 * 5    # 5 min for intraday
PRICE_TTL_LONG = 60 * 60    # 1 hour otherwise

_SHORT_PERIODS = {"1d", "5d", "1mo"}

# Last fetch error per (ticker, period, interval) so the UI can surface
# the underlying yfinance failure instead of a generic "no data" message.
# Cleared on the next successful fetch for the same key.
_LAST_FETCH_ERROR: Dict[Tuple[str, str, str], BaseException] = {}


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
def fetch_price(ticker: str, period: str, interval: str) -> Any:
    """Cached wrapper around :meth:`StockDataFetcher.get_price`.

    Streamlit's ``@st.cache_data`` TTL is fixed at decoration time, so
    we use a single one-hour TTL regardless of period. To bypass the
    cache (e.g. after a transient upstream failure), call
    :func:`fetch_price.clear` before invoking this function.

    On failure (whether the new ``StockFetchError`` or any other
    exception, including the older "returns ``None`` silently" behavior),
    the underlying exception is stashed via :func:`last_fetch_error` and
    ``None`` is returned so the page keeps its existing "no data" UX
    while still surfacing the detail.
    """
    fetcher = StockDataFetcher(data_dir=data_dir())
    key = (ticker, period, interval)
    try:
        df = fetcher.get_price(ticker, period=period, interval=interval)
    except BaseException as exc:  # noqa: BLE001 — stash anything yfinance raises
        _LAST_FETCH_ERROR[key] = exc
        logger.warning(
            "fetch_price(%s, %s, %s) raised %s: %s",
            ticker, period, interval, type(exc).__name__, exc,
        )
        return None
    finally:
        fetcher.close()

    if df is None:
        # Older ``get_price`` returns ``None`` instead of raising. Synthesize
        # an error so the UI can still show something useful.
        synthetic = StockFetchError(ticker=ticker, attempts=0, last_exc=None, empty=True)
        _LAST_FETCH_ERROR[key] = synthetic
        logger.warning("fetch_price(%s, %s, %s) returned None", ticker, period, interval)
        return None

    _LAST_FETCH_ERROR.pop(key, None)
    return df


def last_fetch_error(ticker: str, period: str, interval: str) -> Optional[BaseException]:
    """Return the most recent fetch exception for a key, if any.

    The exception type depends on what the deployed ``python_stocks``
    raised — typically :class:`StockFetchError`, but possibly a plain
    ``Exception`` if the package is stale. UI code should duck-type on
    ``.attempts`` / ``.empty`` / ``.last_exc`` rather than relying on a
    specific class.
    """
    return _LAST_FETCH_ERROR.get((ticker, period, interval))


def ttl_for(period: str) -> int:
    """Expose the TTL selection so pages can ``st.cache_data(ttl=…)``."""
    return _ttl_for_period(period)


def clear_caches() -> Callable[[], None]:
    """Return a callable that clears all Streamlit caches."""
    return st.cache_data.clear
