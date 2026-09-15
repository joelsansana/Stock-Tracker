"""Shared helpers used by every page."""

from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

from python_stocks import (
    SentimentAnalyzer,
    StockSentimentFetcher,
    __version__,
)
from python_stocks.ark_fetcher import ARK_ETF_URLS


def package_version() -> str:
    return __version__


def available_etfs() -> list[str]:
    return list(ARK_ETF_URLS.keys())


def backend_status() -> dict[str, bool]:
    """One-shot detection of which optional backends are installed."""
    import importlib.util

    status = {
        "yfinance": importlib.util.find_spec("yfinance") is not None,
        "textblob": importlib.util.find_spec("textblob") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "tweepy": importlib.util.find_spec("tweepy") is not None,
    }
    try:
        analyzer = SentimentAnalyzer()
        status["sentiment_backend"] = analyzer.backend != "none"
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.debug("sentiment backend detection failed: %s", exc)
        status["sentiment_backend"] = False
    return status


def twitter_configured() -> Optional[str]:
    """Return the bearer token if one is set, otherwise ``None``."""
    try:
        token = st.secrets.get("TWEET_BEARER_TOKEN")
    except (FileNotFoundError, KeyError):
        token = None
    if token:
        return token
    import os

    return os.environ.get("TWEET_BEARER_TOKEN") or None


def twitter_fetcher() -> Optional[StockSentimentFetcher]:
    """Build a :class:`StockSentimentFetcher` if credentials exist."""
    token = twitter_configured()
    if not token:
        return None
    return StockSentimentFetcher(bearer_token=token)


def sidebar_nav() -> None:
    """Render a small sidebar nav. Streamlit does this automatically
    when ``pages/`` exists, but we add a tiny info block too."""
    st.sidebar.markdown(f"**python_stocks v{package_version()}**")
    st.sidebar.caption("Stock analysis toolkit")