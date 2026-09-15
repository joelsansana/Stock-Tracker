"""Tests for StockDataFetcher (no live network calls)."""

from __future__ import annotations

import pandas as pd
import pytest

from python_stocks import StockDataFetcher


def test_close_is_callable(tmp_path) -> None:
    """The README shows .close(); make sure it exists and doesn't blow up."""
    fetcher = StockDataFetcher(data_dir=tmp_path)
    assert hasattr(fetcher, "close")
    fetcher.close()


def test_context_manager(tmp_path) -> None:
    with StockDataFetcher(data_dir=tmp_path) as f:
        assert isinstance(f, StockDataFetcher)


def test_save_load_roundtrip(tmp_path) -> None:
    fetcher = StockDataFetcher(data_dir=tmp_path)
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3))
    path = fetcher.save_to_csv(df, "roundtrip.csv")
    assert path.exists()
    loaded = fetcher.load_from_csv("roundtrip.csv")
    assert loaded is not None
    assert loaded["Close"].tolist() == [1.0, 2.0, 3.0]


def test_load_missing_returns_none(tmp_path) -> None:
    fetcher = StockDataFetcher(data_dir=tmp_path)
    assert fetcher.load_from_csv("does_not_exist.csv") is None


def test_get_multiple_prices_empty_list_returns_empty(tmp_path) -> None:
    fetcher = StockDataFetcher(data_dir=tmp_path)
    # Skip if yfinance not installed; this is a behavioral test either way.
    if not fetcher.YFINANCE_AVAILABLE if hasattr(fetcher, "YFINANCE_AVAILABLE") else True:
        pytest.skip("yfinance not available")
    assert fetcher.get_multiple_prices([]) == {}