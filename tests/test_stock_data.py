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


# --- get_price retry behavior ------------------------------------------------


def _sample_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": range(n), "High": range(n), "Low": range(n), "Close": range(n), "Volume": range(n)},
        index=pd.date_range("2024-01-01", periods=n),
    )


def test_get_price_retries_on_exception(tmp_path, monkeypatch) -> None:
    """First call raises, second succeeds — get_price should return data."""
    import python_stocks.stock_data as mod

    fetcher = StockDataFetcher(data_dir=tmp_path)
    calls = {"n": 0}

    class FakeTicker:
        def history(self, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient network error")
            return _sample_df()

    monkeypatch.setattr(mod, "yf", type("M", (), {"Ticker": lambda _self, _t: FakeTicker()})())
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)  # no real sleep in tests

    df = fetcher.get_price("AAPL", max_retries=3, retry_delay=0.0)
    assert df is not None
    assert len(df) == 5
    assert calls["n"] == 2
    fetcher.close()


def test_get_price_retries_on_empty_response(tmp_path, monkeypatch) -> None:
    """Empty results are also retried (e.g., yfinance rate-limit returning empty)."""
    import python_stocks.stock_data as mod

    fetcher = StockDataFetcher(data_dir=tmp_path)
    calls = {"n": 0}

    class FakeTicker:
        def history(self, **_kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                return pd.DataFrame()  # empty
            return _sample_df()

    monkeypatch.setattr(mod, "yf", type("M", (), {"Ticker": lambda _self, _t: FakeTicker()})())
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

    df = fetcher.get_price("AAPL", max_retries=3, retry_delay=0.0)
    assert df is not None
    assert calls["n"] == 3
    fetcher.close()


def test_get_price_returns_none_when_all_attempts_fail(tmp_path, monkeypatch) -> None:
    """All retries exhausted -> None, no crash."""
    import python_stocks.stock_data as mod

    fetcher = StockDataFetcher(data_dir=tmp_path)

    class FakeTicker:
        def history(self, **_kwargs):
            raise RuntimeError("upstream down")

    monkeypatch.setattr(mod, "yf", type("M", (), {"Ticker": lambda _self, _t: FakeTicker()})())
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

    df = fetcher.get_price("AAPL", max_retries=2, retry_delay=0.0)
    assert df is None
    fetcher.close()


def test_get_price_max_retries_one_disables_retry(tmp_path, monkeypatch) -> None:
    """Passing max_retries=1 means no retries on the first failure."""
    import python_stocks.stock_data as mod

    fetcher = StockDataFetcher(data_dir=tmp_path)
    calls = {"n": 0}

    class FakeTicker:
        def history(self, **_kwargs):
            calls["n"] += 1
            raise RuntimeError("nope")

    monkeypatch.setattr(mod, "yf", type("M", (), {"Ticker": lambda _self, _t: FakeTicker()})())

    df = fetcher.get_price("AAPL", max_retries=1)
    assert df is None
    assert calls["n"] == 1
    fetcher.close()


# --- get_multiple_prices single-ticker column bug ----------------------------


def test_get_multiple_prices_single_ticker_has_flat_columns(tmp_path, monkeypatch) -> None:
    """yfinance returns MultiIndex columns even for a single ticker; we must xs().

    Regression test for the case where the Compare page is used with one
    ticker — previously the returned DataFrame had tuple columns like
    ('Close', 'AAPL') instead of 'Close'.
    """
    import python_stocks.stock_data as mod

    fetcher = StockDataFetcher(data_dir=tmp_path)
    columns = pd.MultiIndex.from_tuples(
        [
            ("Close", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Open", "AAPL"),
            ("Volume", "AAPL"),
        ]
    )
    fake = pd.DataFrame(
        [[1, 2, 1, 1, 100]],
        columns=columns,
        index=pd.date_range("2024-01-01", periods=1),
    )

    monkeypatch.setattr(
        mod,
        "yf",
        type(
            "M",
            (),
            {
                "download": lambda *_a, **_k: fake,
            },
        )(),
    )

    out = fetcher.get_multiple_prices(["AAPL"])
    assert "AAPL" in out
    df = out["AAPL"]
    assert list(df.columns) == ["Close", "High", "Low", "Open", "Volume"]
    fetcher.close()


def test_get_multiple_prices_multi_ticker_has_flat_columns(tmp_path, monkeypatch) -> None:
    """Multi-ticker path: each per-ticker slice still has flat columns."""
    import python_stocks.stock_data as mod

    fetcher = StockDataFetcher(data_dir=tmp_path)
    columns = pd.MultiIndex.from_tuples(
        [
            ("Close", "AAPL"),
            ("Close", "MSFT"),
            ("High", "AAPL"),
            ("High", "MSFT"),
            ("Low", "AAPL"),
            ("Low", "MSFT"),
            ("Open", "AAPL"),
            ("Open", "MSFT"),
            ("Volume", "AAPL"),
            ("Volume", "MSFT"),
        ]
    )
    fake = pd.DataFrame(
        [[1, 10, 2, 20, 1, 10, 1, 10, 100, 200]],
        columns=columns,
        index=pd.date_range("2024-01-01", periods=1),
    )

    monkeypatch.setattr(
        mod,
        "yf",
        type("M", (), {"download": lambda *_a, **_k: fake})(),
    )

    out = fetcher.get_multiple_prices(["AAPL", "MSFT"])
    assert set(out) == {"AAPL", "MSFT"}
    for df in out.values():
        assert list(df.columns) == ["Close", "High", "Low", "Open", "Volume"]
    fetcher.close()


def test_get_multiple_prices_retries_on_exception(tmp_path, monkeypatch) -> None:
    """Batch fetcher should also retry on transient errors."""
    import python_stocks.stock_data as mod

    fetcher = StockDataFetcher(data_dir=tmp_path)
    calls = {"n": 0}

    def fake_download(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        columns = pd.MultiIndex.from_tuples(
            [("Close", "AAPL"), ("High", "AAPL"), ("Low", "AAPL"), ("Open", "AAPL"), ("Volume", "AAPL")]
        )
        return pd.DataFrame([[1, 2, 1, 1, 100]], columns=columns, index=pd.date_range("2024-01-01", periods=1))

    monkeypatch.setattr(mod, "yf", type("M", (), {"download": fake_download})())
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

    out = fetcher.get_multiple_prices(["AAPL"], max_retries=3, retry_delay=0.0)
    assert "AAPL" in out
    assert calls["n"] == 2
    fetcher.close()
