"""Tests for the reusable chart helpers.

The helpers don't import Streamlit, so we can exercise them with plain
pandas + plotly inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

plotly = pytest.importorskip("plotly")

from app.components import charts


@pytest.fixture
def holdings_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "company": ["Alpha", "Beta", "Gamma", "Delta"],
            "weight": [25.0, 20.0, 15.0, 10.0],
        }
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    n = 60
    base = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(
        {
            "Open": base + rng.normal(0, 0.5, n),
            "High": base + 1,
            "Low": base - 1,
            "Close": base,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["RSI_14"] = 50 + rng.normal(0, 10, n)
    return df


def test_holdings_bar_returns_figure(holdings_df: pd.DataFrame) -> None:
    fig = charts.holdings_bar(holdings_df, value_col="weight")
    assert fig is not None
    assert len(fig.data) == 1
    # Bar chart should be sorted ascending (so biggest on top when horizontal).
    xvals = list(fig.data[0].x)
    assert xvals == sorted(xvals)


def test_holdings_treemap_returns_figure(holdings_df: pd.DataFrame) -> None:
    fig = charts.holdings_treemap(holdings_df, value_col="weight")
    assert fig is not None
    assert len(fig.data) == 1


def test_price_chart_with_overlays(ohlcv: pd.DataFrame) -> None:
    fig = charts.price_chart(ohlcv, overlays=[("SMA_20", "SMA 20")], show_volume=True)
    # 1 candlestick + 1 SMA + 1 volume = 3 traces
    assert len(fig.data) == 3


def test_price_chart_without_volume(ohlcv: pd.DataFrame) -> None:
    fig = charts.price_chart(ohlcv, overlays=(), show_volume=False)
    assert len(fig.data) == 1


def test_indicator_subplot_skips_missing_columns(ohlcv: pd.DataFrame) -> None:
    fig = charts.indicator_subplot(ohlcv, ["RSI_14", "NOPE"], title="RSI")
    assert len(fig.data) == 1


def test_indicator_subplot_skips_all_nan(ohlcv: pd.DataFrame) -> None:
    ohlcv["blank"] = np.nan
    fig = charts.indicator_subplot(ohlcv, ["blank"])
    assert len(fig.data) == 0


def test_normalized_compare_rebases_to_100(ohlcv: pd.DataFrame) -> None:
    fig = charts.normalized_compare({"A": ohlcv, "B": ohlcv})
    assert len(fig.data) == 2
    # Both series should start at 100.
    for trace in fig.data:
        assert trace.y[0] == pytest.approx(100.0)


def test_normalized_compare_skips_empty() -> None:
    empty = pd.DataFrame(columns=["Close"])
    fig = charts.normalized_compare({"A": empty, "B": pd.DataFrame()})
    assert len(fig.data) == 0


def test_data_dir_creates_directory(tmp_path, monkeypatch) -> None:
    """``data_dir()`` must create the directory if it doesn't exist."""
    target = tmp_path / "fresh_data"
    monkeypatch.setenv("PYTHON_STOCKS_DATA_DIR", str(target))
    # Reset cached function (it's a plain function, but be explicit)
    import app.components.cache as cache_mod

    result = cache_mod.data_dir()
    assert result.exists()
    assert result.is_dir()


# --- last_fetch_error round-trip ---------------------------------------------


def test_last_fetch_error_stored_on_failure_and_cleared_on_success(tmp_path, monkeypatch) -> None:
    """``fetch_price`` stashes StockFetchError so the UI can show it; cleared on success."""
    import app.components.cache as cache_mod
    from python_stocks import StockFetchError

    monkeypatch.setenv("PYTHON_STOCKS_DATA_DIR", str(tmp_path))
    # Earlier tests in this module (and `test_pages_importable_with_only_app_on_path`)
    # can populate fetch_price's Streamlit cache via the default AAPL/1y/1d
    # call in 2_Stock_Analysis.main(); reset so the fake is actually invoked.
    cache_mod.fetch_price.clear()
    cache_mod._LAST_FETCH_ERROR.clear()

    boom = RuntimeError("rate limited")
    calls = {"n": 0}

    class FakeFetcher:
        def __init__(self, *a, **kw):
            pass

        def get_price(self, ticker, period="1y", interval="1d"):
            calls["n"] += 1
            if calls["n"] == 1:
                raise StockFetchError("AAPL", attempts=1, last_exc=boom, empty=False)
            import pandas as pd
            return pd.DataFrame(
                {"Close": [1.0]},
                index=pd.date_range("2024-01-01", periods=1),
            )

        def close(self):
            pass

    monkeypatch.setattr(cache_mod, "StockDataFetcher", FakeFetcher)

    key_fail = ("AAPL", "1y", "1d")
    key_ok = ("MSFT", "1y", "1d")
    assert cache_mod.last_fetch_error(*key_fail) is None

    # First call: fetcher raises; cache stores None + stashes the error.
    df = cache_mod.fetch_price(*key_fail)
    assert df is None
    err = cache_mod.last_fetch_error(*key_fail)
    assert err is not None
    assert err.ticker == "AAPL"
    assert err.last_exc is boom

    # Different key with the same fake: this call returns success.
    df2 = cache_mod.fetch_price(*key_ok)
    assert df2 is not None
    assert cache_mod.last_fetch_error(*key_ok) is None
    # The previous failure key is still tracked until that specific key succeeds.
    assert cache_mod.last_fetch_error(*key_fail) is not None


def test_fetch_price_stashes_error_when_get_price_returns_none(tmp_path, monkeypatch) -> None:
    """Older ``get_price`` returned None silently; we synthesize an error for the UI."""
    import app.components.cache as cache_mod

    monkeypatch.setenv("PYTHON_STOCKS_DATA_DIR", str(tmp_path))
    cache_mod.fetch_price.clear()
    cache_mod._LAST_FETCH_ERROR.clear()

    class OldStyleFetcher:
        def __init__(self, *a, **kw):
            pass

        def get_price(self, ticker, period="1y", interval="1d"):
            return None  # old behavior: silent None

        def close(self):
            pass

    monkeypatch.setattr(cache_mod, "StockDataFetcher", OldStyleFetcher)

    df = cache_mod.fetch_price("AAPL", "1y", "1d")
    assert df is None
    err = cache_mod.last_fetch_error("AAPL", "1y", "1d")
    assert err is not None
    assert err.ticker == "AAPL"
    assert err.empty is True
    assert err.last_exc is None


def test_fetch_price_catches_plain_exception(tmp_path, monkeypatch) -> None:
    """If get_price raises something other than StockFetchError, we still stash it."""
    import app.components.cache as cache_mod

    monkeypatch.setenv("PYTHON_STOCKS_DATA_DIR", str(tmp_path))
    cache_mod.fetch_price.clear()
    cache_mod._LAST_FETCH_ERROR.clear()

    boom = ValueError("weird data shape")

    class WeirdFetcher:
        def __init__(self, *a, **kw):
            pass

        def get_price(self, ticker, period="1y", interval="1d"):
            raise boom

        def close(self):
            pass

    monkeypatch.setattr(cache_mod, "StockDataFetcher", WeirdFetcher)

    df = cache_mod.fetch_price("AAPL", "1y", "1d")
    assert df is None
    err = cache_mod.last_fetch_error("AAPL", "1y", "1d")
    # Stash whatever we got; UI duck-types on .attempts / .empty / .last_exc.
    assert err is boom


def test_cache_module_loads_when_stock_fetch_error_missing(tmp_path, monkeypatch) -> None:
    """Module import must not fail if python_stocks.stock_data is stale (no StockFetchError).

    Reproduces the Streamlit Cloud ImportError where a partial deploy leaves the
    submodule without the new class.
    """
    monkeypatch.setenv("PYTHON_STOCKS_DATA_DIR", str(tmp_path))
    # Simulate a stale submodule by reloading cache.py with StockFetchError blocked.
    import importlib
    import sys

    # Hide StockFetchError on the submodule so the import falls back.
    import python_stocks.stock_data as sd

    real_has = hasattr(sd, "StockFetchError")
    if real_has:
        # Temporarily pop the class so the ``except ImportError`` branch triggers.
        sentinel = sd.StockFetchError
        delattr(sd, "StockFetchError")
        try:
            sys.modules.pop("app.components.cache", None)
            cache_mod = importlib.import_module("app.components.cache")
            # The local fallback must exist and have the same shape.
            assert hasattr(cache_mod, "StockFetchError")
            err = cache_mod.StockFetchError("AAPL", attempts=2)
            assert err.ticker == "AAPL"
            assert err.attempts == 2
        finally:
            sd.StockFetchError = sentinel
    else:
        # Already absent in this environment; the fallback path is exercised.
        sys.modules.pop("app.components.cache", None)
        cache_mod = importlib.import_module("app.components.cache")
        assert hasattr(cache_mod, "StockFetchError")


# --- Holdings chart label resolution -----------------------------------------


def _trace_labels(fig) -> list[str]:
    """Extract the y-categories or treemap labels from a Plotly figure."""
    trace = fig.data[0]
    # px.bar(orientation='h') sets y as categorical.
    if hasattr(trace, "y") and trace.y is not None:
        return list(trace.y)
    # px.treemap sets labels as the leaf ids.
    if hasattr(trace, "labels"):
        return list(trace.labels)
    return []


def test_holdings_bar_prefers_ticker_column() -> None:
    """When both ticker and company are present, the chart uses the ticker."""
    df = pd.DataFrame(
        {
            "ticker": ["TSLA", "COIN"],
            "company": ["Tesla Inc", "Coinbase"],
            "weight": [12.5, 7.5],
        }
    )
    fig = charts.holdings_bar(df, value_col="weight")
    labels = _trace_labels(fig)
    assert "TSLA" in labels
    assert "COIN" in labels
    assert "Tesla Inc" not in labels
    assert "Coinbase" not in labels


def test_holdings_treemap_prefers_ticker_column() -> None:
    """Same behavior for the treemap variant."""
    df = pd.DataFrame(
        {
            "ticker": ["TSLA", "COIN"],
            "company": ["Tesla Inc", "Coinbase"],
            "weight": [12.5, 7.5],
        }
    )
    fig = charts.holdings_treemap(df, value_col="weight")
    labels = _trace_labels(fig)
    assert "TSLA" in labels
    assert "COIN" in labels
    assert "Tesla Inc" not in labels


def test_holdings_bar_falls_back_to_company() -> None:
    """If no ticker column is present, fall back to a name column."""
    df = pd.DataFrame(
        {
            "company": ["Tesla Inc", "Coinbase"],
            "shares": [1000, 500],
            "weight": [12.5, 7.5],
        }
    )
    fig = charts.holdings_bar(df, value_col="weight")
    labels = _trace_labels(fig)
    assert "Tesla Inc" in labels
    assert "Coinbase" in labels


def test_holdings_bar_accepts_symbol_column() -> None:
    """``symbol`` is the second ticker candidate after ``ticker``."""
    df = pd.DataFrame(
        {
            "symbol": ["TSLA", "COIN"],
            "weight": [12.5, 7.5],
        }
    )
    fig = charts.holdings_bar(df, value_col="weight")
    labels = _trace_labels(fig)
    assert "TSLA" in labels
    assert "COIN" in labels


def test_holdings_bar_honors_explicit_name_col() -> None:
    """An explicit ``name_col`` always wins over the candidates."""
    df = pd.DataFrame(
        {
            "ticker": ["TSLA", "COIN"],
            "company": ["Tesla Inc", "Coinbase"],
            "weight": [12.5, 7.5],
        }
    )
    fig = charts.holdings_bar(df, value_col="weight", name_col="company")
    labels = _trace_labels(fig)
    assert "Tesla Inc" in labels
    assert "Coinbase" in labels
    assert "TSLA" not in labels


def test_holdings_bar_sums_duplicate_tickers() -> None:
    """Multiple rows with the same ticker are summed into one bar.

    ARK CSVs include duplicate tickers for share classes and warrants;
    we must aggregate so the bar chart has one entry per ticker.
    """
    df = pd.DataFrame(
        {
            "ticker": ["TSLA", "TSLA", "COIN"],
            "weight": [5.0, 7.5, 12.0],
        }
    )
    fig = charts.holdings_bar(df, value_col="weight")
    labels = _trace_labels(fig)
    # Duplicate is collapsed.
    assert labels.count("TSLA") == 1
    # x values should be the aggregated weight (sum of the two TSLA rows).
    xs = list(fig.data[0].x)
    assert pytest.approx(12.5) in xs
    assert pytest.approx(12.0) in xs


def test_holdings_treemap_drops_nan_tickers() -> None:
    """Rows with NaN tickers (private placements, warrants) are dropped.

    Otherwise Plotly treats them as a non-leaf internal node and raises.
    """
    df = pd.DataFrame(
        {
            "ticker": ["TSLA", None, "COIN"],
            "weight": [10.0, 1.0, 12.0],
        }
    )
    fig = charts.holdings_treemap(df, value_col="weight")
    labels = _trace_labels(fig)
    assert "TSLA" in labels
    assert "COIN" in labels
    assert None not in labels
    assert "" not in labels