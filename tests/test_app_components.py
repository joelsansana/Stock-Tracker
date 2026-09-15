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