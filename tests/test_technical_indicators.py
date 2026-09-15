"""Tests for the TechnicalIndicators class."""

from __future__ import annotations

import numpy as np
import pandas as pd

from python_stocks import TechnicalIndicators


def test_sma_window(close_series: pd.Series) -> None:
    out = TechnicalIndicators.sma(close_series, 20)
    assert len(out) == len(close_series)
    assert out.iloc[:19].isna().all()
    assert not out.iloc[19:].isna().any()
    assert np.isclose(out.iloc[19], close_series.iloc[:20].mean())


def test_ema_window(close_series: pd.Series) -> None:
    out = TechnicalIndicators.ema(close_series, 20)
    assert len(out) == len(close_series)
    assert not out.isna().any()


def test_rsi_wilder_known_value() -> None:
    """RSI on a monotonic uptrend should be 100 (all gains)."""
    prices = pd.Series(np.arange(1.0, 30.0))
    rsi = TechnicalIndicators.rsi(prices, period=14)
    assert rsi.iloc[-1] == 100.0


def test_rsi_monotonic_downtrend() -> None:
    """RSI on a monotonic downtrend should be 0 (all losses)."""
    prices = pd.Series(np.arange(30.0, 0.0, -1.0))
    rsi = TechnicalIndicators.rsi(prices, period=14)
    assert rsi.iloc[-1] == 0.0


def test_rsi_wilder_not_sma(close_series: pd.Series) -> None:
    """Wilder's RSI must differ from the SMA-based textbook variant."""
    rsi_wilder = TechnicalIndicators.rsi(close_series, period=14)
    delta = close_series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_sma = 100 - (100 / (1 + rs))
    assert not np.allclose(rsi_wilder.dropna().values, rsi_sma.dropna().values)


def test_macd_returns_three_series(close_series: pd.Series) -> None:
    macd, signal, hist = TechnicalIndicators.macd(close_series)
    assert isinstance(macd, pd.Series)
    assert isinstance(signal, pd.Series)
    assert isinstance(hist, pd.Series)
    assert np.allclose((macd - signal).dropna().values, hist.dropna().values)


def test_bollinger_bands_order(close_series: pd.Series) -> None:
    """Order is (upper, middle, lower)."""
    upper, middle, lower = TechnicalIndicators.bollinger_bands(close_series)
    valid = upper.notna() & middle.notna() & lower.notna()
    assert (upper[valid] >= middle[valid]).all()
    assert (middle[valid] >= lower[valid]).all()