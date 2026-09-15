"""
Stock Data Fetcher
==================

Thin wrapper around :mod:`yfinance` for price history plus a small
:class:`TechnicalIndicators` collection (SMA, EMA, RSI, MACD, Bollinger).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn(
        "yfinance not installed. Stock fetching will be limited. "
        "Install with: pip install yfinance",
        stacklevel=2,
    )

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetch and persist stock market data via :mod:`yfinance`.

    Args:
        data_dir: Directory used to cache downloaded CSVs.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available")

    def get_price(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data for a ticker.

        Args:
            ticker: Ticker symbol, e.g. ``"AAPL"``.
            period: ``1d|5d|1mo|3mo|6mo|1y|2y|5y|10y|ytd|max``.
            interval: ``1m|2m|5m|15m|30m|60m|1h|1d|1wk|1mo``.

        Returns:
            OHLCV DataFrame indexed by date, or ``None`` on failure.
        """
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not available")
            return None

        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval)
        except Exception as exc:  # yfinance raises many bespoke subclasses
            logger.error("Error fetching %s: %s", ticker, exc)
            return None

        if df.empty:
            logger.warning("No data returned for %s", ticker)
            return None

        logger.info("Fetched %d rows for %s", len(df), ticker)
        return df

    def get_multiple_prices(
        self,
        tickers: List[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Get historical OHLCV data for several tickers in one round-trip.

        Returns:
            Mapping of ticker -> DataFrame. Tickers with no data are
            omitted from the result.
        """
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not available")
            return {}
        if not tickers:
            return {}

        try:
            data = yf.download(
                tickers,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                group_by="column",
            )
        except Exception as exc:
            logger.error("Error fetching %s: %s", tickers, exc)
            return {}

        results: Dict[str, pd.DataFrame] = {}
        if data.empty:
            logger.warning("No data returned for %s", tickers)
            return results

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = data
                else:
                    df = data.xs(ticker, axis=1, level=1)
                if not df.empty:
                    results[ticker] = df
            except (KeyError, ValueError) as exc:
                logger.warning("No data for %s: %s", ticker, exc)
        return results

    def get_info(self, ticker: str) -> Optional[Dict]:
        """Get company info dict for ``ticker``."""
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not available")
            return None
        try:
            info = yf.Ticker(ticker).info
        except Exception as exc:
            logger.error("Error fetching info for %s: %s", ticker, exc)
            return None
        if not info:
            logger.warning("No info returned for %s", ticker)
            return None
        logger.info("Fetched info for %s", ticker)
        return info

    def save_to_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """Persist a DataFrame to ``data_dir/filename``."""
        filepath = self.data_dir / filename
        try:
            df.to_csv(filepath)
            logger.info("Saved data to %s", filepath)
        except OSError as exc:
            logger.error("Error saving to CSV: %s", exc)
            raise
        return filepath

    def load_from_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a DataFrame previously written by :meth:`save_to_csv`."""
        filepath = self.data_dir / filename
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        except FileNotFoundError:
            logger.warning("File not found: %s", filepath)
            return None
        except (pd.errors.ParserError, OSError, UnicodeDecodeError) as exc:
            logger.error("Error loading CSV %s: %s", filepath, exc)
            return None
        logger.info("Loaded data from %s", filepath)
        return df

    def close(self) -> None:
        """No-op retained for API parity with other fetchers."""
        # yfinance uses a per-call session; nothing to clean up.
        return

    def __enter__(self) -> StockDataFetcher:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class TechnicalIndicators:
    """Pure-Pandas implementations of common technical indicators.

    All methods are static and accept a :class:`pandas.Series` of close
    prices (or any univariate numeric series).
    """

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=window, min_periods=window).mean()

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index using Wilder's smoothing.

        Uses an exponentially-weighted moving average (Wilder's alpha of
        ``1/period``) for the average gain/loss rather than a simple
        moving average — matching TradingView, TA-Lib, and most retail
        charting platforms.

        Convention: when ``avg_loss`` is zero (all gains), RSI is 100.
        When ``avg_gain`` is zero (all losses), RSI is 0.
        """
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        # Edge cases per the TradingView convention.
        rsi = rsi.where(avg_loss != 0, 100.0)
        rsi = rsi.where(avg_gain != 0, 0.0)
        return rsi

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence).

        Returns:
            ``(macd_line, signal_line, histogram)``
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        window: int = 20,
        num_std: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands.

        Returns:
            ``(upper, middle, lower)`` — middle is the SMA over ``window``.
        """
        middle = series.rolling(window=window, min_periods=window).mean()
        std = series.rolling(window=window, min_periods=window).std()
        upper = middle + std * num_std
        lower = middle - std * num_std
        return upper, middle, lower


if __name__ == "__main__":
    with StockDataFetcher() as fetcher:
        aapl = fetcher.get_price("AAPL")
        if aapl is not None:
            print(aapl.tail())
            close = aapl["Close"]
            print("\nSMA(20):", TechnicalIndicators.sma(close, 20).iloc[-1])
            print("RSI(14):", TechnicalIndicators.rsi(close).iloc[-1])