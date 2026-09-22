"""
Stock Data Fetcher
==================

Thin wrapper around :mod:`yfinance` for price history plus a small
:class:`TechnicalIndicators` collection (SMA, EMA, RSI, MACD, Bollinger).
"""

from __future__ import annotations

import logging
import time
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

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0


class StockFetchError(Exception):
    """Raised when yfinance cannot return data after all retries.

    Carries enough context for the UI to surface the real failure
    mode (rate-limit vs empty response vs network error) without
    forcing callers to wrap every call in ``try/except``.
    """

    def __init__(
        self,
        ticker: str,
        attempts: int,
        last_exc: Optional[BaseException] = None,
        empty: bool = False,
    ) -> None:
        if empty and last_exc is None:
            msg = (
                f"yfinance returned an empty response for {ticker!r} "
                f"after {attempts} attempt(s). The ticker may be invalid "
                f"or the requested range/interval may be unsupported."
            )
        elif last_exc is not None:
            msg = (
                f"yfinance failed for {ticker!r} after {attempts} attempt(s); "
                f"last error: {type(last_exc).__name__}: {last_exc}"
            )
        else:
            msg = f"yfinance failed for {ticker!r} after {attempts} attempt(s)"
        super().__init__(msg)
        self.ticker = ticker
        self.attempts = attempts
        self.last_exc = last_exc
        self.empty = empty


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
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data for a ticker.

        Args:
            ticker: Ticker symbol, e.g. ``"AAPL"``.
            period: ``1d|5d|1mo|3mo|6mo|1y|2y|5y|10y|ytd|max``.
            interval: ``1m|2m|5m|15m|30m|60m|1h|1d|1wk|1mo``.
            max_retries: Number of attempts on transient yfinance errors
                (network issues, rate limits). Set to ``1`` to disable.
            retry_delay: Initial backoff in seconds; doubled each retry.

        Returns:
            OHLCV DataFrame indexed by date, or ``None`` on failure.

        Raises:
            StockFetchError: When yfinance cannot return data after the
                retry budget is exhausted. The exception carries the
                ticker, attempt count, last underlying exception (if any),
                and whether every attempt returned an empty DataFrame.
                ``yfinance`` not being installed also raises.
        """
        if not YFINANCE_AVAILABLE:
            raise StockFetchError(ticker, attempts=0, last_exc=None, empty=False)

        last_exc: Optional[BaseException] = None
        for attempt in range(max_retries):
            try:
                df = yf.Ticker(ticker).history(period=period, interval=interval)
            except Exception as exc:  # yfinance raises many bespoke subclasses
                last_exc = exc
                logger.warning(
                    "Fetch attempt %d/%d for %s failed: %s",
                    attempt + 1, max_retries, ticker, exc,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                continue

            if df.empty:
                last_exc = None  # empty is not an exception
                logger.warning(
                    "No data returned for %s (attempt %d/%d)",
                    ticker, attempt + 1, max_retries,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                continue

            if attempt > 0:
                logger.info(
                    "Fetched %d rows for %s on attempt %d",
                    len(df), ticker, attempt + 1,
                )
            else:
                logger.info("Fetched %d rows for %s", len(df), ticker)
            return df

        if last_exc is not None:
            logger.error(
                "Giving up on %s after %d attempts; last error: %s",
                ticker, max_retries, last_exc,
            )
            raise StockFetchError(
                ticker, attempts=max_retries, last_exc=last_exc, empty=False
            )
        logger.error("Giving up on %s after %d empty attempts", ticker, max_retries)
        raise StockFetchError(
            ticker, attempts=max_retries, last_exc=None, empty=True
        )

    def get_multiple_prices(
        self,
        tickers: List[str],
        period: str = "1y",
        interval: str = "1d",
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
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

        last_exc: Optional[BaseException] = None
        data = pd.DataFrame()
        for attempt in range(max_retries):
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
                last_exc = exc
                logger.warning(
                    "Batch fetch attempt %d/%d for %s failed: %s",
                    attempt + 1, max_retries, tickers, exc,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                continue

            if not data.empty:
                break

            logger.warning(
                "Batch fetch attempt %d/%d for %s returned empty",
                attempt + 1, max_retries, tickers,
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))

        results: Dict[str, pd.DataFrame] = {}
        if data.empty:
            if last_exc is not None:
                logger.error(
                    "Giving up on %s after %d attempts; last error: %s",
                    tickers, max_retries, last_exc,
                )
            else:
                logger.warning("No data returned for %s", tickers)
            return results

        for ticker in tickers:
            try:
                # yfinance returns a MultiIndex even for a single ticker
                # under group_by="column"; xs() unifies both cases.
                df = data.xs(ticker, axis=1, level=1)
            except (KeyError, ValueError) as exc:
                logger.warning("No data for %s: %s", ticker, exc)
                continue
            if not df.empty:
                results[ticker] = df
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
        try:
            aapl = fetcher.get_price("AAPL")
        except StockFetchError as exc:
            print(f"Failed to fetch AAPL: {exc}")
            raise SystemExit(1)
        print(aapl.tail())
        close = aapl["Close"]
        print("\nSMA(20):", TechnicalIndicators.sma(close, 20).iloc[-1])
        print("RSI(14):", TechnicalIndicators.rsi(close).iloc[-1])