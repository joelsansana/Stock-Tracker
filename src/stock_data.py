"""
Stock Data Fetcher
Utilities for fetching and processing stock market data.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd

# Try to import yfinance, make optional
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not installed. Stock fetching will be limited.")

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetch and process stock market data."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize stock data fetcher.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available. Install with: pip install yfinance")
    
    def get_price(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d, 1wk, 1mo)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not available")
            return None
            
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
            
            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None
    
    def get_multiple_prices(
        self,
        tickers: List[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical prices for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not available")
            return {}
        
        try:
            # Download all at once (more efficient)
            data = yf.download(tickers, period=period, interval=interval, progress=False)
            
            results = {}
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        df = data
                    else:
                        df = data['Close'][ticker] if 'Close' in data.columns else pd.DataFrame()
                    
                    if not df.empty:
                        results[ticker] = df
                except KeyError:
                    logger.warning(f"No data for {ticker}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching multiple tickers: {e}")
            return {}
    
    def get_info(self, ticker: str) -> Optional[Dict]:
        """
        Get company info for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company info or None if failed
        """
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not available")
            return None
            
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            if not info:
                logger.warning(f"No info returned for {ticker}")
                return None
            
            logger.info(f"Fetched info for {ticker}")
            return info
            
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {e}")
            return None
    
    def save_to_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save DataFrame to CSV.
        
        Args:
            df: DataFrame to save
            filename: Filename (without path)
            
        Returns:
            Path to saved file
        """
        filepath = self.data_dir / filename
        try:
            df.to_csv(filepath)
            logger.info(f"Saved data to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            raise
    
    def load_from_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from CSV.
        
        Args:
            filename: Filename (without path)
            
        Returns:
            DataFrame or None if failed
        """
        filepath = self.data_dir / filename
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Loaded data from {filepath}")
            return df
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
        return None


class TechnicalIndicators:
    """Calculate technical indicators for stocks."""
    
    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=window).mean()
    
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """
        MACD (Moving Average Convergence Divergence).
        
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
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
        num_std: float = 2.0
    ) -> tuple:
        """Bollinger Bands."""
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return upper, sma, lower


if __name__ == "__main__":
    # Example usage
    fetcher = StockDataFetcher()
    
    # Get Apple data
    aapl = fetcher.get_price("AAPL")
    if aapl is not None:
        print(aapl.tail())
        
        # Calculate indicators
        close = aapl['Close']
        print("\nSMA(20):", TechnicalIndicators.sma(close, 20).iloc[-1])
        print("RSI(14):", TechnicalIndicators.rsi(close).iloc[-1])
    
    fetcher.close()
