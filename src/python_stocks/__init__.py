"""
python_stocks - Stock Analysis and Machine Learning.

A package for fetching market data, computing technical indicators,
downloading ARK ETF holdings, and running sentiment analysis.
"""

from .ark_fetcher import ARKDataFetcher
from .sentiment import SentimentAnalyzer, StockSentimentFetcher
from .stock_data import StockDataFetcher, TechnicalIndicators

__all__ = [
    "ARKDataFetcher",
    "SentimentAnalyzer",
    "StockDataFetcher",
    "StockSentimentFetcher",
    "TechnicalIndicators",
]

__version__ = "1.1.0"