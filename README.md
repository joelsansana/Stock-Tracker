# Python Stocks 📈

Stock analysis and machine learning modules.

## Description

A collection of Python modules for stock market analysis, sentiment analysis, and machine learning.

## Module Structure

```
python_stocks/
├── src/
│   ├── __init__.py          # Package exports
│   ├── ark_fetcher.py       # ARK ETF holdings fetcher
│   ├── stock_data.py        # Stock data fetching & technical indicators
│   └── sentiment.py         # Sentiment analysis for stocks
├── data/                    # Data storage directory
├── notebooks/               # Jupyter notebooks
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/joelsansana/python_stocks.git
cd python_stocks

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### ARK ETF Holdings

```python
from src.ark_fetcher import ARKDataFetcher

fetcher = ARKDataFetcher()

# Get single ETF
arkk = fetcher.get_holding('ARKK')

# Get all ETFs
all_holdings = fetcher.get_all_holdings()

fetcher.close()
```

### Stock Data

```python
from src.stock_data import StockDataFetcher, TechnicalIndicators

fetcher = StockDataFetcher()

# Get price data
aapl = fetcher.get_price('AAPL', period='1y')

# Calculate indicators
close = aapl['Close']
sma_20 = TechnicalIndicators.sma(close, 20)
rsi = TechnicalIndicators.rsi(close)
macd, signal, hist = TechnicalIndicators.macd(close)

fetcher.close()
```

### Sentiment Analysis

```python
from src.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Analyze text
result = analyzer.analyze_textblob("Stock is going up!")
print(result)  # {'polarity': 0.5, 'subjectivity': 0.5, 'sentiment': 'positive'}

# Batch analysis
texts = ["Good news", "Bad news", "Neutral"]
df = analyzer.analyze_batch_textblob(texts)
```

## Requirements

- Python 3.8+
- numpy
- pandas
- requests
- yfinance (for stock data)
- textblob (for basic sentiment)
- transformers (optional, for advanced sentiment)

See `requirements.txt` for full list.

## License

MIT License - See LICENSE file.

## Author

Joel Sansana
