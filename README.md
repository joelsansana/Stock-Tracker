# Python Stocks 📈

Stock analysis toolkit: ARK ETF holdings, market data, technical
indicators, and sentiment analysis.

> **Note:** This is the package formerly known as `python_stocks`. The
> importable package name is now `python_stocks` (the source lives in
> `src/python_stocks/`).

## Features

- **ARK ETF holdings** — download and cache holdings CSVs for ARKK,
  ARKQ, ARKW, ARKG, ARKF from ark-funds.com (`ARKDataFetcher`).
- **Market data** — pull OHLCV history via `yfinance` with optional
  CSV caching (`StockDataFetcher`).
- **Technical indicators** — SMA, EMA, Wilder's RSI, MACD, Bollinger
  Bands (`TechnicalIndicators`).
- **Sentiment** — lexicon (TextBlob) and neural (Hugging Face)
  backends, plus a Twitter v2 fetcher (`SentimentAnalyzer`,
  `StockSentimentFetcher`).

## Repository layout

```
Stock-Tracker/
├── src/
│   └── python_stocks/
│       ├── __init__.py
│       ├── ark_fetcher.py
│       ├── stock_data.py
│       └── sentiment.py
├── tests/
├── notebooks/
│   └── stock_analysis_example.ipynb
├── pyproject.toml         # installable via pip
├── requirements.txt       # runtime deps (extras in pyproject.toml)
├── LICENSE                # MIT
├── README.md
└── .gitignore
```

## Installation

```bash
git clone https://github.com/joelsansana/python_stocks.git
cd Stock-Tracker
python3 -m venv .venv
source .venv/bin/activate

# Core install (small):
pip install -e .

# Optional extras:
pip install -e ".[dev]"            # tests + matplotlib + notebook deps
pip install -e ".[transformers]"   # Hugging Face sentiment backend
pip install -e ".[twitter]"        # Twitter API v2 client
```

The package is named `python_stocks` so all imports below resolve to
`from python_stocks import ...`.

## Usage

### ARK ETF holdings

```python
from python_stocks import ARKDataFetcher

with ARKDataFetcher(data_dir="data") as fetcher:
    arkk = fetcher.get_holding("ARKK")
    all_holdings = fetcher.get_all_holdings()  # parallel download
```

### Stock data & technical indicators

```python
from python_stocks import StockDataFetcher, TechnicalIndicators

with StockDataFetcher(data_dir="data") as fetcher:
    aapl = fetcher.get_price("AAPL", period="1y")
    close = aapl["Close"]
    sma_20 = TechnicalIndicators.sma(close, 20)
    rsi_14 = TechnicalIndicators.rsi(close)            # Wilder's smoothing
    macd, signal, hist = TechnicalIndicators.macd(close)
    upper, middle, lower = TechnicalIndicators.bollinger_bands(close)
```

### Sentiment analysis

```python
from python_stocks import SentimentAnalyzer

analyzer = SentimentAnalyzer()            # picks the best available backend
print(analyzer.backend)                    # "huggingface" | "textblob" | "none"

result = analyzer.analyze_textblob("Stock is going up!")
# {'polarity': 0.5, 'subjectivity': 0.5, 'sentiment': 'positive'}

df = analyzer.analyze_batch_textblob(["Good news", "Bad news", "Neutral"])
print(analyzer.aggregate_sentiment(df))
```

For finance-tuned neural sentiment, pass a FinBERT model:

```python
analyzer = SentimentAnalyzer(hf_model="ProsusAI/finbert")
```

### Twitter sentiment

The Twitter fetcher uses the **v2 API** (v1.1 was retired by Twitter in
2023). Set `TWEET_BEARER_TOKEN` in your environment, then:

```python
from python_stocks import StockSentimentFetcher

fetcher = StockSentimentFetcher()
df = fetcher.analyze_stock_sentiment("AAPL", count=100)
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

## Requirements

Python 3.8+. Runtime dependencies are listed in
[`requirements.txt`](requirements.txt); everything heavier
(`transformers`, `torch`, `tweepy`, …) lives in optional extras in
[`pyproject.toml`](pyproject.toml).

## License

[MIT](LICENSE) © Joel Sansana.

## Author

Joel Sansana