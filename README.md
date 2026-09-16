# Python Stocks 📈

### 👉 **[Try the live demo → https://stock-ark-tracker.streamlit.app](https://stock-ark-tracker.streamlit.app)** 👈

Stock analysis toolkit: ARK ETF holdings, market data, technical
indicators, and sentiment analysis — with a hosted Streamlit UI
that's free to use, no install required.

---

> **Note:** This is the package formerly known as `python_stocks`. The
> importable package name is now `python_stocks` (the source lives in
> `src/python_stocks/`).

## Features

- **ARK ETF holdings** — download and cache holdings CSVs for ARKK,
  ARKQ, ARKW, ARKG, ARKF, ARKX from ark-funds.com (`ARKDataFetcher`).
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
├── app/                       # Streamlit UI
│   ├── Home.py
│   ├── pages/
│   │   ├── 1_ARK_Holdings.py
│   │   ├── 2_Stock_Analysis.py
│   │   ├── 3_Sentiment_Lab.py        # stub
│   │   ├── 4_Twitter_Sentiment.py    # stub
│   │   └── 5_Status.py
│   └── components/
├── tests/
├── docs/
│   └── WEB_UI.md              # web UI design doc
├── notebooks/
│   └── stock_analysis_example.ipynb
├── pyproject.toml             # installable via pip
├── requirements.txt           # runtime deps (extras in pyproject.toml)
├── LICENSE                    # MIT
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
pip install -e ".[ui]"             # Streamlit web UI
pip install -e ".[streamlit]"      # all UI extras combined
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

## Web UI

A Streamlit UI lives in `app/`. After installing the `[ui]` extra:

```bash
streamlit run app/Home.py
```

It currently ships:

- **Home** — backend status overview
- **ARK Holdings** — browse ARK ETF holdings with treemap/bar charts and CSV download
- **Stock Analysis** — candlestick + technical indicators (SMA, EMA, Bollinger, RSI, MACD) and multi-ticker compare
- **Sentiment Lab** — score text with TextBlob and Hugging Face; single text or batch CSV
- **Twitter Sentiment** — fetch recent tweets via Twitter API v2 and score them
- **Status** — diagnostic info, cache maintenance, data folder

Twitter credentials are read from `.streamlit/secrets.toml` (see
`.streamlit/secrets.toml.example`) or the `TWEET_BEARER_TOKEN` env var.

### Running locally

The UI works regardless of the directory you launch Streamlit from —
each page imports `app/_bootstrap.py` first, which adds the project
root to `sys.path`. Just make sure you installed the package first:

```bash
git clone https://github.com/joelsansana/python_stocks.git
cd Stock-Tracker
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ui,dev]"   # or "[streamlit]" for everything

streamlit run app/Home.py
```

Then open <http://localhost:8501>.

### Deploying to Streamlit Community Cloud

`requirements.txt` is wired for [share.streamlit.io](https://share.streamlit.io):

1. Push to `main` (or your default branch).
2. On Streamlit Cloud, click **Create app** → pick the repo,
   branch `main`, main file `app/Home.py`.
3. Under **Advanced settings** → **Secrets**, paste:
   ```toml
   TWEET_BEARER_TOKEN = "your-twitter-v2-bearer-token"
   ```
4. Deploy. Cloud reads `requirements.txt`, installs the local
   package and Streamlit/Plotly, and serves the app at
   `<subdomain>.streamlit.app`.

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the full guide
(secret management, persistence caveats, Hugging Face on Cloud,
troubleshooting).

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'app'` | Make sure `app/_bootstrap.py` exists and is the first import in every page (it should be — don't reorder). |
| `ModuleNotFoundError: No module named 'python_stocks'` | You didn't install the package. Run `pip install -e .` (or one of the extras). |
| ARK page shows "Could not fetch ARKK" | ARK may have changed their CSV URL — check `ARK_ETF_URLS` in `src/python_stocks/ark_fetcher.py`. |
| Stock page shows "No data returned" | Verify the ticker on Yahoo Finance; some intervals are limited to recent data. |
| Twitter page asks for a bearer token | Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in `TWEET_BEARER_TOKEN`. |
| Hugging Face model never loads | First load downloads ~250 MB. Check the Status page to confirm `transformers` is installed. |

## Requirements

Python 3.10+ (required by Streamlit ≥ 1.30). Runtime dependencies are listed in
[`requirements.txt`](requirements.txt); everything heavier
(`transformers`, `torch`, `tweepy`, …) lives in optional extras in
[`pyproject.toml`](pyproject.toml).

## License

[MIT](LICENSE) © Joel Sansana.

## Author

Joel Sansana