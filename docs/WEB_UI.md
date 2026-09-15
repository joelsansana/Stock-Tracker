# Web UI Scope

> Design document for a Streamlit-based UI on top of `python_stocks`.
> **Status: planned.** MVP scaffold lives in `app/`.

## Goals

- Make `python_stocks` usable from a browser without writing Python.
- Keep the UI thin: every page delegates to existing modules in
  `src/python_stocks/`.
- Stay out of the way of programmatic users — no new abstractions or
  duplicated logic in the UI layer.

## Non-goals

- Replacing the Python API.
- Production-grade auth or multi-tenant isolation.
- Real-time streaming tick data (we batch via `yfinance`).

## Framework

**Streamlit** — pure Python, no JS, built-in caching, native multi-page
support, easy to deploy.

Alternatives considered:

| Option | Verdict |
|---|---|
| Streamlit | ✅ chosen — fastest path to interactive data UI |
| Dash (Plotly) | More layout control but ~2× the boilerplate |
| Flask + React | Production-grade but a separate codebase |
| Gradio | Skews toward ML demos; weaker for tabular data |

Charts use **Plotly** (already supported via Streamlit) for
zoom/pan/hover; matplotlib stays in the dev-only notebook path.

## Page structure

```
app/
├── Home.py                 # entry point + nav
├── pages/
│   ├── 1_ARK_Holdings.py│   ├── 2_Stock_Analysis.py
│   ├── 3_Sentiment_Lab.py
│   ├── 4_Twitter_Sentiment.py
│   └── 5_Status.py        # which APIs/deps are available
└── components/
    ├── charts.py           # reusable Plotly wrappers
    └── cache.py            # shared cache_resource setup
```

Five pages total. MVP ships the first two.

## Per-page spec

### 1. Home (`app/Home.py`)

- Hero card: package version, list of available backends
  (yfinance / textblob / transformers / tweepy).
- Quick links into each page.
- Recent activity log (last 5 fetched tickers).

### 2. ARK Holdings (MVP)

- **Sidebar:** ETF selector (ARKK / ARKQ / ARKW / ARKG / ARKF),
  "force refresh" checkbox, cache age display.
- **KPI strip:** # holdings, top-10 concentration %.
- **Treemap** or horizontal bar of weights (Plotly).
- **Sortable, searchable table** (`streamlit-aggrid`).
- "Download CSV" button.

### 3. Stock Analysis (MVP)

- **Sidebar:** ticker input, period (1mo..max), interval, indicator
  checkboxes (SMA20/50, EMA20, BBands, RSI, MACD).
- **Main:** candlestick + volume + selected indicators overlaid.
- **Indicator tabs:** RSI subplot, MACD subplot.
- **Compare tickers** toggle → normalized price overlay for 2–5
  tickers.

### 4. Sentiment Lab

Two-pane: TextBlob vs HuggingFace, side-by-side. Text area + sample
buttons. Model selector (default + FinBERT option if installed). Batch
mode: upload CSV → download scored CSV.

- **Sidebar:** mode (single text / batch CSV), HF model preset
  (`default` = DistilBERT SST-2, `finbert` = ProsusAI/FinBERT), device
  toggle (CPU/GPU), "Load model" button (HF pipelines are loaded once
  via `@st.cache_resource`).
- **Single mode:** sample-text buttons populate the textarea. Each pane
  shows the label, polarity bar (-1..+1) or score bar (0..1), and a
  metric strip.
- **Batch mode:** CSV uploader → column selector → backend pick → run.
  Result is a merged DataFrame (original columns + sentiment columns)
  with counts and a CSV download button.

### 5. Twitter Sentiment

Ticker input, max-tweets slider (cap 100, v2 limit), filterable tweet
table, aggregate stats card. Clear error if `TWEET_BEARER_TOKEN`
missing.

- **Empty state:** explicit error pointing at
  `.streamlit/secrets.toml` or `TWEET_BEARER_TOKEN`.
- **Filter:** radio to show all / positive / neutral / negative rows.
- **Columns:** posted, user, tweet text, sentiment label, retweets,
  likes.
- **Export:** download the full scored DataFrame as CSV.

### 6. Status

Diagnostic panel: Python version, installed dep versions, which
optional extras are present, current `data_dir` size, last cache hit
times. Useful first stop when something's wrong.

## State & caching

| Concern | Mechanism |
|---|---|
| Expensive downloads (prices, ARK CSVs, HF model) | `@st.cache_data(ttl=…)` for CSVs; `@st.cache_resource` for HF pipeline |
| Per-user selections (ticker, period) | `st.session_state` |
| Twitter / HF API keys | `.streamlit/secrets.toml` (gitignored), exposed via `st.secrets` |
| ARK CSV cache | Already disk-cached by `ARKDataFetcher`; Streamlit cache wraps that |

**TTL strategy:**

- ARK holdings: 1 hour (they update daily; intra-day refresh is wasteful)
- Prices: 5 minutes for ≤5d period, 1 hour otherwise
- HF model: cached for process lifetime (don't reload per rerun)

## New dependencies

Add to `pyproject.toml` as a new optional extra:

```toml
ui = [
    "streamlit>=1.30",
    "plotly>=5.18",
    "streamlit-aggrid>=0.3",
]
streamlit = ["python-stocks[ui,transformers,twitter]"]
```

Plotly is intentionally **not** a runtime dep of `python_stocks`; it's
only pulled in by users of the UI.

## Configuration

`.streamlit/config.toml` (committed) sets sensible defaults:

```toml
[theme]
base = "light"
primaryColor = "#0066cc"

[server]
maxUploadSize = 50

[browser]
gatherUsageStats = false
```

`.streamlit/secrets.toml` is **gitignored** and holds API keys:

```toml
TWEET_BEARER_TOKEN = "…"
HF_TOKEN = "…"   # for gated HF models (optional)
```

## Deployment

| Option | Cost | Notes |
|---|---|---|
| Streamlit Community Cloud | Free | 1-click from GitHub; secrets via dashboard; best for getting started |
| Docker on Fly.io / Render / Railway | Free–$5/mo | More control, persistent disk for the `data/` cache |
| Internal VM (`streamlit run`) | – | Easiest if the audience is already on your network |

**Recommended:** Streamlit Community Cloud pointing at `app/Home.py`,
installing the `[streamlit]` meta-extra so all backends are available.

## Phased delivery

| Phase | Scope | Effort | Status |
|---|---|---|---|
| **MVP** | Home + ARK Holdings + Stock Analysis | 1–2 days | ✅ shipped |
| **v1** | + Sentiment Lab, Status page | +1 day | ✅ shipped |
| **v1.1** | + Twitter Sentiment, multi-ticker compare | +1 day | ✅ shipped |
| **Polish** | Error toasts, empty states, light/dark, README updates | +1 day | pending |
| **Deploy** | Community Cloud + secrets + CI smoke test | 0.5 day | pending |

Total to a useful v1: ~3 working days.

## What's shipped so far

**MVP (committed):** Home, ARK Holdings, Stock Analysis (single +
compare), stub pages for the rest.

**v1 + v1.1 (committed):** Sentiment Lab (single text + batch CSV +
HF model loader), Twitter Sentiment (full implementation), Status
page. Helper modules:

- `app/components/sentiment_models.py` — `@st.cache_resource`-wrapped
  Hugging Face pipeline loader with `default` and `finbert` presets.
- `app/components/twitter_helpers.py` — `classify_sentiment` and
  `aggregate_counts` (extracted from the page so they're unit-tested).

Test counts: 48 passed (10 new tests for the helper modules and
headless smoke tests of the new pages).

## Risks & open questions

1. **HF model size** — default model is ~250 MB; first load is slow.
   Pre-warm on startup, or default to TextBlob-only and put HF behind
   a "load neural model" button.
2. **Twitter rate limits** — `wait_on_rate_limit=True` is already on,
   but a "burst" user could exhaust quota. Add a per-IP token bucket in
   `components/cache.py`.
3. **yfinance flakiness** — `StockDataFetcher.get_price` already
   returns `None` on failure; the UI must show that gracefully, not
   crash.
4. **Ephemeral storage on Community Cloud** — the `data/` cache is
   lost on redeploy. Acceptable; document it.
5. **Multi-user safety** — caching keyed only on ticker+period is fine;
   do not include user tokens in cache keys.

## Local development

```bash
pip install -e ".[ui,dev]"
streamlit run app/Home.py
```

The app reads ARK CSV / price data through the existing
`python_stocks` API; no separate data layer.

## Testing

The UI itself is hard to unit-test cleanly (Streamlit rerun model).
Strategy:

- Cover the **components** (`charts.py`, `cache.py`) with pure-Python
  tests against mock data.
- Add a smoke test that runs `streamlit run` headlessly and checks each
  page returns HTTP 200 without exception. Use
  `streamlit.testing.v1.AppTest` (Streamlit ≥1.28) for this.
- Existing `tests/` for the underlying `python_stocks` modules stay
  unchanged.