"""Stock Analysis page — price chart + technical indicators."""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path setup; see app/_bootstrap.py)
import pandas as pd
import streamlit as st

from app.components.cache import fetch_price
from app.components.charts import (
    indicator_subplot,
    normalized_compare,
    price_chart,
)
from app.components.shared import sidebar_nav
from python_stocks import StockDataFetcher, TechnicalIndicators

PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
INTERVALS = ["1d", "1wk", "1mo"]


def _decorate(df: pd.DataFrame, indicators: dict[str, bool]) -> pd.DataFrame:
    """Attach the selected indicator columns to ``df``."""
    if df is None or df.empty or "Close" not in df.columns:
        return df
    close = df["Close"]
    out = df.copy()
    if indicators["SMA 20"]:
        out["SMA_20"] = TechnicalIndicators.sma(close, 20)
    if indicators["SMA 50"]:
        out["SMA_50"] = TechnicalIndicators.sma(close, 50)
    if indicators["EMA 20"]:
        out["EMA_20"] = TechnicalIndicators.ema(close, 20)
    if indicators["Bollinger"]:
        upper, middle, lower = TechnicalIndicators.bollinger_bands(close)
        out["BB_upper"] = upper
        out["BB_middle"] = middle
        out["BB_lower"] = lower
    if indicators["RSI"]:
        out["RSI_14"] = TechnicalIndicators.rsi(close, 14)
    if indicators["MACD"]:
        macd, signal, hist = TechnicalIndicators.macd(close)
        out["MACD"] = macd
        out["MACD_signal"] = signal
        out["MACD_hist"] = hist
    return out


def main() -> None:
    st.set_page_config(
        page_title="Stock Analysis — python_stocks",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
    )
    sidebar_nav()
    st.title("Stock Analysis")

    mode = st.sidebar.radio("Mode", ("Single ticker", "Compare tickers"), horizontal=False)
    period = st.sidebar.selectbox("Period", PERIODS, index=3)
    interval = st.sidebar.selectbox("Interval", INTERVALS, index=0)

    st.sidebar.subheader("Indicators")
    indicators = {
        "SMA 20": st.sidebar.checkbox("SMA 20", value=True),
        "SMA 50": st.sidebar.checkbox("SMA 50", value=False),
        "EMA 20": st.sidebar.checkbox("EMA 20", value=False),
        "Bollinger": st.sidebar.checkbox("Bollinger Bands (20, 2σ)", value=False),
        "RSI": st.sidebar.checkbox("RSI (14)", value=True),
        "MACD": st.sidebar.checkbox("MACD (12,26,9)", value=False),
    }

    if mode == "Single ticker":
        _render_single(period, interval, indicators)
    else:
        _render_compare(period, interval)


def _render_single(period: str, interval: str, indicators: dict[str, bool]) -> None:
    raw_ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
    if not raw_ticker:
        st.info("Enter a ticker in the sidebar to begin.")
        return

    force_refresh = st.sidebar.checkbox(
        "Force refresh",
        value=False,
        help="Bypass the cached response and re-download from yfinance. "
        "Useful when yfinance returned no data and the error has been "
        "cached for an hour.",
    )

    df = fetch_price(raw_ticker, period, interval, force_refresh=force_refresh)
    if df is None or df.empty:
        st.error(
            f"No data returned for **{raw_ticker}** "
            f"(period=`{period}`, interval=`{interval}`)."
        )
        st.info(
            "Troubleshooting:\n"
            "- Check the ticker is valid on Yahoo Finance.\n"
            "- Some tickers don't support intervals smaller than `1d`.\n"
            "- Intraday data (`1m`, `5m`, `15m`, etc.) is limited to the last 60 days.\n"
            "- Yahoo may have rate-limited this server. Tick **Force refresh** "
            "above to retry once the upstream is reachable again."
        )
        return

    df = _decorate(df, indicators)

    overlays: list[tuple[str, str]] = []
    if indicators["SMA 20"]:
        overlays.append(("SMA_20", "SMA 20"))
    if indicators["SMA 50"]:
        overlays.append(("SMA_50", "SMA 50"))
    if indicators["EMA 20"]:
        overlays.append(("EMA_20", "EMA 20"))
    if indicators["Bollinger"]:
        overlays.extend(
            [
                ("BB_upper", "BB upper"),
                ("BB_middle", "BB middle"),
                ("BB_lower", "BB lower"),
            ]
        )

    st.plotly_chart(price_chart(df, overlays=overlays, show_volume=True), use_container_width=True)

    if indicators["RSI"] and "RSI_14" in df.columns:
        fig = indicator_subplot(df, ["RSI_14"], title="RSI (14)")
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
        st.plotly_chart(fig, use_container_width=True)

    if indicators["MACD"] and "MACD" in df.columns:
        fig = indicator_subplot(
            df,
            ["MACD", "MACD_signal"],
            title="MACD (12, 26, 9)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(df.tail(60), use_container_width=True)


def _render_compare(period: str, interval: str) -> None:
    raw = st.sidebar.text_input(
        "Tickers (comma-separated, max 5)",
        value="AAPL, MSFT, GOOGL",
    ).strip()
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not tickers:
        st.info("Enter at least one ticker.")
        return
    if len(tickers) > 5:
        st.warning("Only the first 5 tickers will be plotted.")
        tickers = tickers[:5]

    fetcher = StockDataFetcher()
    try:
        data = fetcher.get_multiple_prices(tickers, period=period, interval=interval)
    finally:
        fetcher.close()

    if not data:
        st.error("No data returned for any of the requested tickers.")
        return

    st.plotly_chart(normalized_compare(data), use_container_width=True)

    st.subheader("Last close")
    last = {t: float(df["Close"].iloc[-1]) for t, df in data.items()}
    st.json(last)


if __name__ == "__main__":
    main()