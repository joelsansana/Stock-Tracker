"""Twitter Sentiment page — fetch recent tweets and score them."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.shared import (
    backend_status,
    sidebar_nav,
    twitter_configured,
    twitter_fetcher,
)
from app.components.twitter_helpers import aggregate_counts, classify_sentiment
from python_stocks import SentimentAnalyzer

MAX_TWEETS = 100  # Twitter v2 search_recent_tweets cap per request


def main() -> None:
    st.set_page_config(
        page_title="Twitter Sentiment — python_stocks",
        page_icon=":bird:",
        layout="wide",
    )
    sidebar_nav()
    st.title("Twitter Sentiment")

    if not twitter_configured():
        st.error(
            "Twitter API v2 bearer token is not configured.\n\n"
            "Set `TWEET_BEARER_TOKEN` in `.streamlit/secrets.toml` "
            "(see `.streamlit/secrets.toml.example`) or as an environment "
            "variable, then restart the app."
        )
        st.stop()

    fetcher = twitter_fetcher()
    if fetcher is None:
        st.error("Could not initialize Twitter client.")
        st.stop()

    backend_status()
    analyzer = SentimentAnalyzer()
    if analyzer.backend == "none":
        st.error(
            "No sentiment backend installed. Install TextBlob "
            "(`pip install textblob`) or the `[transformers]` extra."
        )
        st.stop()

    st.sidebar.subheader("Search")
    raw_ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
    if not raw_ticker:
        st.info("Enter a ticker in the sidebar.")
        return
    count = st.sidebar.slider(
        "Max tweets",
        min_value=10,
        max_value=MAX_TWEETS,
        value=50,
        step=10,
        help="Twitter v2 caps a single search at 100 tweets.",
    )

    run = st.sidebar.button("Fetch & analyze", type="primary")

    if "twitter_df" not in st.session_state:
        st.session_state["twitter_df"] = pd.DataFrame()

    if run:
        with st.spinner(f"Fetching recent tweets for ${raw_ticker}…"):
            try:
                df = fetcher.analyze_stock_sentiment(raw_ticker, count=count)
            except Exception as exc:  # tweepy raises varied exceptions
                st.error(f"Twitter fetch failed: {exc}")
                return
        if df is None or df.empty:
            st.warning(f"No tweets returned for ${raw_ticker}.")
            st.session_state["twitter_df"] = pd.DataFrame()
            return
        st.session_state["twitter_df"] = df
        st.session_state["twitter_ticker"] = raw_ticker

    df = st.session_state["twitter_df"]
    if df.empty:
        st.info("Click **Fetch & analyze** in the sidebar to start.")
        return

    df = df.copy()
    df["sentiment_label"] = df.apply(classify_sentiment, axis=1)

    # --- Aggregate metrics -------------------------------------------------
    counts = aggregate_counts(df)
    total = int(counts.sum())
    if total > 0:
        pos_pct = counts.get("positive", 0) / total * 100
        neg_pct = counts.get("negative", 0) / total * 100
        neu_pct = counts.get("neutral", 0) / total * 100
    else:
        pos_pct = neg_pct = neu_pct = 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tweets", total)
    c2.metric("Positive", f"{counts.get('positive', 0)} ({pos_pct:.0f}%)")
    c3.metric("Neutral", f"{counts.get('neutral', 0)} ({neu_pct:.0f}%)")
    c4.metric("Negative", f"{counts.get('negative', 0)} ({neg_pct:.0f}%)")

    # --- Filter ---------------------------------------------------------
    st.subheader(f"Tweets for ${st.session_state.get('twitter_ticker', raw_ticker)}")
    filter_choice = st.radio(
        "Filter",
        ("All", "Positive", "Neutral", "Negative"),
        horizontal=True,
    )
    if filter_choice != "All":
        df_view = df[df["sentiment_label"] == filter_choice.lower()]
    else:
        df_view = df

    if df_view.empty:
        st.caption(f"No tweets matching filter: {filter_choice}")
        return

    display = df_view[
        [c for c in ["created_at", "user", "text", "sentiment_label", "retweets", "likes"] if c in df_view.columns]
    ].copy()
    if "retweets" in display.columns:
        display["retweets"] = pd.to_numeric(display["retweets"], errors="coerce").fillna(0).astype(int)
    if "likes" in display.columns:
        display["likes"] = pd.to_numeric(display["likes"], errors="coerce").fillna(0).astype(int)
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "text": st.column_config.TextColumn("Tweet", width="large"),
            "user": st.column_config.TextColumn("User", width="small"),
            "created_at": st.column_config.DatetimeColumn("Posted", format="YYYY-MM-DD HH:mm"),
            "sentiment_label": st.column_config.TextColumn("Sentiment", width="small"),
            "retweets": st.column_config.NumberColumn("RT", width="small"),
            "likes": st.column_config.NumberColumn("♥", width="small"),
        },
    )

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{st.session_state.get('twitter_ticker', raw_ticker)}_tweets.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()