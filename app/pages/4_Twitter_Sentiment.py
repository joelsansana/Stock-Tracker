"""Twitter Sentiment page — STUB. See docs/WEB_UI.md."""

from __future__ import annotations

import streamlit as st

from app.components.shared import sidebar_nav, twitter_configured


def main() -> None:
    st.set_page_config(page_title="Twitter Sentiment — python_stocks", layout="wide")
    sidebar_nav()
    st.title("Twitter Sentiment")
    st.info("This page is planned but not yet implemented. See `docs/WEB_UI.md`.")

    token = twitter_configured()
    if token:
        st.success("TWEET_BEARER_TOKEN is configured.")
    else:
        st.warning(
            "No `TWEET_BEARER_TOKEN` found. Set it in `.streamlit/secrets.toml` "
            "or as an environment variable to enable this page."
        )


if __name__ == "__main__":
    main()