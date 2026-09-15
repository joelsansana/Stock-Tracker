"""Home page for the python_stocks Streamlit UI.

Run with::

    streamlit run app/Home.py
"""  # noqa: N999  (Streamlit requires this filename)

from __future__ import annotations

import streamlit as st

from app.components.shared import (
    available_etfs,
    backend_status,
    package_version,
    sidebar_nav,
    twitter_configured,
)


def main() -> None:
    st.set_page_config(
        page_title="python_stocks",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
    )

    sidebar_nav()
    st.title("python_stocks :chart_with_upwards_trend:")
    st.caption(f"v{package_version()} — stock analysis toolkit")

    status = backend_status()
    cols = st.columns(4)
    cols[0].metric("yfinance", "ready" if status["yfinance"] else "missing")
    cols[1].metric(
        "sentiment",
        "ready" if status.get("sentiment_backend", False) else "missing",
    )
    cols[2].metric(
        "twitter",
        "configured" if twitter_configured() else "no token",
    )
    cols[3].metric("ARK ETFs", len(available_etfs()))

    st.markdown(
        """
        ## Pages

        Use the sidebar to navigate. Each page calls into the
        `python_stocks` library — no separate data layer.

        | Page | What it does |
        |---|---|
        | **ARK Holdings** | Browse ARK Invest ETF holdings (ARKK, ARKQ, ARKW, ARKG, ARKF). |
        | **Stock Analysis** | OHLCV chart + technical indicators for any symbol. |
        | **Sentiment Lab** | *(planned)* Score text with TextBlob and Hugging Face. |
        | **Twitter Sentiment** | *(planned)* Live tweet sentiment for a ticker. |
        | **Status** | *(planned)* Diagnostic info about this installation. |

        See [`docs/WEB_UI.md`](../docs/WEB_UI.md) for the full design doc.
        """
    )

    with st.expander("Backend details"):
        for name, ok in status.items():
            st.write(f"- **{name}**: {'✅' if ok else '❌'}")


if __name__ == "__main__":
    main()