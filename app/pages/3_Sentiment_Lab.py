"""Sentiment Lab page — STUB. See docs/WEB_UI.md."""

from __future__ import annotations

import streamlit as st

from app.components.shared import backend_status, sidebar_nav


def main() -> None:
    st.set_page_config(page_title="Sentiment Lab — python_stocks", layout="wide")
    sidebar_nav()
    st.title("Sentiment Lab")
    st.info("This page is planned but not yet implemented. See `docs/WEB_UI.md`.")
    status = backend_status()
    st.write("Backend status:", {k: v for k, v in status.items() if k.startswith("sent") or "blob" in k})


if __name__ == "__main__":
    main()