"""Sentiment Lab page — TextBlob vs Hugging Face, single text or batch CSV."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import sentiment_models
from app.components.shared import backend_status, sidebar_nav
from python_stocks import SentimentAnalyzer

SAMPLE_TEXTS: dict[str, str] = {
    "Bullish earnings": (
        "Apple reported blowout earnings, beating analyst estimates on both "
        "revenue and EPS. The stock rallied 8% in after-hours trading."
    ),
    "Bearish macro": (
        "Inflation is surging and the Fed has no room to cut rates. Analysts "
        "expect a prolonged bear market with significant downside risk."
    ),
    "Neutral news": (
        "The company held its annual shareholder meeting on Tuesday. "
        "No material announcements were made."
    ),
}


def _render_polarity_bar(polarity: float, label: str = "polarity") -> None:
    """Render a small horizontal bar from -1 (red) to +1 (green)."""
    if polarity is None or pd.isna(polarity):
        st.caption("(no score)")
        return
    normalized = (float(polarity) + 1.0) / 2.0  # map -1..1 -> 0..1
    st.progress(min(max(normalized, 0.0), 1.0), text=f"{label}: {polarity:+.2f}")


def _render_score_bar(score: float) -> None:
    if score is None or pd.isna(score):
        st.caption("(no score)")
        return
    st.progress(min(max(float(score), 0.0), 1.0), text=f"score: {score:.2f}")


def _format_textblob(result: dict) -> None:
    col1, col2 = st.columns(2)
    col1.metric("Polarity", f"{result['polarity']:+.3f}")
    col2.metric("Subjectivity", f"{result['subjectivity']:.3f}")
    _render_polarity_bar(result["polarity"])
    sentiment = result["sentiment"]
    color = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(sentiment, "⚪")
    st.markdown(f"**Label:** {color} {sentiment}")


def _format_hf(result: dict) -> None:
    label = str(result.get("label", "")).upper()
    score = float(result.get("score", 0.5))
    color = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "⚪"}.get(label, "⚪")
    st.markdown(f"**Label:** {color} {label}")
    _render_score_bar(score)


def main() -> None:
    st.set_page_config(
        page_title="Sentiment Lab — python_stocks",
        page_icon=":speech_balloon:",
        layout="wide",
    )
    sidebar_nav()
    st.title("Sentiment Lab")

    backend_status()
    analyzer = SentimentAnalyzer()

    if analyzer.backend == "none":
        st.error(
            "No sentiment backend is installed. Install at least "
            "`pip install textblob` (and run `python -m textblob.download_corpora`)."
        )
        st.stop()

    mode = st.sidebar.radio("Mode", ("Single text", "Batch CSV"), horizontal=False)

    hf_enabled = sentiment_models.hf_available()
    hf_pipeline: sentiment_models.HuggingFaceSentiment | None = None

    if hf_enabled:
        st.sidebar.subheader("Hugging Face")
        preset = st.sidebar.selectbox(
            "Model",
            sentiment_models.list_presets(),
            index=0,
            help="`default` = DistilBERT SST-2; `finbert` = ProsusAI/FinBERT (finance).",
        )
        device_choice = st.sidebar.radio(
            "device",
            ("CPU", "GPU (cuda:0)"),
            index=0,
            horizontal=True,
        )
        device = 0 if device_choice.startswith("GPU") else -1
        if st.sidebar.button("Load model", type="primary"):
            with st.spinner(f"Loading {preset}…"):
                hf_pipeline = sentiment_models.load_hf_pipeline(preset, device)
            if hf_pipeline.available:
                st.sidebar.success(f"{preset} loaded")
            else:
                st.sidebar.error("Model failed to load")
        # Reuse an already-cached pipeline across reruns.
        if hf_pipeline is None and analyzer.huggingface.available:
            hf_pipeline = analyzer.huggingface
    else:
        st.sidebar.info(
            "Hugging Face backend not installed. "
            "Install with `pip install -e .[transformers]` to enable."
        )

    if mode == "Single text":
        _render_single(analyzer, hf_pipeline)
    else:
        _render_batch(analyzer, hf_pipeline)


def _render_single(
    analyzer: SentimentAnalyzer,
    hf_pipeline: sentiment_models.HuggingFaceSentiment | None,
) -> None:
    st.subheader("Single text")

    cols = st.columns(len(SAMPLE_TEXTS))
    for col, (label, text) in zip(cols, SAMPLE_TEXTS.items()):
        if col.button(label, use_container_width=True):
            st.session_state["sentiment_text"] = text

    text = st.text_area(
        "Text to analyze",
        value=st.session_state.get("sentiment_text", ""),
        height=140,
        placeholder="Paste a tweet, headline, or paragraph here…",
    )

    if not text.strip():
        st.info("Type or paste text above, or click a sample.")
        return

    left, right = st.columns(2)

    with left:
        st.markdown("### TextBlob (lexicon)")
        with st.spinner("Scoring…"):
            tb_result = analyzer.analyze_textblob(text)
        _format_textblob(tb_result)

    with right:
        st.markdown("### Hugging Face")
        if hf_pipeline is None:
            st.caption(
                "Click **Load model** in the sidebar to enable. "
                "Or install the `[transformers]` extra."
            )
        else:
            with st.spinner("Scoring…"):
                hf_result = analyzer.analyze_huggingface(text)
            _format_hf(hf_result)


def _render_batch(
    analyzer: SentimentAnalyzer,
    hf_pipeline: sentiment_models.HuggingFaceSentiment | None,
) -> None:
    st.subheader("Batch CSV")
    st.caption(
        "Upload a CSV with a text column. The scored output keeps your "
        "original columns and appends the sentiment columns."
    )

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to begin.")
        return

    try:
        df = pd.read_csv(uploaded)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    if df.empty:
        st.warning("CSV is empty.")
        return

    text_col = st.selectbox(
        "Text column",
        options=list(df.columns),
        index=0,
        help="Column containing the text to score.",
    )

    backend_choice = st.radio(
        "Backend",
        ("TextBlob", "Hugging Face" if hf_pipeline is not None else "Hugging Face (load model in sidebar first)"),
        index=0,
        horizontal=True,
        disabled=hf_pipeline is None,
    )
    if backend_choice.startswith("Hugging Face") and hf_pipeline is None:
        st.warning("Load a Hugging Face model in the sidebar first.")
        return

    if st.button("Score", type="primary"):
        texts = df[text_col].astype(str).tolist()
        if backend_choice == "TextBlob":
            with st.spinner(f"Scoring {len(texts)} rows with TextBlob…"):
                scored = analyzer.analyze_batch_textblob(texts)
        else:
            with st.spinner(f"Scoring {len(texts)} rows with Hugging Face…"):
                scored = analyzer.analyze_batch_huggingface(texts)

        # Merge the original frame with the scored columns.
        merged = pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
        st.session_state["scored_df"] = merged
        st.session_state["scored_backend"] = backend_choice

    if "scored_df" in st.session_state:
        merged = st.session_state["scored_df"]
        backend_used = st.session_state.get("scored_backend", "?")
        st.success(f"Scored {len(merged)} rows with {backend_used}.")

        if "sentiment" in merged.columns:
            counts = merged["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
            c1, c2, c3 = st.columns(3)
            c1.metric("Positive", int(counts.get("positive", 0)))
            c2.metric("Neutral", int(counts.get("neutral", 0)))
            c3.metric("Negative", int(counts.get("negative", 0)))
        elif "label" in merged.columns:
            counts = merged["label"].str.upper().value_counts().reindex(["POSITIVE", "NEUTRAL", "NEGATIVE"], fill_value=0)
            c1, c2, c3 = st.columns(3)
            c1.metric("Positive", int(counts.get("POSITIVE", 0)))
            c2.metric("Neutral", int(counts.get("NEUTRAL", 0)))
            c3.metric("Negative", int(counts.get("NEGATIVE", 0)))

        st.dataframe(merged, use_container_width=True, hide_index=True)

        st.download_button(
            "Download scored CSV",
            data=merged.to_csv(index=False).encode("utf-8"),
            file_name="scored.csv",
            mime="text/csv",
            type="primary",
        )


if __name__ == "__main__":
    main()