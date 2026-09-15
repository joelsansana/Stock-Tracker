"""Helpers for the Twitter Sentiment page.

Kept Streamlit-free so the classification / aggregation logic is
unit-testable without a Streamlit runtime.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def classify_sentiment(row: pd.Series) -> Optional[str]:
    """Normalize either backend's result column into a single label.

    TextBlob results have a ``sentiment`` column with values
    ``"positive" | "neutral" | "negative"``. Hugging Face results have
    a ``label`` column with values ``"POSITIVE" | "NEGATIVE" | "NEUTRAL"``.

    Returns ``None`` if neither column is present or the value isn't
    recognized.
    """
    if "sentiment" in row and isinstance(row["sentiment"], str):
        return row["sentiment"]
    if "label" in row and isinstance(row["label"], str):
        lbl = row["label"].upper()
        if lbl == "POSITIVE":
            return "positive"
        if lbl == "NEGATIVE":
            return "negative"
        return "neutral"
    return None


def aggregate_counts(df: pd.DataFrame) -> pd.Series:
    """Count positive/neutral/negative labels, filling missing with 0."""
    if df.empty or "sentiment_label" not in df.columns:
        return pd.Series({"positive": 0, "neutral": 0, "negative": 0}, dtype=int)
    return (
        df["sentiment_label"]
        .value_counts()
        .reindex(["positive", "neutral", "negative"], fill_value=0)
        .astype(int)
    )