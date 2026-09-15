"""Tests for the Twitter Sentiment helper functions."""

from __future__ import annotations

import pandas as pd

from app.components.twitter_helpers import aggregate_counts, classify_sentiment


def test_classify_textblob_sentiment() -> None:
    assert classify_sentiment(pd.Series({"sentiment": "positive"})) == "positive"
    assert classify_sentiment(pd.Series({"sentiment": "neutral"})) == "neutral"
    assert classify_sentiment(pd.Series({"sentiment": "negative"})) == "negative"


def test_classify_huggingface_label() -> None:
    assert classify_sentiment(pd.Series({"label": "POSITIVE"})) == "positive"
    assert classify_sentiment(pd.Series({"label": "NEGATIVE"})) == "negative"
    assert classify_sentiment(pd.Series({"label": "NEUTRAL"})) == "neutral"
    # case-insensitive
    assert classify_sentiment(pd.Series({"label": "positive"})) == "positive"


def test_classify_unknown_label_falls_back_to_neutral() -> None:
    """Unknown labels map to 'neutral' (we don't want to drop the row)."""
    assert classify_sentiment(pd.Series({"label": "WEIRD"})) == "neutral"


def test_classify_no_columns_returns_none() -> None:
    assert classify_sentiment(pd.Series({})) is None


def test_classify_non_string_is_none() -> None:
    assert classify_sentiment(pd.Series({"sentiment": 1.0})) is None
    assert classify_sentiment(pd.Series({"label": None})) is None


def test_aggregate_counts_empty_df() -> None:
    counts = aggregate_counts(pd.DataFrame())
    assert counts.to_dict() == {"positive": 0, "neutral": 0, "negative": 0}


def test_aggregate_counts_missing_column() -> None:
    counts = aggregate_counts(pd.DataFrame({"text": ["a", "b"]}))
    assert counts.to_dict() == {"positive": 0, "neutral": 0, "negative": 0}


def test_aggregate_counts_basic() -> None:
    df = pd.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "sentiment_label": ["positive", "positive", "negative", "neutral"],
        }
    )
    counts = aggregate_counts(df)
    assert counts["positive"] == 2
    assert counts["neutral"] == 1
    assert counts["negative"] == 1


def test_aggregate_counts_fills_missing_labels() -> None:
    """If 'positive' is missing from the data, it should still be 0, not NaN."""
    df = pd.DataFrame(
        {
            "text": ["a", "b"],
            "sentiment_label": ["negative", "negative"],
        }
    )
    counts = aggregate_counts(df)
    assert counts["positive"] == 0
    assert counts["negative"] == 2
    assert counts["neutral"] == 0