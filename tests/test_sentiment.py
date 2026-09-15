"""Tests for the sentiment module."""

from __future__ import annotations

import pandas as pd
import pytest

from python_stocks import SentimentAnalyzer
from python_stocks.sentiment import (
    HuggingFaceSentiment,
    TextBlobSentiment,
    _classify_polarity,
)


def test_classify_polarity_thresholds() -> None:
    assert _classify_polarity(0.5) == "positive"
    assert _classify_polarity(0.05) == "neutral"
    assert _classify_polarity(-0.5) == "negative"
    assert _classify_polarity(0.1) == "neutral"  # boundary belongs to neutral


def test_textblob_returns_dict_when_unavailable() -> None:
    """If textblob is not installed, we still return a dict (never raise)."""
    tb = TextBlobSentiment()
    if tb.available:
        pytest.skip("textblob installed; covered by other tests")
    out = tb.analyze("anything")
    assert out == {"polarity": 0.0, "subjectivity": 0.0, "sentiment": "neutral"}


def test_huggingface_returns_dict_when_unavailable() -> None:
    hf = HuggingFaceSentiment()
    if hf.available:
        pytest.skip("transformers installed; covered by other tests")
    out = hf.analyze("anything")
    assert out == {"label": "NEUTRAL", "score": 0.5}


def test_aggregate_sentiment_empty_textblob() -> None:
    """Empty input must not raise ZeroDivisionError."""
    out = SentimentAnalyzer.aggregate_sentiment(pd.DataFrame(), method="textblob")
    assert out["count"] == 0
    assert out["avg_polarity"] == 0.0


def test_aggregate_sentiment_empty_huggingface() -> None:
    out = SentimentAnalyzer.aggregate_sentiment(pd.DataFrame(), method="huggingface")
    assert out["count"] == 0
    assert out["positive_ratio"] == 0.0


def test_aggregate_sentiment_textblob_basic() -> None:
    df = pd.DataFrame(
        [
            {"polarity": 0.5, "subjectivity": 0.5, "sentiment": "positive"},
            {"polarity": -0.5, "subjectivity": 0.5, "sentiment": "negative"},
            {"polarity": 0.0, "subjectivity": 0.5, "sentiment": "neutral"},
        ]
    )
    out = SentimentAnalyzer.aggregate_sentiment(df, method="textblob")
    assert out["count"] == 3
    assert out["avg_polarity"] == pytest.approx(0.0, abs=1e-9)
    assert out["positive_ratio"] == pytest.approx(1 / 3)
    assert out["negative_ratio"] == pytest.approx(1 / 3)
    assert out["neutral_ratio"] == pytest.approx(1 / 3)


def test_aggregate_sentiment_unknown_method_raises() -> None:
    df = pd.DataFrame([{"polarity": 0.0, "subjectivity": 0.0, "sentiment": "neutral"}])
    with pytest.raises(ValueError):
        SentimentAnalyzer.aggregate_sentiment(df, method="bogus")


def test_sentiment_analyzer_backend_selection() -> None:
    analyzer = SentimentAnalyzer()
    assert analyzer.backend in {"huggingface", "textblob", "none"}