"""Tests for the sentiment_models helper module."""

from __future__ import annotations

from app.components import sentiment_models


def test_list_presets_has_default_and_finbert() -> None:
    presets = sentiment_models.list_presets()
    assert "default" in presets
    assert "finbert" in presets


def test_preset_model_id_resolves_known() -> None:
    assert (
        sentiment_models.preset_model_id("default")
        == "distilbert-base-uncased-finetuned-sst-2-english"
    )
    assert (
        sentiment_models.preset_model_id("finbert")
        == "ProsusAI/finbert"
    )


def test_preset_model_id_passes_through_unknown() -> None:
    """Unknown names should be returned as-is."""
    assert (
        sentiment_models.preset_model_id("some-user/some-model")
        == "some-user/some-model"
    )


def test_hf_available_returns_bool() -> None:
    """Always returns a bool, regardless of whether transformers/torch are installed."""
    assert isinstance(sentiment_models.hf_available(), bool)