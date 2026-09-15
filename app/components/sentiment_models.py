"""Cached Hugging Face pipeline loader for the Streamlit UI.

Wraps :class:`python_stocks.sentiment.HuggingFaceSentiment` with
``@st.cache_resource`` so the model is loaded once per server process
and reused across reruns. The load is slow (downloading weights) and
the pipeline itself isn't picklable, so caching by resource id is the
right tool.

Two model ids are pre-registered:

* ``"default"`` — DistilBERT SST-2 (small, English, general sentiment).
* ``"finbert"`` — ProsusAI/FinBERT (finance-tuned). Only useful if the
  user has installed the ``[transformers]`` extra.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from python_stocks.sentiment import HuggingFaceSentiment

logger = logging.getLogger(__name__)


PRESETS: dict[str, str] = {
    "default": "distilbert-base-uncased-finetuned-sst-2-english",
    "finbert": "ProsusAI/finbert",
}


def hf_available() -> bool:
    """True if the transformers/torch deps are importable."""
    import importlib.util

    return (
        importlib.util.find_spec("transformers") is not None
        and importlib.util.find_spec("torch") is not None
    )


@st.cache_resource(show_spinner="Loading model…")
def load_hf_pipeline(model_id: str, device: int = -1) -> HuggingFaceSentiment:
    """Load (or reuse) a Hugging Face sentiment pipeline.

    Args:
        model_id: Hugging Face model id, or one of the keys of
            :data:`PRESETS` (``"default"`` or ``"finbert"``).
        device: ``-1`` for CPU, ``0`` for first GPU, etc.

    Returns:
        A :class:`HuggingFaceSentiment` instance. Its ``available``
        attribute is ``False`` if loading failed.
    """
    resolved = PRESETS.get(model_id, model_id)
    return HuggingFaceSentiment(model=resolved, device=device)


def list_presets() -> list[str]:
    """Names of preset models. Useful for populating selectboxes."""
    return list(PRESETS.keys())


def preset_model_id(name: str) -> str:
    """Resolve a preset name to its Hugging Face id, or pass through."""
    return PRESETS.get(name, name)


def score_dataframe(
    pipeline: HuggingFaceSentiment,
    texts: list[str],
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Score ``texts`` via the cached pipeline, returning a list of dicts.

    Wraps :meth:`HuggingFaceSentiment.analyze_batch` so callers don't
    have to know the DataFrame layout.
    """
    if not pipeline.available:
        return [
            {"label": "NEUTRAL", "score": 0.5, "text": t[:100]}
            for t in texts
        ]
    df = pipeline.analyze_batch(texts, batch_size=batch_size)
    return df.to_dict(orient="records")