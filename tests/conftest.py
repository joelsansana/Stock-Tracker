"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def close_series() -> pd.Series:
    """100 trading days of synthetic close prices with known structure."""
    rng = np.random.default_rng(seed=42)
    base = 100 + np.cumsum(rng.normal(0, 1, 100))
    return pd.Series(base, name="Close")