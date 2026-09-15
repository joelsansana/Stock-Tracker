"""Reusable Plotly wrappers for the Streamlit UI.

Kept dependency-light: only ``plotly`` is required, which is pulled in
via the ``[ui]`` extra of ``python-stocks``. Nothing in this module
imports Streamlit itself, so it can be unit-tested without a Streamlit
runtime.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Columns we try in order to find a tickers-style label for holdings
# charts. The first one present wins.
_TICKER_CANDIDATES = ("ticker", "symbol", "cusip")
# Fallback: human-readable names.
_NAME_CANDIDATES = ("company", "name", "fund name", "holding")


def _pick_label_col(df: pd.DataFrame, value_col: str, explicit: Optional[str]) -> Optional[str]:
    """Resolve the label column for a holdings chart.

    Preference order:

    1. ``explicit`` (caller override).
    2. A ticker-style column (``ticker`` / ``symbol`` / ``cusip``).
    3. A human-readable name column.
    4. ``None`` — caller should fall back to the index.
    """
    if explicit is not None and explicit in df.columns:
        return explicit
    for cand in _TICKER_CANDIDATES:
        if cand in df.columns and cand != value_col:
            return cand
    for cand in _NAME_CANDIDATES:
        if cand in df.columns and cand != value_col:
            return cand
    return None


def _ensure_label(df: pd.DataFrame, value_col: str, name_col: Optional[str]) -> tuple[pd.DataFrame, str]:
    """Return ``(df, name_col)`` ready for plotting.

    If no usable label column exists, fabricate one from the index so
    Plotly always has something to label with.
    """
    if name_col is None:
        df = df.copy()
        df["__name__"] = df.index.astype(str)
        return df, "__name__"
    return df, name_col


def _aggregate_for_chart(df: pd.DataFrame, label_col: str, value_col: str) -> pd.DataFrame:
    """Aggregate a holdings frame so every label is unique.

    ARK CSVs can include several rows with the same ticker (or NaN
    tickers) for different share classes, warrants, and private
    placements. Plotly's hierarchical charts (treemap) treat duplicate
    labels as non-leaf internal nodes and raise an error. We sum the
    value column per label and drop rows missing the label entirely.
    """
    work = df[[label_col, value_col]].copy()
    work = work.dropna(subset=[label_col])
    work = work.groupby(label_col, as_index=False, dropna=False)[value_col].sum()
    return work


def holdings_bar(df: pd.DataFrame, value_col: str = "weight", name_col: Optional[str] = None) -> go.Figure:
    """Horizontal bar chart of holdings sorted by weight.

    Labels default to the ticker column (``ticker`` / ``symbol`` /
    ``cusip``) when present, otherwise the company name, otherwise the
    index. Rows with NaN labels are dropped; rows with duplicate
    labels are summed.

    Args:
        df: Holdings DataFrame.
        value_col: Column holding the weight (or market value).
        name_col: Optional explicit override for the label column.
    """
    name_col = _pick_label_col(df, value_col, name_col)
    df, name_col = _ensure_label(df, value_col, name_col)
    df = _aggregate_for_chart(df, name_col, value_col)

    sorted_df = df.sort_values(value_col, ascending=True)
    fig = px.bar(
        sorted_df,
        x=value_col,
        y=name_col,
        orientation="h",
        height=min(800, 30 * len(sorted_df) + 100),
    )
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis_title=None,
        xaxis_title=value_col.title(),
    )
    return fig


def holdings_treemap(df: pd.DataFrame, value_col: str = "weight", name_col: Optional[str] = None) -> go.Figure:
    """Treemap of holdings sized by ``value_col``.

    Same label-resolution rules as :func:`holdings_bar`. Aggregation
    is required here because Plotly's hierarchical charts treat
    duplicate path labels as internal nodes and refuse to render.
    """
    name_col = _pick_label_col(df, value_col, name_col)
    df, name_col = _ensure_label(df, value_col, name_col)
    df = _aggregate_for_chart(df, name_col, value_col)

    fig = px.treemap(df, path=[name_col], values=value_col, height=500)
    fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10})
    return fig


def price_chart(
    df: pd.DataFrame,
    overlays: Iterable[tuple[str, str]] = (),
    show_volume: bool = True,
) -> go.Figure:
    """Candlestick + optional indicator overlays + optional volume row.

    Args:
        df: OHLCV DataFrame indexed by date.
        overlays: Iterable of ``(column, label)`` pairs to draw on the
            price row, e.g. ``[("SMA_20", "SMA 20")]``.
        show_volume: If True and ``Volume`` is in ``df``, draw a volume
            subplot beneath.
    """
    rows = 2 if show_volume and "Volume" in df.columns else 1
    row_heights = [0.75, 0.25] if rows == 2 else [1.0]
    specs = [[{"secondary_y": False}]] * rows

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
        specs=specs,
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    for col, label in overlays:
        if col in df.columns and not df[col].isna().all():
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], mode="lines", name=label),
                row=1,
                col=1,
            )

    if rows == 2:
        colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(df["Open"], df["Close"])]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume", showlegend=False),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=600,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        legend_y=1.02,
        legend_x=0,
    )
    return fig


def indicator_subplot(df: pd.DataFrame, columns: list[str], title: str = "") -> go.Figure:
    """Single-row line chart for one or more indicator columns."""
    fig = go.Figure()
    for col in columns:
        if col in df.columns and not df[col].isna().all():
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(
        title=title,
        height=300,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend_orientation="h",
        legend_y=1.1,
    )
    return fig


def normalized_compare(dfs: dict[str, pd.DataFrame]) -> go.Figure:
    """Plot multiple tickers rebased to 100 on their first observation."""
    fig = go.Figure()
    for ticker, df in dfs.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        normalized = df["Close"] / df["Close"].iloc[0] * 100.0
        fig.add_trace(go.Scatter(x=df.index, y=normalized, mode="lines", name=ticker))
    fig.update_layout(
        title="Normalized price (start = 100)",
        height=500,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis_title="Index",
        legend_orientation="h",
        legend_y=1.02,
    )
    return fig