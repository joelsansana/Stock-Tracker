"""ARK ETF Holdings page."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from app.components.cache import data_dir, fetch_ark
from app.components.charts import holdings_bar, holdings_treemap
from app.components.shared import available_etfs, sidebar_nav

# ARK's CSVs have used several column-name variants over the years;
# fall back through candidates so the UI keeps working as they rename.
WEIGHT_CANDIDATES = (
    "weight (%)",
    "weight(%)",
    "weight",
    "% of fund",
    "fund weight",
)
MARKET_VALUE_CANDIDATES = (
    "market value ($)",
    "market value($)",
    "market value",
    "market value (usd)",
)
COMPANY_CANDIDATES = ("company", "name", "fund name", "holding")
TICKER_CANDIDATES = ("ticker", "symbol", "cusip")


def _pick(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lowered = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with stable columns ``ticker``, ``company``,
    ``shares``, ``market_value``, ``weight``."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    col_ticker = _pick(df, TICKER_CANDIDATES)
    col_company = _pick(df, COMPANY_CANDIDATES)
    col_shares = _pick(df, ("shares",))
    col_mv = _pick(df, MARKET_VALUE_CANDIDATES)
    col_weight = _pick(df, WEIGHT_CANDIDATES)

    out = pd.DataFrame()
    out["ticker"] = df[col_ticker] if col_ticker else pd.Series(range(len(df)))
    out["company"] = df[col_company] if col_company else out["ticker"].astype(str)
    out["shares"] = pd.to_numeric(df[col_shares], errors="coerce") if col_shares else pd.NA
    out["market_value"] = pd.to_numeric(
        df[col_mv].astype(str).str.replace(r"[\$,]", "", regex=True),
        errors="coerce",
    ) if col_mv else pd.NA
    out["weight"] = pd.to_numeric(
        df[col_weight].astype(str).str.rstrip("%"),
        errors="coerce",
    ) if col_weight else pd.NA

    return out


def _cache_age(ticker: str) -> str | None:
    path = data_dir() / f"{ticker}_holdings.csv"
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime)  # noqa: DTZ006 (UI uses local time)
    delta = datetime.now() - mtime  # noqa: DTZ005
    if delta.days >= 1:
        return f"{delta.days}d ago"
    hours = delta.seconds // 3600
    if hours >= 1:
        return f"{hours}h ago"
    minutes = max(1, delta.seconds // 60)
    return f"{minutes}m ago"


def main() -> None:
    st.set_page_config(page_title="ARK Holdings — python_stocks", page_icon=":bar_chart:", layout="wide")
    sidebar_nav()
    st.title("ARK ETF Holdings")

    etfs = available_etfs()
    ticker = st.sidebar.selectbox("ETF", etfs, index=0)
    force_refresh = st.sidebar.checkbox("Force refresh", value=False, help="Re-download even if a cached CSV exists.")
    cache_age = _cache_age(ticker)
    if cache_age:
        st.sidebar.caption(f"Cached copy: {cache_age}")

    chart_kind = st.sidebar.radio("Chart", ("Treemap", "Bar"), horizontal=True)

    df = fetch_ark(ticker, force_refresh)
    if df is None or df.empty:
        st.error(f"Could not fetch {ticker}. Check the URL or your network.")
        st.stop()

    norm = _normalize(df)
    if norm["weight"].notna().sum() == 0:
        st.warning(
            "No weight column detected. Showing the raw CSV instead — "
            f"columns: {list(df.columns)}"
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.stop()

    norm = norm.sort_values("weight", ascending=False).reset_index(drop=True)
    top10 = norm.head(10)["weight"].sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("Holdings", len(norm))
    k2.metric("Top-10 weight", f"{top10:.1f}%")
    k3.metric("Cache age", cache_age or "fresh")

    if chart_kind == "Treemap":
        st.plotly_chart(holdings_treemap(norm, value_col="weight"), use_container_width=True)
    else:
        st.plotly_chart(holdings_bar(norm, value_col="weight"), use_container_width=True)

    display = norm.copy()
    display["weight"] = display["weight"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")
    if display["market_value"].dtype.kind in "fiu":
        display["market_value"] = display["market_value"].map(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
        )
    if display["shares"].dtype.kind in "fiu":
        display["shares"] = display["shares"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")

    st.subheader("Holdings")
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "company": st.column_config.TextColumn("Company"),
            "shares": st.column_config.TextColumn("Shares"),
            "market_value": st.column_config.TextColumn("Market value"),
            "weight": st.column_config.TextColumn("Weight"),
        },
    )

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_holdings.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()