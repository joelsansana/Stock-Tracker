"""Status page — diagnostic info about this installation."""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path setup; see app/_bootstrap.py)
import streamlit as st

from app.components.cache import data_dir, last_fetch_error
from app.components.shared import backend_status, package_version, sidebar_nav
from python_stocks import StockDataFetcher, StockFetchError


def main() -> None:
    st.set_page_config(page_title="Status — python_stocks", layout="wide")
    sidebar_nav()
    st.title("Status")

    status = backend_status()
    env_cols = st.columns(2)
    env_cols[0].markdown("### Runtime")
    env_cols[0].write(f"- Python `{sys.version.split()[0]}`")
    env_cols[0].write(f"- Platform `{platform.platform()}`")
    env_cols[0].write(f"- python_stocks `{package_version()}`")

    env_cols[1].markdown("### Optional backends")
    for name, ok in status.items():
        env_cols[1].write(f"- **{name}**: {'✅' if ok else '❌'}")

    st.divider()
    _render_yfinance_probe()

    st.divider()
    st.markdown("### Data directory")
    d = data_dir()
    st.write(f"- Path: `{d.resolve()}`")

    csvs = sorted(d.glob("*.csv")) if d.exists() else []
    if csvs:
        rows = []
        for p in csvs:
            stat = p.stat()
            rows.append(
                {
                    "file": p.name,
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified": pd_mtime(stat.st_mtime),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No cached CSVs yet.")

    st.divider()
    st.markdown("### Maintenance")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Clear in-memory cache", help="Forget all @st.cache_data results. Disk cache untouched."):
        st.cache_data.clear()
        st.toast("In-memory cache cleared", icon="🧹")
        st.rerun()
    if c2.button(
        "Delete ARK CSVs", help="Remove cached ARK ETF CSV files. They will be re-downloaded on next visit."
    ):
        removed = 0
        for p in d.glob("*_holdings.csv"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
        st.toast(f"Deleted {removed} CSV file(s)", icon="🗑️")
        st.rerun()
    if c3.button("Show data path", help="Show the absolute path of the data directory."):
        st.code(str(d.resolve()), language="text")
    if c4.button(
        "Clear yfinance cookie cache",
        help="Drop the persistent yfinance cookie/crumb cache. Use this if Yahoo "
        "Finance has rotated its auth and the cached cookie is now invalid.",
    ):
        cleared = _clear_yfinance_cookie_cache()
        if cleared:
            st.toast("yfinance cookie cache cleared", icon="🍪")
        else:
            st.toast("yfinance cookie cache not found", icon="🍪")
        st.rerun()


def _render_yfinance_probe() -> None:
    """Run a single yfinance fetch against AAPL and report what comes back.

    This deliberately bypasses the Streamlit cache so each click reflects
    the current state of the upstream. The result tells us whether the
    failure mode is "exception", "empty response", or "OK".
    """
    st.markdown("### yfinance connectivity test")
    st.caption(
        "Bypasses the Streamlit cache so each click reflects the current "
        "state of Yahoo Finance from this server. Useful when the Stock "
        "Analysis page reports 'No data' — run this to see the actual "
        "underlying error."
    )
    if st.button("Test yfinance (AAPL, 5d, 1d)"):
        fetcher = StockDataFetcher(data_dir=data_dir())
        try:
            t0 = time.perf_counter()
            try:
                df = fetcher.get_price("AAPL", period="5d", interval="1d", max_retries=1)
            except StockFetchError as exc:
                elapsed = time.perf_counter() - t0
                st.error(f"Failed in {elapsed:.2f}s: {exc}")
                st.caption(
                    f"attempts={exc.attempts}  empty_response={exc.empty}  "
                    f"last_exception={type(exc.last_exc).__name__ if exc.last_exc else '—'}"
                )
                if exc.last_exc is not None:
                    with st.expander("Last exception", expanded=False):
                        st.code(repr(exc.last_exc))
            else:
                elapsed = time.perf_counter() - t0
                st.success(f"Fetched {len(df)} rows in {elapsed:.2f}s")
                st.dataframe(df.tail(5), use_container_width=True)
        finally:
            fetcher.close()
    else:
        err = last_fetch_error("AAPL", "1y", "1d")
        if err is not None:
            st.warning(
                f"Last cached fetch for AAPL/1y/1d failed: {err}",
                icon="⚠️",
            )


def _clear_yfinance_cookie_cache() -> bool:
    """Remove yfinance's persistent cookie/crumb cache file. Returns True on success."""
    try:
        from yfinance import cache as yf_cache
    except ImportError:
        return False
    try:
        cookie_cache = yf_cache.get_cookie_cache()
    except Exception:
        return False
    removed = False
    # yfinance exposes the cache DB location via attribute; fall back to a
    # best-effort search of platformdirs if not available.
    for attr in ("location", "path", "_db_path"):
        path = getattr(cookie_cache, attr, None)
        if path:
            try:
                Path(path).unlink(missing_ok=True)
                removed = True
            except OSError:
                pass
    return removed


def pd_mtime(ts: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")  # noqa: DTZ006


if __name__ == "__main__":
    main()