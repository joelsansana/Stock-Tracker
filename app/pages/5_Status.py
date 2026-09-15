"""Status page — diagnostic info about this installation."""

from __future__ import annotations

import platform
import sys

import streamlit as st

from app.components.cache import data_dir
from app.components.shared import backend_status, package_version, sidebar_nav


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


def pd_mtime(ts: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")  # noqa: DTZ006


if __name__ == "__main__":
    main()