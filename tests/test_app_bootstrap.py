"""Tests for the Streamlit UI bootstrap.

Verifies that the bootstrap module does what it claims:

1. Imports cleanly when the script's directory is on sys.path
   (the Streamlit case).
2. Adds the project root to sys.path so ``app.components.shared``
   can be imported.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def fresh_sys_path(monkeypatch):
    """Strip sys.path and sys.modules of project-root effects from the test runner.

    Without this, ``_bootstrap`` (already imported by an earlier test or
    by the test runner) is in ``sys.modules`` and its sys.path side-effect
    has already been undone by the path reset — leaving the page imports
    broken.
    """
    project_root = Path(__file__).resolve().parent.parent
    app_dir = project_root / "app"
    cleaned = [p for p in sys.path if Path(p).resolve() not in {project_root, app_dir}]
    monkeypatch.setattr(sys, "path", cleaned)

    # Force a fresh import of _bootstrap so its side-effect fires now.
    monkeypatch.delitem(sys.modules, "_bootstrap", raising=False)

    return project_root, app_dir


def test_bootstrap_importable_from_app_dir(fresh_sys_path, monkeypatch) -> None:
    """``import _bootstrap`` works when sys.path[0] is the app/ dir."""
    _project_root, app_dir = fresh_sys_path
    monkeypatch.setattr(sys, "path", [str(app_dir)] + sys.path)

    # The module is loaded once per process; force a reload.
    if "_bootstrap" in sys.modules:
        del sys.modules["_bootstrap"]
    bootstrap = importlib.import_module("_bootstrap")
    assert bootstrap is not None


def test_bootstrap_adds_project_root(fresh_sys_path, monkeypatch) -> None:
    """After importing the bootstrap, the project root is on sys.path."""
    project_root, app_dir = fresh_sys_path
    monkeypatch.setattr(sys, "path", [str(app_dir)] + sys.path)

    if "_bootstrap" in sys.modules:
        del sys.modules["_bootstrap"]
    importlib.import_module("_bootstrap")

    assert str(project_root) in sys.path


def test_bootstrap_idempotent(fresh_sys_path, monkeypatch) -> None:
    """Importing twice doesn't duplicate the path entry."""
    project_root, app_dir = fresh_sys_path
    monkeypatch.setattr(sys, "path", [str(app_dir)] + sys.path)

    if "_bootstrap" in sys.modules:
        del sys.modules["_bootstrap"]
    importlib.import_module("_bootstrap")
    importlib.import_module("_bootstrap")

    count = sys.path.count(str(project_root))
    assert count == 1


def test_pages_importable_with_only_app_on_path(fresh_sys_path, monkeypatch) -> None:
    """End-to-end: a page module loads with only ``app/`` on sys.path.

    This mirrors the user's reported failure mode and ensures the
    bootstrap + page-import pattern fixes it.
    """
    _, app_dir = fresh_sys_path
    monkeypatch.setattr(sys, "path", [str(app_dir)] + sys.path)

    # Need runpy for the simulation.
    import runpy

    for page in [
        "app/Home.py",
        "app/pages/2_Stock_Analysis.py",
        "app/pages/5_Status.py",
    ]:
        # Skip the network-bound pages for this smoke test.
        path = Path(__file__).resolve().parent.parent / page
        try:
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit:
            # Some pages call st.stop() under specific conditions.
            pass
        except AttributeError as exc:
            # Streamlit raises AttributeError when there's no ScriptRunContext
            # in a bare runpy.run_path() — that's fine for this smoke test.
            if "ScriptRunContext" not in str(exc) and "NoneType" not in str(exc):
                raise
        except (KeyError, RuntimeError, ValueError) as exc:
            # Some pages fail early without streamlit runtime; acceptable.
            if "ScriptRunContext" not in str(exc):
                # Re-raise unexpected exceptions.
                raise