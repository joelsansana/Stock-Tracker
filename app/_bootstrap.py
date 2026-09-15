"""Sys.path bootstrap for the Streamlit UI.

This module is a side-effect import that puts the project root on
``sys.path`` so the ``app.*`` imports in each page resolve regardless
of the current working directory. See ``app/_bootstrap_inline`` for
the importable snippet every page uses.

Why a separate module?
    Streamlit runs each page as ``__main__`` and only adds the script's
    directory (``app/`` or ``app/pages/``) to ``sys.path``. This means
    ``from app.components.shared import ...`` only works when the user
    launches Streamlit from the project root — fragile when run from
    an IDE, a launcher, or any other working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT_STR = str(_PROJECT_ROOT)

if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)

__all__: list[str] = []  # side-effect module