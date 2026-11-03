"""Choice between the NumPy reference code and the optional compiled kernels.

The compiled kernels (package ``particlelab_fast``, built from ``rust/``) are used
when they are installed and match this version; otherwise everything runs in NumPy.
Both give the same hits and fits for the same random numbers (tested in
tests/unit/test_fast_backend.py). Force a choice with the environment variable
PARTICLELAB_BACKEND=numpy|rust or with ``use("numpy")``.
"""
from __future__ import annotations

import os
import warnings

API_VERSION = 1

try:
    import particlelab_fast as fast
    if getattr(fast, "API_VERSION", None) != API_VERSION:
        warnings.warn(f"particlelab_fast has API {getattr(fast, 'API_VERSION', None)}, expected {API_VERSION}; "
                      "using NumPy. Rebuild it from rust/ (see the README).")
        fast = None
except ImportError:
    fast = None

_choice = os.environ.get("PARTICLELAB_BACKEND", "auto").strip().lower()


def available() -> bool:
    return fast is not None


def active() -> str:
    """'rust' or 'numpy'."""
    return "rust" if fast is not None and _choice != "numpy" else "numpy"


def use(name: str) -> None:
    """Select 'numpy', 'rust' or 'auto' for the rest of the session."""
    global _choice
    name = name.lower()
    if name not in ("numpy", "rust", "auto"):
        raise ValueError(name)
    if name == "rust" and fast is None:
        raise RuntimeError("particlelab_fast is not installed")
    _choice = name
