"""Load system prompt text files from the magpie.prompts package."""

from __future__ import annotations

from functools import lru_cache

try:
    from importlib.resources import files as _files
except ImportError:
    from importlib_resources import files as _files  # type: ignore[import-not-found]


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Return the text of ``magpie/prompts/{name}.txt`` as a string."""
    return _files("magpie.prompts").joinpath(f"{name}.txt").read_text(encoding="utf-8").strip()
