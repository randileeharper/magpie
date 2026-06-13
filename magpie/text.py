"""Text normalization for malformed provider output."""

from __future__ import annotations

from typing import Any


def valid_unicode(value: str) -> str:
    """Replace unpaired UTF-16 surrogates while preserving valid Unicode."""
    return value.encode("utf-8", errors="replace").decode("utf-8")


def valid_unicode_tree(value: Any) -> Any:
    """Recursively normalize strings received from external providers."""
    if isinstance(value, str):
        return valid_unicode(value)
    if isinstance(value, list):
        return [valid_unicode_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(valid_unicode_tree(item) for item in value)
    if isinstance(value, dict):
        return {
            valid_unicode(key) if isinstance(key, str) else key: valid_unicode_tree(item)
            for key, item in value.items()
        }
    return value
