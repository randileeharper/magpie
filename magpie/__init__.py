"""Magpie package."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("magpie")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0+local"

