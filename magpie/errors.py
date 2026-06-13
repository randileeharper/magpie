"""Magpie error types."""


class MagpieError(Exception):
    """Base error for the project."""


class ConfigError(MagpieError):
    """Raised when configuration is invalid."""


class StorageError(MagpieError):
    """Raised when storage setup or access fails."""


class ResolverError(MagpieError):
    """Raised when the resolver returns unusable data."""


class A2AUnavailableError(MagpieError):
    """Raised when the optional A2A SDK or server is unavailable."""


class A2ARequestError(MagpieError):
    """Raised when an A2A request fails after it may have been accepted."""


class SearchError(MagpieError):
    """Raised when the search provider fails."""


class FetchError(MagpieError):
    """Raised when page fetching fails."""


class WeatherError(MagpieError):
    """Raised when a specialized weather request fails."""


class AnimeError(MagpieError):
    """Raised when a specialized anime request fails."""


class DependencyError(MagpieError):
    """Raised when an optional runtime dependency is unavailable or uninitialized."""


class ResearchCancelled(MagpieError):
    """Raised internally when a durable run cancellation is observed."""
