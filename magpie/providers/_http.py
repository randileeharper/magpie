"""Shared HTTP helpers for provider clients.

Provides retry-with-backoff for transient HTTP failures (429 / 5xx) so that a
single rate-limit or server hiccup does not fail an entire research run.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)

# Status codes that are worth retrying: 429 (rate limited) and 5xx (server
# errors). 4xx (except 429) are permanent client errors and must not be retried.
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


def is_retryable_status(status_code: int) -> bool:
    """Return True for transient HTTP statuses that warrant a retry."""
    return status_code in _RETRYABLE_STATUSES


def request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_attempts: int,
    backoff_seconds: float,
    **kwargs: Any,
) -> httpx.Response:
    """Issue an HTTP request, retrying on 429/5xx with exponential backoff.

    *max_attempts* is the total number of attempts (1 = no retry).  The backoff
    between attempts is ``backoff_seconds * 2**(attempt - 1)`` (1×, 2×, 4× …).

    Connection-level errors (``httpx.TransportError``) are also retried, since
    they are typically transient.  Permanent HTTP errors (4xx except 429) are
    returned immediately so the caller can raise the appropriate domain error.
    """
    last_response: httpx.Response | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            if attempt < max_attempts and backoff_seconds > 0:
                delay = backoff_seconds * (2 ** (attempt - 1))
                LOGGER.debug("HTTP %s %s transport error (attempt %d/%d): %s — retrying in %.2fs",
                             method, url, attempt, max_attempts, exc, delay)
                time.sleep(delay)
                continue
            raise
        if not is_retryable_status(response.status_code):
            return response
        last_response = response
        if attempt < max_attempts and backoff_seconds > 0:
            delay = backoff_seconds * (2 ** (attempt - 1))
            LOGGER.debug("HTTP %s %s returned %d (attempt %d/%d) — retrying in %.2fs",
                         method, url, response.status_code, attempt, max_attempts, delay)
            time.sleep(delay)
    # All attempts exhausted; return the last retryable response so the caller
    # can raise a domain-specific error with the status code.
    assert last_response is not None
    return last_response
