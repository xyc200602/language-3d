"""Retry infrastructure for LLM API calls.

Provides exponential backoff with jitter for rate limits (429),
server errors (5xx), and connection failures.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behaviour."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504)


def _is_retryable_error(exc: BaseException, config: RetryConfig) -> bool:
    """Return True if *exc* is worth retrying."""
    # --- OpenAI / httpx status errors ---
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return int(status_code) in config.retry_on_status

    # --- openai SDK exception hierarchy ---
    try:
        import openai

        if isinstance(exc, openai.RateLimitError):
            return True
        if isinstance(exc, openai.APITimeoutError):
            return True
        if isinstance(exc, openai.APIConnectionError):
            return True
        if isinstance(exc, openai.InternalServerError):
            return True
    except ImportError:
        pass

    # --- httpx errors (Ollama) ---
    try:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in config.retry_on_status
        if isinstance(exc, httpx.ConnectError):
            return True
        if isinstance(exc, httpx.ReadTimeout):
            return True
        if isinstance(exc, httpx.ConnectTimeout):
            return True
    except ImportError:
        pass

    # --- Generic connection errors ---
    if isinstance(exc, (ConnectionError, OSError, TimeoutError)):
        return True

    return False


def _extract_retry_after(exc: BaseException) -> float | None:
    """Try to read Retry-After from an API error response."""
    # openai SDK stores headers on the exception
    headers = getattr(exc, "headers", None) or {}
    if hasattr(exc, "response"):
        headers = getattr(exc.response, "headers", {}) or {}

    ra = headers.get("retry-after")
    if ra is not None:
        try:
            return float(ra)
        except (ValueError, TypeError):
            pass
    return None


def _compute_delay(attempt: int, config: RetryConfig, exc: BaseException | None = None) -> float:
    """Compute the delay before the next retry."""
    # Prefer server-supplied Retry-After for 429
    if exc is not None:
        retry_after = _extract_retry_after(exc)
        if retry_after is not None:
            return min(retry_after, config.max_delay)

    delay = config.base_delay * (2 ** attempt) + random.uniform(0, 0.5)
    return min(delay, config.max_delay)


def call_with_retry(
    fn: Callable[..., T],
    *args: Any,
    retry_config: RetryConfig | None = None,
    **kwargs: Any,
) -> T:
    """Call *fn* with automatic retry on transient errors.

    Args:
        fn: The callable to invoke.
        *args: Positional arguments forwarded to *fn*.
        retry_config: Retry parameters. Uses defaults if None.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn*.

    Raises:
        The last exception if all retries are exhausted.
    """
    cfg = retry_config or RetryConfig()
    last_exc: BaseException | None = None

    for attempt in range(cfg.max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:
            last_exc = exc
            if attempt >= cfg.max_retries or not _is_retryable_error(exc, cfg):
                raise

            delay = _compute_delay(attempt, cfg, exc)
            logger.warning(
                "Retry %d/%d after %.1fs: %s: %s",
                attempt + 1,
                cfg.max_retries,
                delay,
                type(exc).__name__,
                str(exc)[:200],
            )
            time.sleep(delay)

    # Should not reach here, but satisfy type checker
    raise last_exc  # type: ignore[misc]
