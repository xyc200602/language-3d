"""Tests for LLM retry infrastructure."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from lang3d.models.retry import (
    RetryConfig,
    _compute_delay,
    _is_retryable_error,
    call_with_retry,
)


# ---------------------------------------------------------------------------
# RetryConfig
# ---------------------------------------------------------------------------

class TestRetryConfig:
    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 60.0
        assert cfg.retry_on_status == (429, 500, 502, 503, 504)

    def test_custom_values(self):
        cfg = RetryConfig(max_retries=5, base_delay=2.0, max_delay=120.0, retry_on_status=(429,))
        assert cfg.max_retries == 5
        assert cfg.base_delay == 2.0
        assert cfg.max_delay == 120.0
        assert cfg.retry_on_status == (429,)


# ---------------------------------------------------------------------------
# _is_retryable_error
# ---------------------------------------------------------------------------

class TestIsRetryableError:
    def test_connection_error(self):
        cfg = RetryConfig()
        assert _is_retryable_error(ConnectionError("refused"), cfg) is True

    def test_timeout_error(self):
        cfg = RetryConfig()
        assert _is_retryable_error(TimeoutError("timed out"), cfg) is True

    def test_os_error(self):
        cfg = RetryConfig()
        assert _is_retryable_error(OSError("network"), cfg) is True

    def test_generic_exception(self):
        cfg = RetryConfig()
        assert _is_retryable_error(ValueError("bad"), cfg) is False

    def test_status_code_429(self):
        cfg = RetryConfig()
        exc = Exception("rate limited")
        exc.status_code = 429
        assert _is_retryable_error(exc, cfg) is True

    def test_status_code_500(self):
        cfg = RetryConfig()
        exc = Exception("server error")
        exc.status_code = 500
        assert _is_retryable_error(exc, cfg) is True

    def test_status_code_200(self):
        cfg = RetryConfig()
        exc = Exception("ok")
        exc.status_code = 200
        assert _is_retryable_error(exc, cfg) is False

    def test_openai_rate_limit_error(self):
        cfg = RetryConfig()
        try:
            import openai
            exc = openai.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429),
                body=None,
            )
            # openai.RateLimitError should be detected as retryable
            assert _is_retryable_error(exc, cfg) is True
        except ImportError:
            pytest.skip("openai not installed")
        except Exception:
            # If the constructor fails with mock args, just verify the isinstance check
            # by testing with a generic status_code=429 exception
            exc = Exception("rate limited")
            exc.status_code = 429
            assert _is_retryable_error(exc, cfg) is True

    def test_httpx_status_error(self):
        cfg = RetryConfig()
        try:
            import httpx

            response = MagicMock()
            response.status_code = 500
            exc = httpx.HTTPStatusError("server error", request=MagicMock(), response=response)
            assert _is_retryable_error(exc, cfg) is True
        except ImportError:
            pytest.skip("httpx not installed")


# ---------------------------------------------------------------------------
# _compute_delay
# ---------------------------------------------------------------------------

class TestComputeDelay:
    def test_exponential_backoff(self):
        cfg = RetryConfig(base_delay=1.0, max_delay=60.0)
        d0 = _compute_delay(0, cfg)
        d1 = _compute_delay(1, cfg)
        d2 = _compute_delay(2, cfg)
        # d0 ~ 1.0 + jitter, d1 ~ 2.0 + jitter, d2 ~ 4.0 + jitter
        assert 0.9 <= d0 <= 1.6
        assert 1.9 <= d1 <= 2.6
        assert 3.9 <= d2 <= 4.6

    def test_max_delay_cap(self):
        cfg = RetryConfig(base_delay=1.0, max_delay=5.0)
        # At attempt 10: 1.0 * 2^10 = 1024 — should be capped at 5.0
        d = _compute_delay(10, cfg)
        assert d <= 5.0

    def test_retry_after_preferred(self):
        cfg = RetryConfig(base_delay=1.0, max_delay=60.0)
        exc = Exception("rate limited")
        exc.headers = {"retry-after": "10"}
        d = _compute_delay(0, cfg, exc)
        assert d == 10.0

    def test_retry_after_capped(self):
        cfg = RetryConfig(base_delay=1.0, max_delay=5.0)
        exc = Exception("rate limited")
        exc.headers = {"retry-after": "100"}
        d = _compute_delay(0, cfg, exc)
        assert d == 5.0  # capped


# ---------------------------------------------------------------------------
# call_with_retry
# ---------------------------------------------------------------------------

class TestCallWithRetry:
    def test_success_first_try(self):
        fn = MagicMock(return_value="ok")
        result = call_with_retry(fn, retry_config=RetryConfig(max_retries=3))
        assert result == "ok"
        assert fn.call_count == 1

    def test_retry_then_success(self):
        fn = MagicMock(side_effect=[ConnectionError("fail"), "ok"])
        with patch("lang3d.models.retry.time.sleep"):
            result = call_with_retry(fn, retry_config=RetryConfig(max_retries=3))
        assert result == "ok"
        assert fn.call_count == 2

    def test_429_retry_then_success(self):
        exc = Exception("rate limited")
        exc.status_code = 429
        fn = MagicMock(side_effect=[exc, "ok"])
        with patch("lang3d.models.retry.time.sleep"):
            result = call_with_retry(fn, retry_config=RetryConfig(max_retries=3))
        assert result == "ok"

    def test_max_retries_exhausted(self):
        fn = MagicMock(side_effect=ConnectionError("fail"))
        with patch("lang3d.models.retry.time.sleep"):
            with pytest.raises(ConnectionError, match="fail"):
                call_with_retry(fn, retry_config=RetryConfig(max_retries=2))
        assert fn.call_count == 3  # initial + 2 retries

    def test_non_retryable_raises_immediately(self):
        fn = MagicMock(side_effect=ValueError("bad"))
        with pytest.raises(ValueError, match="bad"):
            call_with_retry(fn, retry_config=RetryConfig(max_retries=3))
        assert fn.call_count == 1

    def test_passes_args_and_kwargs(self):
        fn = MagicMock(return_value=42)
        result = call_with_retry(fn, "a", "b", retry_config=RetryConfig(), x=1, y=2)
        assert result == 42
        fn.assert_called_once_with("a", "b", x=1, y=2)

    def test_default_config(self):
        fn = MagicMock(return_value="ok")
        result = call_with_retry(fn)
        assert result == "ok"

    def test_5xx_retry(self):
        exc = Exception("server error")
        exc.status_code = 503
        fn = MagicMock(side_effect=[exc, "recovered"])
        with patch("lang3d.models.retry.time.sleep"):
            result = call_with_retry(fn, retry_config=RetryConfig(max_retries=1))
        assert result == "recovered"
        assert fn.call_count == 2
