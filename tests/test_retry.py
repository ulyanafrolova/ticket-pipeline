"""Unit tests for src/retry.py — retry decorator with exponential backoff."""

from unittest.mock import patch

import pytest

from src.retry import retry


def test_retries_on_retryable():
    calls = {"count": 0}

    @retry(max_attempts=3, base_delay=0.01)
    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("transient")
        return "ok"

    with patch("src.retry.time.sleep"):
        assert flaky() == "ok"
    assert calls["count"] == 3


def test_reraises_on_last_attempt():
    calls = {"count": 0}

    @retry(max_attempts=3, base_delay=0.01)
    def always_fails():
        calls["count"] += 1
        raise ValueError("permanent")

    with patch("src.retry.time.sleep"), pytest.raises(ValueError, match="permanent"):
        always_fails()
    assert calls["count"] == 3


def test_no_retry_on_non_retryable():
    calls = {"count": 0}

    @retry(max_attempts=3, base_delay=0.01, retryable_exceptions=(ValueError,))
    def wrong_type():
        calls["count"] += 1
        raise TypeError("not retryable")

    with patch("src.retry.time.sleep") as mock_sleep, pytest.raises(TypeError):
        wrong_type()
    assert calls["count"] == 1
    mock_sleep.assert_not_called()


def test_exponential_backoff():
    @retry(max_attempts=3, base_delay=1.0, max_delay=30.0, jitter=False)
    def always_fails():
        raise ValueError("boom")

    with patch("src.retry.time.sleep") as mock_sleep, pytest.raises(ValueError):
        always_fails()

    delays = [call.args[0] for call in mock_sleep.call_args_list]
    assert delays == [1.0, 2.0], "delay must double each attempt"


def test_jitter_adds_randomness():
    # max_delay caps the exponential term at 1.0 for both attempts, so any
    # difference between the two delays comes from the jitter alone.
    @retry(max_attempts=3, base_delay=1.0, max_delay=1.0, jitter=True)
    def always_fails():
        raise ValueError("boom")

    with patch("src.retry.time.sleep") as mock_sleep, pytest.raises(ValueError):
        always_fails()

    delays = [call.args[0] for call in mock_sleep.call_args_list]
    assert len(delays) == 2
    assert delays[0] != delays[1], "two successive delays must not be identical"


def test_wraps_preserves_name():
    @retry(max_attempts=2)
    def my_special_function():
        return 42

    assert my_special_function.__name__ == "my_special_function"
