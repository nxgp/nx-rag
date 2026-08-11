import json
import logging
import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException
from mentera_rag.utils.logging import StructuredJSONFormatter, request_id_var
from mentera_rag.utils.rate_limit import TokenBucketRateLimiter


@pytest.mark.unit
def test_json_log_formatter():
    """Verify StructuredJSONFormatter formats logs as single-line JSON with context fields."""
    formatter = StructuredJSONFormatter()
    token = request_id_var.set("test-request-uuid-123")

    try:
        log_record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_file.py",
            lineno=42,
            msg="System health check: %s",
            args=("OK",),
            exc_info=None,
        )

        formatted = formatter.format(log_record)
        log_data = json.loads(formatted)

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "System health check: OK"
        assert log_data["request_id"] == "test-request-uuid-123"
        assert "timestamp" in log_data
    finally:
        request_id_var.reset(token)


@pytest.mark.unit
def test_rate_limiter_allows_and_blocks():
    """Verify TokenBucketRateLimiter blocks requests when bucket is exhausted."""
    # Limiter with capacity of 2.0 and replenishment rate of 0.0 (won't replenish in test time)
    limiter = TokenBucketRateLimiter(rate=0.0, capacity=2.0)

    # 1. First request allowed
    limiter.check("tenant_a")

    # 2. Second request allowed
    limiter.check("tenant_a")

    # 3. Third request throws 429 error (capacity is 2.0)
    with pytest.raises(HTTPException) as exc_info:
        limiter.check("tenant_a")

    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in exc_info.value.detail

    # Verify a different tenant is still allowed (separate bucket)
    limiter.check("tenant_b")
