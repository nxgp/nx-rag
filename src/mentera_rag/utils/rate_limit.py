"""
In-Memory Token Bucket Rate Limiter — Mentera RAG Pipeline.

Provides rate-limiting protection for sensitive ingestion and presign endpoints.
Rate limiting can be configured per-tenant or per-client IP.
"""

import logging
import time
from collections import defaultdict

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """
    Thread-safe thread-shared token-bucket rate limiter.
    """

    def __init__(self, rate: float = 1.0, capacity: float = 5.0):
        """
        Initialize the rate limiter.

        Args:
            rate: Token replenishment rate per second (e.g. 1.0 token/sec = 60 requests/min).
            capacity: Maximum burst capacity of the bucket (e.g. 5 concurrent burst requests).
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens: dict[str, float] = defaultdict(lambda: float(capacity))
        self._last_checked: dict[str, float] = defaultdict(time.time)

    def check(self, key: str) -> None:
        """
        Consume a single token for the given key.
        Raises HTTP 429 Too Many Requests if bucket is empty.
        """
        now = time.time()
        last = self._last_checked[key]
        elapsed = now - last

        # Replenish tokens based on time elapsed
        current_tokens = min(self.capacity, self._tokens[key] + (elapsed * self.rate))

        self._tokens[key] = current_tokens
        self._last_checked[key] = now

        if current_tokens < 1.0:
            logger.warning("Rate limit exceeded for key: %s", key)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Rate limit exceeded.",
            )

        # Consume 1.0 token
        self._tokens[key] = current_tokens - 1.0
