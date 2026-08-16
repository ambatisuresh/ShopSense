"""
Retry + circuit breaker + call logging, wrapping every tool the same way the
Session 2 lab wrapped calculator/weather/currency/calendar.

TOOL_CALL_LOG entries are shaped so M4 can route them into Langfuse spans
without reformatting: {tool, args, kwargs, ts, result|error}.
"""
from __future__ import annotations

import functools
import random
import time
from typing import Any

TOOL_CALL_LOG: list[dict[str, Any]] = []


def retry_with_backoff(max_retries: int = 3, base_delay: float = 0.5):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


class CircuitBreakerOpen(Exception):
    pass


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self._consecutive_failures: dict[str, int] = {}
        self._open: set[str] = set()

    def before_call(self, tool_name: str) -> None:
        if tool_name in self._open:
            raise CircuitBreakerOpen(
                f"Circuit open for '{tool_name}' after {self.failure_threshold} consecutive failures."
            )

    def record_success(self, tool_name: str) -> None:
        self._consecutive_failures[tool_name] = 0

    def record_failure(self, tool_name: str) -> None:
        n = self._consecutive_failures.get(tool_name, 0) + 1
        self._consecutive_failures[tool_name] = n
        if n >= self.failure_threshold:
            self._open.add(tool_name)


CIRCUIT_BREAKER = CircuitBreaker(failure_threshold=3)


def make_robust_tool(fn, name: str):
    """Wraps fn with circuit breaker + retry + call logging, in that order."""
    retried = retry_with_backoff(max_retries=3)(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        CIRCUIT_BREAKER.before_call(name)
        entry: dict[str, Any] = {"tool": name, "args": args, "kwargs": kwargs, "ts": time.time()}
        try:
            result = retried(*args, **kwargs)
            CIRCUIT_BREAKER.record_success(name)
            entry["result"] = result
            TOOL_CALL_LOG.append(entry)
            return result
        except Exception as e:
            CIRCUIT_BREAKER.record_failure(name)
            entry["error"] = str(e)
            TOOL_CALL_LOG.append(entry)
            raise

    return wrapper


def clear_log() -> None:
    TOOL_CALL_LOG.clear()