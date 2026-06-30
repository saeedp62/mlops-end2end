"""
mlops_utils._retry
~~~~~~~~~~~~~~~~~~
Internal module providing a retry decorator for Databricks API calls.
"""

import functools
from mlops_utils.logger import get_logger
import time
from typing import Any, Callable, TypeVar

logger = get_logger(__name__)

F = TypeVar('F', bound=Callable[..., Any])

def with_retry(max_attempts: int = 3, backoff_base: float = 2.0) -> Callable[[F], F]:
    """Decorator: retry on transient errors with exponential back-off.

    Parameters
    ----------
    max_attempts:
        Maximum number of attempts before giving up.
    backoff_base:
        Base multiplier for exponential back-off (e.g., 2.0 gives 1, 2, 4 seconds...).
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 1
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt >= max_attempts:
                        logger.error("Function '%s' failed after %d attempts: %s", func.__name__, attempt, exc)
                        raise
                    
                    # Log warning and wait
                    sleep_time = backoff_base ** (attempt - 1)
                    logger.warning(
                        "Function '%s' failed (attempt %d/%d). Retrying in %.1fs... Error: %s",
                        func.__name__, attempt, max_attempts, sleep_time, exc
                    )
                    time.sleep(sleep_time)
                    attempt += 1

        return wrapper  # type: ignore[return-value]
    return decorator
