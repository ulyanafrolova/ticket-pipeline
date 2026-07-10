import functools
import logging
import random
import time

logger = logging.getLogger(__name__)

def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,)
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    if attempt < max_attempts - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        if jitter:
                            delay += random.uniform(0, 1.0)
                        logger.warning(
                            "retry: %s attempt %d/%d failed: %s. Retrying in %.2fs",
                            func.__name__, attempt + 1, max_attempts, e, delay,
                        )
                        time.sleep(delay)
                    else:
                        raise
        return wrapper
    return decorator
