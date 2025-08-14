"""Rate limiting for Gemini API requests"""  # noqa: D415

from collections import deque
from contextlib import contextmanager
import time

from .configuration import RateLimitConfig


class RateLimiter:
    """Manages rate limiting for API requests"""  # noqa: D415

    def __init__(self, config: RateLimitConfig):  # noqa: D107
        self.config = config
        self.request_timestamps = deque()

    @contextmanager
    def request_context(self):
        """Context manager for rate-limited API requests"""  # noqa: D415
        self._wait_if_needed()
        try:
            yield
        finally:
            self._record_request()

    def _wait_if_needed(self):
        """Wait if approaching rate limit before making request"""  # noqa: D415
        now = time.time()

        # Remove timestamps older than the rate limit window
        while (
            self.request_timestamps
            and now - self.request_timestamps[0] > self.config.window_seconds
        ):
            self.request_timestamps.popleft()

        # If we're at the limit, wait for the oldest request to age out
        if len(self.request_timestamps) >= self.config.requests_per_minute:
            sleep_time = (
                self.config.window_seconds - (now - self.request_timestamps[0]) + 1
            )
            if sleep_time > 0:
                print(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")  # noqa: T201
                time.sleep(sleep_time)
                # Clean up timestamps after waiting
                now = time.time()
                while (
                    self.request_timestamps
                    and now - self.request_timestamps[0] > self.config.window_seconds
                ):
                    self.request_timestamps.popleft()

    def _record_request(self):
        """Record timestamp of completed request"""  # noqa: D415
        self.request_timestamps.append(time.time())
