"""Per-user sliding-window rate limiting for the VinBank pipeline."""
from __future__ import annotations

from collections import defaultdict, deque
import time

from core.adk_compat import base_plugin, types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Stop burst abuse before it consumes model or review capacity."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        if max_requests < 1 or window_seconds < 1:
            raise ValueError("max_requests and window_seconds must both be positive")
        super().__init__(name="rate_limiter")
        self.max_requests = int(max_requests)
        self.window_seconds = int(window_seconds)
        self.user_windows: dict[str, deque[float]] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0

    @staticmethod
    def _block_response(message: str) -> types.Content:
        """Return an ADK replacement response without invoking the model."""
        return types.Content(
            role="model", parts=[types.Part.from_text(text=message)]
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Allow at most ``max_requests`` events per user in a rolling window."""
        self.total_count += 1
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = time.time()
        window = self.user_windows[str(user_id)]
        cutoff = now - self.window_seconds

        # Expire only timestamps outside the inclusive rolling window.
        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= self.max_requests:
            retry_after = max(0.0, self.window_seconds - (now - window[0]))
            self.blocked_count += 1
            return self._block_response(
                f"Rate limit exceeded. Try again in {retry_after:.0f}s."
            )

        window.append(now)
        return None
