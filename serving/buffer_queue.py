"""
BufferQueue: Async-safe bounded deque for buffering completed GRPO groups
between the generator and trainer.

Design goals:
  - Generator (async) calls `await queue.put(group)` after each complete group.
  - Trainer (sync) calls `queue.get_nowait()` or `queue.get_batch(n)` to pull data.
  - maxlen caps staleness: when full the *oldest* group is silently evicted so the
    newest (most on-policy) data always wins. A warning is emitted on each eviction.

Sizing rationale (default maxlen=32):
  - With batch_size=10 groups per training step this is ~3 full training batches.
  - Large enough to absorb brief trainer stalls without blocking the generator.
  - Small enough that data is at most ~3 policy-update versions old before being
    evicted, keeping training close to on-policy.
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger("BufferQueue")


class BufferQueue:
    """
    Thread- and async-safe bounded deque for GRPO training groups.

    Each item is a fully-assembled grouped_record dict:
        {
            "group_id":   str,
            "prompt":     str,
            "prompt_ids": List[int],
            "completions": [ { "completion_ids", "reward", "old_logprobs", "action_mask" }, ... ],
            "metadata":   { "num_completions": int, "timestamp": float }
        }

    Producer (async generator) → put(item)
    Consumer (sync trainer)    → get_nowait() | get_batch(n)
    """

    DEFAULT_MAXLEN = 32  # ~3 batches of 10 groups; tune via constructor

    def __init__(self, maxlen: int = DEFAULT_MAXLEN):
        if maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {maxlen}")
        self.maxlen = maxlen
        self._deque: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

        # Counters for observability
        self._total_put: int = 0
        self._total_get: int = 0
        self._dropped: int = 0

        logger.info(f"[BufferQueue] Initialized. maxlen={maxlen}")

    # ------------------------------------------------------------------
    # Producer API (called from async generator coroutines)
    # ------------------------------------------------------------------

    async def put(self, item: Dict[str, Any]) -> bool:
        """
        Put a completed GRPO group into the queue.

        Because deque(maxlen=N) auto-evicts the *leftmost* (oldest) element
        when full, we just append — the eviction is implicit. We track whether
        an eviction occurred so we can warn the caller.

        Returns:
            True  – item added with no eviction.
            False – item added but one older item was silently dropped.
        """
        with self._lock:
            evicted = len(self._deque) == self.maxlen
            if evicted:
                self._dropped += 1
                logger.warning(
                    "[BufferQueue] Queue full (%d/%d). Oldest group evicted "
                    "(total dropped=%d). Consider slowing generator or speeding trainer.",
                    self.maxlen, self.maxlen, self._dropped,
                )
            self._deque.append(item)
            self._total_put += 1
        return not evicted

    # ------------------------------------------------------------------
    # Consumer API (called from sync trainer / distributed_worker thread)
    # ------------------------------------------------------------------

    def get_nowait(self) -> Optional[Dict[str, Any]]:
        """
        Pop and return the oldest group. Returns None if the queue is empty.
        Non-blocking; safe to call from any thread.
        """
        with self._lock:
            if self._deque:
                item = self._deque.popleft()
                self._total_get += 1
                return item
            return None

    def get_batch(self, n: int) -> List[Dict[str, Any]]:
        """
        Pop and return up to *n* groups in FIFO order.
        Returns an empty list if the queue is empty.
        """
        with self._lock:
            count = min(n, len(self._deque))
            batch = [self._deque.popleft() for _ in range(count)]
            self._total_get += count
            return batch

    def wait_for_batch(
        self,
        n: int,
        poll_interval: float = 0.5,
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Blocking helper for the sync trainer: waits until *n* groups are
        available (or timeout expires) then returns them via get_batch(n).

        Args:
            n:             Number of groups to wait for.
            poll_interval: Seconds between availability checks.
            timeout:       Maximum seconds to wait. None = wait forever.

        Returns:
            List of up to *n* groups (may be fewer if timeout fired).
        """
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        while True:
            with self._lock:
                if len(self._deque) >= n:
                    break
            if deadline is not None and time.monotonic() >= deadline:
                logger.debug("[BufferQueue] wait_for_batch timed out.")
                break
            time.sleep(poll_interval)
        return self.get_batch(n)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of groups in the queue."""
        with self._lock:
            return len(self._deque)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    @property
    def is_full(self) -> bool:
        return self.size == self.maxlen

    def count_available(self) -> int:
        """Alias for size; mirrors the DataLoader API."""
        return self.size

    def stats(self) -> Dict[str, Any]:
        """Return a snapshot of queue statistics for logging / monitoring."""
        with self._lock:
            sz = len(self._deque)
        return {
            "size": sz,
            "maxlen": self.maxlen,
            "fill_pct": round(100.0 * sz / self.maxlen, 1),
            "total_put": self._total_put,
            "total_get": self._total_get,
            "dropped": self._dropped,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"BufferQueue(size={s['size']}/{s['maxlen']} "
            f"[{s['fill_pct']}%], put={s['total_put']}, "
            f"get={s['total_get']}, dropped={s['dropped']})"
        )
