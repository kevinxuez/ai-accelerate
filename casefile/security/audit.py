"""Redacted security-event audit and lightweight in-process rate limiting."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .prompt_guard import (
    GuardDecision,
    fingerprint,
    redact_secrets,
    summarize_untrusted_text,
)


class SecurityAuditor:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

    def record(
        self,
        event: str,
        *,
        decision: GuardDecision | None = None,
        request_id: str | None = None,
        user_id: str | None = None,
        raw_text: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "request_id_sha256": fingerprint(request_id) if request_id else None,
            "user_id": user_id,
        }
        if decision is not None:
            entry["decision"] = decision.to_dict()
        if raw_text is not None:
            entry["content"] = summarize_untrusted_text(raw_text)
        if details:
            entry["details"] = redact_secrets(details)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


class RateLimiter:
    """Fixed-window limiter suitable for the single-process demo service."""

    def __init__(self, requests: int = 60, window_seconds: int = 60) -> None:
        self.requests = requests
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.RLock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] < cutoff:
                events.popleft()
            if len(events) >= self.requests:
                return False
            events.append(now)
            return True
