"""Request-correlated audit logging for security investigation and replay."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4


class AuditLogPlugin:
    """Record decisions without becoming an enforcement layer itself."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    @staticmethod
    def _redact_for_log(text: str) -> str:
        """Never persist detected PII or synthetic secrets in audit evidence."""
        try:
            from guardrails.output_guardrails import content_filter

            return content_filter(str(text or ""))["redacted"]
        except Exception:
            # Logging less is safer than accidentally storing a secret when a
            # dependent filter is unavailable during an incident.
            return "[REDACTED: audit sanitizer unavailable]"

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Open a correlated audit span and return its stable request ID."""
        correlation_id = request_id or f"req-{uuid4().hex}"
        self._open[correlation_id] = {
            "request_id": correlation_id,
            "correlation_id": correlation_id,
            "user_id": str(user_id or "anonymous"),
            "input": self._redact_for_log(text),
            "input_timestamp": utc_now_iso(),
            "started_at": perf_counter(),
        }
        return correlation_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ) -> dict:
        """Close a span with decision, sanitized output, and measured latency."""
        correlation_id = request_id or f"req-{uuid4().hex}"
        opened = self._open.pop(correlation_id, None)
        started_at = opened.pop("started_at", None) if opened else None
        latency_ms = (
            round((perf_counter() - started_at) * 1000, 2)
            if started_at is not None
            else None
        )
        record = {
            "request_id": correlation_id,
            "correlation_id": correlation_id,
            "user_id": str(user_id or (opened or {}).get("user_id") or "anonymous"),
            "input": (opened or {}).get("input"),
            "input_timestamp": (opened or {}).get("input_timestamp"),
            "output": self._redact_for_log(text),
            "timestamp": utc_now_iso(),
            "blocked": bool(blocked),
            "decision": "blocked" if blocked else "allowed",
            "layer": layer,
            "latency_ms": latency_ms,
        }
        self.logs.append(record)
        return record

    @classmethod
    def _sanitize_metadata(cls, value):
        """Recursively sanitize event metadata before it enters an audit log."""
        if isinstance(value, dict):
            return {
                str(key): cls._sanitize_metadata(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._sanitize_metadata(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return cls._redact_for_log(str(value)) if isinstance(value, str) else value
        return cls._redact_for_log(str(value))

    def record_event(
        self,
        *,
        user_id: str,
        request_id: str,
        event_type: str,
        decision: str | None = None,
        layer: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Append a correlated non-terminal event without closing an open span.

        HITL requests need an observable ``queued`` event before a reviewer
        makes a terminal decision.  Metadata is sanitized recursively so the
        event can contain useful context without becoming a secret sink.
        """
        record = {
            "request_id": str(request_id),
            "correlation_id": str(request_id),
            "user_id": str(user_id or "anonymous"),
            "event_type": str(event_type),
            "decision": decision,
            "layer": layer,
            "metadata": self._sanitize_metadata(metadata or {}),
            "timestamp": utc_now_iso(),
        }
        self.logs.append(record)
        return record

    def export_json(self, filepath: str = "outputs/audit_log.json") -> Path:
        """Persist sanitized, inspectable audit records in UTF-8 JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def utc_now_iso() -> str:
    """Return an explicit UTC timestamp suitable for cross-system correlation."""
    return datetime.now(timezone.utc).isoformat()
