"""Runtime service used by the Streamlit VinBank application.

The Streamlit layer is intentionally thin.  This module owns the protected
agent lifecycle, request-correlated observability, and the deterministic HITL
and egress boundary so the UI cannot accidentally become a second policy
implementation.
"""
from __future__ import annotations

import asyncio
import json
import math
import secrets
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from assignment.audit_log import AuditLogPlugin
from assignment.monitoring import MonitoringAlert
from assignment.pipeline import is_egress_allowed
from core.adk_compat import ADK_AVAILABLE
from core.model_provider import has_openrouter_api_key
from guardrails.output_guardrails import content_filter
from hitl.hitl import ConfidenceRouter, HIGH_RISK_ACTIONS


SAFE_FALLBACK_RESPONSE = (
    "Mình chưa thể xử lý yêu cầu này an toàn lúc này. "
    "Vui lòng thử lại hoặc liên hệ hỗ trợ VinBank."
)
REVIEW_TIMEOUT_SECONDS = 15 * 60


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _now_utc()).isoformat()


def _from_iso(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        # A malformed expiry is treated as already expired (fail closed).
        return datetime.min.replace(tzinfo=timezone.utc)


def _safe_preview(value: Any, limit: int = 600) -> str:
    """Return a short, deterministic, sanitized display/log value."""
    filtered = content_filter(str(value or ""))
    return str(filtered["redacted"] or "")[:limit]


def _run_async(coroutine):
    """Run one service coroutine from Streamlit's synchronous callbacks."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    raise RuntimeError("Use the async runtime method while an event loop is running")


@dataclass(frozen=True)
class RuntimeStatus:
    adk_available: bool
    api_key_configured: bool
    ready: bool
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChatTurn:
    request_id: str
    user_id: str
    input_preview: str
    response: str
    blocked: bool
    layer: str | None
    redacted: bool
    judge_failed: bool
    latency_ms: float
    error_code: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReviewRequest:
    review_id: str
    request_id: str
    user_id: str
    action_type: str
    confidence: float
    route_action: str
    priority: str
    route_reason: str
    context: str
    proposed_diff: str
    destination: str
    destination_safe: bool
    payload: str
    payload_safe: bool
    created_at: str
    expires_at: str
    status: str = "pending"
    reviewer_id: str | None = None
    approval_id: str | None = None
    decision_at: str | None = None
    decision_reason: str | None = None
    egress_allowed: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class VinBankAppSession:
    """One isolated Streamlit browser session.

    The protected ADK runner is created lazily only after the runtime health
    checks pass.  This keeps the Security Console usable on machines that have
    not installed ADK or configured a live OpenRouter key yet.
    """

    def __init__(
        self,
        *,
        user_id: str | None = None,
        use_llm_judge: bool = True,
        review_timeout_seconds: int = REVIEW_TIMEOUT_SECONDS,
    ):
        self.user_id = str(user_id or f"ui-{uuid4().hex}")
        self.use_llm_judge = bool(use_llm_judge)
        self.review_timeout_seconds = max(1, int(review_timeout_seconds))
        self.audit = AuditLogPlugin()
        self.monitor = MonitoringAlert()
        self.router = ConfidenceRouter()
        self.reviews: dict[str, ReviewRequest] = {}
        self._agent = None
        self._runner = None
        self._plugins: list = []
        self._session_id: str | None = None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def status(self) -> RuntimeStatus:
        adk_available = bool(ADK_AVAILABLE)
        api_key_configured = bool(has_openrouter_api_key())
        ready = adk_available and api_key_configured
        if ready:
            message = "Protected agent sẵn sàng sử dụng."
        elif not adk_available:
            message = "Thiếu Google ADK. Cài requirements.txt để bật chat live."
        else:
            message = "Chưa cấu hình OPENROUTER_API_KEY. Đặt key trong .env rồi khởi động lại."
        return RuntimeStatus(adk_available, api_key_configured, ready, message)

    def _ensure_agent(self):
        """Create the protected agent exactly once after health checks."""
        if self._agent is not None and self._runner is not None:
            return
        status = self.status()
        if not status.ready:
            raise RuntimeError("protected_runtime_unavailable")

        from agents.agent import create_protected_agent
        from assignment.pipeline import build_production_plugins

        self._plugins = build_production_plugins(use_llm_judge=self.use_llm_judge)
        self._agent, self._runner = create_protected_agent(self._plugins)

    def _plugin_counters(self) -> dict[str, int]:
        counters = {
            "rate_blocked": 0,
            "input_blocked": 0,
            "output_blocked": 0,
            "output_redacted": 0,
            "judge_checks": 0,
            "judge_failures": 0,
        }
        for plugin in self._plugins:
            name = getattr(plugin, "name", "")
            if name == "rate_limiter":
                counters["rate_blocked"] = int(getattr(plugin, "blocked_count", 0))
            elif name == "input_guardrail":
                counters["input_blocked"] = int(getattr(plugin, "blocked_count", 0))
            elif name == "output_guardrail":
                counters["output_blocked"] = int(getattr(plugin, "blocked_count", 0))
                counters["output_redacted"] = int(getattr(plugin, "redacted_count", 0))
                counters["judge_checks"] = int(getattr(plugin, "judge_checks", 0))
                counters["judge_failures"] = int(getattr(plugin, "judge_failures", 0))
        return counters

    @staticmethod
    def _delta(before: dict[str, int], after: dict[str, int], key: str) -> int:
        return max(0, int(after.get(key, 0)) - int(before.get(key, 0)))

    async def send_chat_async(self, text: str) -> ChatTurn:
        """Run one message through rate limit → input → model → output policy."""
        raw_text = str(text or "")
        started = time.perf_counter()
        request_id = self.audit.record_input(user_id=self.user_id, text=raw_text)
        self.monitor.total_requests += 1
        blocked = False
        layer: str | None = None
        redacted = False
        judge_failed = False
        error_code: str | None = None
        response = SAFE_FALLBACK_RESPONSE

        try:
            self._ensure_agent()
            before = self._plugin_counters()
            from core.utils import chat_with_agent

            response, session = await chat_with_agent(
                self._agent,
                self._runner,
                raw_text,
                session_id=self._session_id,
                user_id=self.user_id,
            )
            self._session_id = getattr(session, "id", self._session_id)
            after = self._plugin_counters()
            rate_delta = self._delta(before, after, "rate_blocked")
            input_delta = self._delta(before, after, "input_blocked")
            output_delta = self._delta(before, after, "output_blocked")
            redaction_delta = self._delta(before, after, "output_redacted")
            judge_delta = self._delta(before, after, "judge_checks")
            judge_failure_delta = self._delta(before, after, "judge_failures")

            if rate_delta:
                blocked, layer = True, "rate_limiter"
            elif input_delta:
                blocked, layer = True, "input_guardrail"
            elif output_delta:
                blocked, layer = True, "output_guardrail"
            elif redaction_delta:
                layer = "output_guardrail"

            self.monitor.rate_limit_hits += rate_delta
            self.monitor.judge_checks += judge_delta
            self.monitor.judge_fails += judge_failure_delta
            judge_failed = bool(judge_failure_delta)
            redacted = bool(redaction_delta)
        except Exception:
            # Never expose provider, prompt, or configuration details to the UI.
            error_code = "protected_runtime_unavailable" if not self.status().ready else "agent_error"
            blocked, layer = True, "runtime"

        filtered = content_filter(str(response or ""))
        if not filtered["safe"]:
            response = filtered["redacted"] or SAFE_FALLBACK_RESPONSE
            redacted = True
            if layer is None:
                layer = "output_guardrail"
        else:
            response = str(filtered["redacted"] or SAFE_FALLBACK_RESPONSE)

        if blocked:
            self.monitor.blocked_requests += 1
        self.monitor.check_metrics()
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        self.audit.record_output(
            user_id=self.user_id,
            text=response,
            blocked=blocked,
            layer=layer,
            request_id=request_id,
        )
        return ChatTurn(
            request_id=request_id,
            user_id=self.user_id,
            input_preview=_safe_preview(raw_text, 800),
            response=response,
            blocked=blocked,
            layer=layer,
            redacted=redacted,
            judge_failed=judge_failed,
            latency_ms=latency_ms,
            error_code=error_code,
        )

    def send_chat(self, text: str) -> ChatTurn:
        return _run_async(self.send_chat_async(text))

    def propose_action(
        self,
        *,
        action_type: str,
        confidence: float,
        context: str,
        proposed_diff: str,
        destination: str,
        payload: str,
    ) -> ReviewRequest:
        """Queue a high-risk action; no side effect occurs at proposal time."""
        normalized_action = str(action_type or "").strip().casefold()
        allowed_actions = {str(action).casefold() for action in HIGH_RISK_ACTIONS}
        if normalized_action not in allowed_actions:
            raise ValueError("Unsupported high-risk action")

        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        score = min(1.0, max(0.0, score))

        route = self.router.route(
            f"Action proposal: {normalized_action}", score, normalized_action
        )
        payload_check = content_filter(str(payload or ""))
        destination_check = content_filter(str(destination or ""))
        created_at = _now_utc()
        expires_at = created_at + timedelta(seconds=self.review_timeout_seconds)
        request_id = self.audit.record_input(
            user_id=self.user_id,
            text=f"Action proposal {normalized_action}: {context}; {proposed_diff}",
        )
        self.monitor.total_requests += 1
        review = ReviewRequest(
            review_id=f"review-{uuid4().hex[:12]}",
            request_id=request_id,
            user_id=self.user_id,
            action_type=normalized_action,
            confidence=score,
            route_action=route.action,
            priority=route.priority,
            route_reason=route.reason,
            context=_safe_preview(context),
            proposed_diff=_safe_preview(proposed_diff),
            destination=_safe_preview(destination, 500),
            destination_safe=bool(destination_check["safe"]),
            payload=_safe_preview(payload, 800),
            payload_safe=bool(payload_check["safe"]),
            created_at=_iso(created_at),
            expires_at=_iso(expires_at),
        )
        self.reviews[review.review_id] = review
        self.audit.record_event(
            user_id=self.user_id,
            request_id=request_id,
            event_type="hitl_queued",
            decision="pending",
            layer="hitl",
            metadata={
                "review_id": review.review_id,
                "action_type": review.action_type,
                "confidence": review.confidence,
                "route": review.route_action,
                "priority": review.priority,
                "context": review.context,
                "proposed_diff": review.proposed_diff,
                "destination": review.destination,
                "destination_safe": review.destination_safe,
                "payload": review.payload,
                "payload_safe": review.payload_safe,
                "expires_at": review.expires_at,
            },
        )
        self.monitor.check_metrics()
        return review

    def _finish_review(
        self,
        review: ReviewRequest,
        *,
        status: str,
        reason: str,
        layer: str,
        reviewer_id: str | None = None,
        approval_id: str | None = None,
        egress_allowed: bool | None = None,
    ) -> ReviewRequest:
        review.status = status
        review.decision_reason = _safe_preview(reason, 400)
        review.reviewer_id = _safe_preview(reviewer_id, 120) if reviewer_id else None
        review.approval_id = approval_id
        review.decision_at = _iso()
        review.egress_allowed = egress_allowed
        self.audit.record_event(
            user_id=self.user_id,
            request_id=review.request_id,
            event_type="hitl_decision",
            decision=status,
            layer=layer,
            metadata={
                "review_id": review.review_id,
                "action_type": review.action_type,
                "reviewer_id": review.reviewer_id,
                "approval_id": review.approval_id,
                "reason": review.decision_reason,
                "egress_allowed": egress_allowed,
            },
        )
        terminal_blocked = status != "simulated_authorized"
        if terminal_blocked:
            self.monitor.blocked_requests += 1
        self.audit.record_output(
            user_id=self.user_id,
            text=f"Action decision: {status}. {reason}",
            blocked=terminal_blocked,
            layer=layer,
            request_id=review.request_id,
        )
        self.monitor.check_metrics()
        return review

    def expire_reviews(self) -> list[ReviewRequest]:
        """Fail closed for every pending review whose deadline has passed."""
        expired: list[ReviewRequest] = []
        now = _now_utc()
        for review in list(self.reviews.values()):
            if review.status == "pending" and now >= _from_iso(review.expires_at):
                expired.append(
                    self._finish_review(
                        review,
                        status="timeout",
                        reason="Review timeout; no side effect was authorized.",
                        layer="hitl",
                    )
                )
        return expired

    def resolve_review(
        self,
        review_id: str,
        *,
        decision: str,
        reviewer_id: str = "reviewer-local",
    ) -> ReviewRequest:
        """Apply reviewer decision, then run the deterministic egress gate."""
        self.expire_reviews()
        review = self.reviews.get(str(review_id))
        if review is None:
            raise KeyError("Unknown review")
        if review.status != "pending":
            return review

        normalized_decision = str(decision or "").strip().casefold()
        if normalized_decision not in {"approve", "reject"}:
            raise ValueError("Decision must be approve or reject")
        safe_reviewer = _safe_preview(reviewer_id, 120) or "reviewer-local"
        if normalized_decision == "reject":
            return self._finish_review(
                review,
                status="rejected",
                reason="Reviewer rejected the proposed action.",
                layer="hitl",
                reviewer_id=safe_reviewer,
            )

        approval_id = f"HITL-{secrets.token_hex(4).upper()}"
        allowed = bool(
            review.action_type == "transfer_money"
            and
            review.destination_safe
            and review.payload_safe
            and is_egress_allowed(review.destination, review.payload)
        )
        if allowed:
            return self._finish_review(
                review,
                status="simulated_authorized",
                reason="Approval recorded and exact egress policy allowed the simulated transfer.",
                layer="egress_policy",
                reviewer_id=safe_reviewer,
                approval_id=approval_id,
                egress_allowed=True,
            )
        return self._finish_review(
            review,
            status="denied_egress",
            reason="Approval was recorded, but deterministic egress policy denied the destination or payload.",
            layer="egress_policy",
            reviewer_id=safe_reviewer,
            approval_id=approval_id,
            egress_allowed=False,
        )

    def snapshot(self) -> dict:
        self.expire_reviews()
        return {
            "user_id": self.user_id,
            "session_id": self._session_id,
            "status": self.status().to_dict(),
            "metrics": self.monitor.snapshot(),
            "audit": list(self.audit.logs),
            "reviews": [review.to_dict() for review in self.reviews.values()],
        }

    def export_json(self) -> dict[str, str]:
        """Return session-only downloads without writing assignment artifacts."""
        snapshot = self.snapshot()
        return {
            "audit_log.json": json.dumps(snapshot["audit"], ensure_ascii=False, indent=2),
            "metrics.json": json.dumps(snapshot["metrics"], ensure_ascii=False, indent=2),
            "hitl_reviews.json": json.dumps(snapshot["reviews"], ensure_ascii=False, indent=2),
        }
