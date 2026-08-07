"""Human-in-the-loop routing and review requirements for VinBank actions."""
from __future__ import annotations

import math
from dataclasses import dataclass


HIGH_RISK_ACTIONS = [
    "transfer_money",
    "close_account",
    "change_password",
    "delete_data",
    "update_personal_info",
]


@dataclass
class RoutingDecision:
    """A deterministic routing result; no LLM may override its risk decision."""

    action: str
    confidence: float
    reason: str
    priority: str
    requires_human: bool


class ConfidenceRouter:
    """Route answers by confidence while fail-closing sensitive bank actions."""

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(
        self, response: str, confidence: float, action_type: str = "general"
    ) -> RoutingDecision:
        """Return an auto-send, review, or escalation decision.

        Any malformed confidence score is treated as low confidence.  Crucially,
        every entry in ``HIGH_RISK_ACTIONS`` escalates even when a model reports
        perfect confidence, because confidence is not a human authorization.
        """
        normalized_action = str(action_type or "general").strip().casefold()
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.0

        if normalized_action in HIGH_RISK_ACTIONS:
            return RoutingDecision(
                action="escalate",
                confidence=score,
                reason=f"High-risk action: {normalized_action}; recorded human approval is required.",
                priority="high",
                requires_human=True,
            )

        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            return RoutingDecision(
                action="escalate",
                confidence=score,
                reason="Invalid confidence score; escalating safely.",
                priority="high",
                requires_human=True,
            )
        if score >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                action="auto_send",
                confidence=score,
                reason="High confidence for a non-sensitive response.",
                priority="low",
                requires_human=False,
            )
        if score >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                action="queue_review",
                confidence=score,
                reason="Medium confidence; reviewer confirmation is needed.",
                priority="normal",
                requires_human=True,
            )
        return RoutingDecision(
            action="escalate",
            confidence=score,
            reason="Low confidence; escalating to a human reviewer.",
            priority="high",
            requires_human=True,
        )


# The approval paths make missing approval, rejection, and timeout all deny the
# proposed side effect.  Reviewers receive intent plus a proposed diff rather
# than relying on an opaque model summary.
hitl_decision_points = [
    {
        "id": 1,
        "name": "Transfer authorization",
        "trigger": "Every transfer_money proposal, regardless of model confidence or amount.",
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "Correlation ID; authenticated customer/session; masked source and destination "
            "accounts; amount/currency; available balance; fraud/risk flags; proposed transfer diff; policy version."
        ),
        "example": "The agent proposes transferring VND 50,000,000 to a new beneficiary.",
        "approval_path": (
            "A reviewer verifies intent and policy, then records approve with reviewer ID and "
            "HITL approval ID. Reject cancels the transfer. Timeout or missing approval denies it."
        ),
        "audit_fields": (
            "request_id, correlation_id, action_type, customer/session reference, masked source/destination, "
            "amount, proposed diff, risk signals, reviewer_id, approval_id, decision, decision_timestamp, policy_version"
        ),
    },
    {
        "id": 2,
        "name": "Account closure, deletion, and personal-data change",
        "trigger": "close_account, delete_data, or update_personal_info is proposed.",
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "Correlation ID; verified identity result; current and requested PII values as a masked diff; "
            "account balance/open obligations; retention/legal-hold flags; customer confirmation evidence."
        ),
        "example": "A request changes the registered phone number and then asks to close an account.",
        "approval_path": (
            "Reviewer approves only after identity and legal checks. Reject preserves existing data. "
            "A timeout leaves the request pending only for display and denies the side effect."
        ),
        "audit_fields": (
            "request_id, correlation_id, intent, masked before/after diff, verification method, legal-hold result, "
            "reviewer_id, approval_id, decision, timeout_at, decision_timestamp"
        ),
    },
    {
        "id": 3,
        "name": "Credential reset under fraud or identity uncertainty",
        "trigger": "change_password is requested, or identity/fraud signals conflict during recovery.",
        "hitl_model": "human-as-tiebreaker",
        "context_needed": (
            "Correlation ID; recovery channel; device/location change; failed-attempt history; identity-check evidence; "
            "risk score; exact proposed reset action; customer-contact-safe channel."
        ),
        "example": "A new device requests an urgent password reset after repeated failed logins.",
        "approval_path": (
            "Automated checks may prepare evidence, but a reviewer resolves conflicts. Approve issues a one-time "
            "reset through the verified channel; reject or timeout denies the reset and may open a fraud case."
        ),
        "audit_fields": (
            "request_id, correlation_id, action_type, identity evidence references, device/risk indicators, proposed diff, "
            "reviewer_id, approval_id, decision, reason, timeout_at, decision_timestamp"
        ),
    },
]


def test_confidence_router():
    """Print representative routing outcomes for a manual Part 4 check."""
    router = ConfidenceRouter()
    cases = [
        ("Balance inquiry", 0.95, "general"),
        ("Interest rate question", 0.82, "general"),
        ("Ambiguous request", 0.55, "general"),
        ("Transfer VND 50,000,000", 0.98, "transfer_money"),
        ("Close account", 0.91, "close_account"),
    ]
    print("Testing ConfidenceRouter:")
    for scenario, confidence, action_type in cases:
        decision = router.route(scenario, confidence, action_type)
        print(
            f"  {scenario}: {decision.action} ({decision.priority}, "
            f"human={decision.requires_human})"
        )


def test_hitl_points():
    """Display the review requirements that must accompany sensitive actions."""
    print("HITL Decision Points:")
    for point in hitl_decision_points:
        print(f"  #{point['id']} {point['name']}: {point['trigger']}")


if __name__ == "__main__":
    test_confidence_router()
    test_hitl_points()
