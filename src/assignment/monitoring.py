"""Aggregate security metrics and threshold alerts for the defense pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Alert:
    """A threshold breach that can be forwarded to incident response."""

    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Track blocks, rate-limit pressure, and judge availability over time."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    def check_metrics(self) -> list[Alert]:
        """Recompute current alerts without accumulating duplicate stale alerts."""
        snapshot = self.snapshot()
        alerts: list[Alert] = []
        if snapshot["block_rate"] >= self.block_rate_threshold:
            alerts.append(Alert(
                metric="block_rate",
                value=snapshot["block_rate"],
                threshold=self.block_rate_threshold,
                message="Block rate exceeded the configured threshold.",
            ))
        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            alerts.append(Alert(
                metric="rate_limit_hits",
                value=float(self.rate_limit_hits),
                threshold=float(self.rate_limit_hit_threshold),
                message="Rate-limit hits indicate a possible flood or cost attack.",
            ))
        if snapshot["judge_fail_rate"] >= self.judge_fail_rate_threshold:
            alerts.append(Alert(
                metric="judge_fail_rate",
                value=snapshot["judge_fail_rate"],
                threshold=self.judge_fail_rate_threshold,
                message="Safety judge failure rate exceeded the configured threshold.",
            ))
        self.alerts = alerts
        return list(alerts)

    def export_json(self, filepath: str = "outputs/metrics.json") -> Path:
        """Write a fresh metrics snapshot plus current alerts in UTF-8 JSON."""
        self.check_metrics()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def snapshot(self) -> dict:
        """Return a JSON-serializable view suitable for dashboards or replay."""
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "thresholds": {
                "block_rate": self.block_rate_threshold,
                "rate_limit_hits": self.rate_limit_hit_threshold,
                "judge_fail_rate": self.judge_fail_rate_threshold,
            },
            "alerts": [
                {
                    "metric": alert.metric,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "message": alert.message,
                }
                for alert in self.alerts
            ],
        }
