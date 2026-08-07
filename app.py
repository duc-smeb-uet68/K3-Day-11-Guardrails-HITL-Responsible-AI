"""Streamlit front end for the protected VinBank assistant."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from app_runtime import VinBankAppSession
from hitl.hitl import HIGH_RISK_ACTIONS
from assignment.pipeline import APPROVED_EGRESS_DESTINATIONS


st.set_page_config(
    page_title="VinBank Secure Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


ACTION_LABELS = {
    "transfer_money": "Chuyển tiền",
    "close_account": "Đóng tài khoản",
    "change_password": "Đổi mật khẩu",
    "delete_data": "Xóa dữ liệu",
    "update_personal_info": "Cập nhật thông tin cá nhân",
}
ACTION_TYPES = [action for action in HIGH_RISK_ACTIONS if action in ACTION_LABELS]
TRANSFER_ENDPOINT = next(iter(APPROVED_EGRESS_DESTINATIONS))


def _runtime() -> VinBankAppSession:
    if "vinbank_runtime" not in st.session_state:
        st.session_state.vinbank_runtime = VinBankAppSession()
    return st.session_state.vinbank_runtime


def _init_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("action_notice", None)


def _status_badge(ready: bool) -> None:
    if ready:
        st.success("Protected agent: sẵn sàng", icon="✅")
    else:
        st.warning("Protected agent: chưa sẵn sàng", icon="⚠️")


def _render_sidebar(runtime: VinBankAppSession) -> None:
    with st.sidebar:
        st.markdown("## 🏦 VinBank")
        st.caption("Controlled Agent Security Lab")
        status = runtime.status()
        _status_badge(status.ready)
        st.caption(status.message)
        st.divider()
        st.markdown("**Runtime**")
        st.write(f"Google ADK: {'có' if status.adk_available else 'thiếu'}")
        st.write(f"OpenRouter key: {'đã cấu hình' if status.api_key_configured else 'chưa có'}")
        st.divider()
        st.caption(f"Session user: `{runtime.user_id}`")
        st.caption("Dữ liệu chỉ tồn tại trong phiên trình duyệt.")
        if st.button("Xóa phiên chat", use_container_width=True):
            st.session_state.pop("vinbank_runtime", None)
            st.session_state.chat_history = []
            st.session_state.action_notice = None
            st.rerun()


def _render_chat(runtime: VinBankAppSession) -> None:
    st.subheader("Trợ lý VinBank")
    st.caption(
        "Mọi tin nhắn đi qua rate limiter, input guardrail, protected agent và output guardrail."
    )
    status = runtime.status()
    if not status.ready:
        st.info(
            "Chat live đang tạm khóa vì runtime chưa sẵn sàng. "
            "Cài dependencies và đặt OPENROUTER_API_KEY trong .env, sau đó khởi động lại app."
        )

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["input_preview"])
        with st.chat_message("assistant"):
            st.write(turn["response"])
            badges = []
            if turn.get("blocked"):
                badges.append(f"BLOCKED · {turn.get('layer') or 'policy'}")
            else:
                badges.append("ALLOWED")
            if turn.get("redacted"):
                badges.append("OUTPUT REDACTED")
            if turn.get("judge_failed"):
                badges.append("JUDGE FAIL-CLOSED")
            st.caption(
                f"{' · '.join(badges)} · request `{turn['request_id']}` · "
                f"{turn['latency_ms']} ms"
            )

    prompt = st.chat_input(
        "Nhập câu hỏi về tài khoản, giao dịch, khoản vay…",
        disabled=not status.ready,
    )
    if prompt:
        with st.spinner("Đang kiểm tra guardrails và xử lý…"):
            turn = runtime.send_chat(prompt)
        st.session_state.chat_history.append(turn.to_dict())
        st.rerun()


def _render_action_review(runtime: VinBankAppSession) -> None:
    st.subheader("Hành động & Human-in-the-Loop")
    st.caption(
        "Đây là mô phỏng side effect. Approve không gọi endpoint và không chuyển tiền thật; "
        "egress policy luôn được kiểm tra lần cuối."
    )

    notice = st.session_state.get("action_notice")
    if notice:
        kind, message = notice
        getattr(st, kind)(message)
        st.session_state.action_notice = None

    with st.form("action_proposal_form", clear_on_submit=True):
        action_label = st.selectbox(
            "Loại hành động rủi ro cao",
            [ACTION_LABELS[action] for action in ACTION_TYPES],
        )
        action_type = next(key for key, value in ACTION_LABELS.items() if value == action_label)
        confidence = st.slider("Confidence của agent", 0.0, 1.0, 0.95, 0.01)
        context = st.text_area(
            "Context cần reviewer kiểm tra",
            value="Khách hàng đã xác nhận yêu cầu qua kênh đã xác thực.",
        )
        proposed_diff = st.text_area(
            "Proposed diff / thay đổi dự kiến",
            value="amount=500000 VND; beneficiary=masked-account-1234",
        )
        if action_type == "transfer_money":
            destination_default = TRANSFER_ENDPOINT
            payload_default = "approved transfer amount 500000 VND"
        else:
            destination_default = "https://api.vinbank.example/v1/unsupported-action"
            payload_default = "approved action context without sensitive data"
        destination = st.text_input("Destination (exact allowlist)", value=destination_default)
        payload = st.text_area("Payload gửi tới sink (sẽ bị kiểm tra lại)", value=payload_default)
        submitted = st.form_submit_button("Đưa vào hàng chờ reviewer", type="primary")

    if submitted:
        try:
            review = runtime.propose_action(
                action_type=action_type,
                confidence=confidence,
                context=context,
                proposed_diff=proposed_diff,
                destination=destination,
                payload=payload,
            )
            st.session_state.action_notice = (
                "success",
                f"Đã tạo review `{review.review_id}` với request `{review.request_id}`.",
            )
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    snapshot = runtime.snapshot()
    pending = [row for row in snapshot["reviews"] if row["status"] == "pending"]
    st.markdown("### Hàng chờ reviewer")
    if not pending:
        st.info("Chưa có action nào đang chờ duyệt.")
        return

    for row in pending:
        with st.container(border=True):
            st.markdown(
                f"**{ACTION_LABELS.get(row['action_type'], row['action_type'])}** · "
                f"`{row['review_id']}` · hết hạn `{row['expires_at']}`"
            )
            left, right = st.columns(2)
            with left:
                st.write(f"Confidence: `{row['confidence']:.2f}` · Priority: `{row['priority']}`")
                st.write(f"Route: `{row['route_action']}` · luôn cần human")
                st.caption(row["route_reason"])
                st.write(f"Context: {row['context']}")
                st.write(f"Diff: {row['proposed_diff']}")
            with right:
                st.write(f"Destination: `{row['destination']}`")
                st.write(f"Payload preview: {row['payload']}")
                if not row["destination_safe"] or not row["payload_safe"]:
                    st.warning("Payload/destination có dữ liệu nhạy cảm; approve sẽ bị egress deny.")
                reviewer_id = st.text_input(
                    "Reviewer ID",
                    value="reviewer-local",
                    key=f"reviewer-{row['review_id']}",
                )
                approve, reject = st.columns(2)
                if approve.button("Approve", key=f"approve-{row['review_id']}"):
                    result = runtime.resolve_review(
                        row["review_id"], decision="approve", reviewer_id=reviewer_id
                    )
                    st.session_state.action_notice = (
                        "success" if result.status == "simulated_authorized" else "warning",
                        f"Review kết thúc: `{result.status}` — {result.decision_reason}",
                    )
                    st.rerun()
                if reject.button("Reject", key=f"reject-{row['review_id']}"):
                    result = runtime.resolve_review(
                        row["review_id"], decision="reject", reviewer_id=reviewer_id
                    )
                    st.session_state.action_notice = (
                        "info",
                        f"Review kết thúc: `{result.status}`.",
                    )
                    st.rerun()

    completed = [row for row in snapshot["reviews"] if row["status"] != "pending"]
    if completed:
        st.markdown("### Lịch sử quyết định")
        st.dataframe(
            [
                {
                    "review_id": row["review_id"],
                    "action": ACTION_LABELS.get(row["action_type"], row["action_type"]),
                    "status": row["status"],
                    "reviewer": row["reviewer_id"],
                    "approval_id": row["approval_id"],
                    "egress_allowed": row["egress_allowed"],
                }
                for row in completed
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_console(runtime: VinBankAppSession) -> None:
    st.subheader("Security Console")
    st.caption("Số liệu và audit chỉ thuộc phiên hiện tại; nội dung đã được sanitize.")
    snapshot = runtime.snapshot()
    metrics = snapshot["metrics"]
    status = snapshot["status"]
    _status_badge(status["ready"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Requests", metrics["total_requests"])
    c2.metric("Blocked", metrics["blocked_requests"])
    c3.metric("Block rate", f"{metrics['block_rate']:.1%}")
    c4.metric("Rate-limit hits", metrics["rate_limit_hits"])
    c5.metric("Judge fail rate", f"{metrics['judge_fail_rate']:.1%}")

    alerts = metrics.get("alerts", [])
    if alerts:
        st.markdown("### Alerts")
        for alert in alerts:
            st.warning(f"{alert['metric']}: {alert['message']}")
    else:
        st.success("Không có alert vượt ngưỡng trong phiên.")

    st.markdown("### Enforcement flow")
    st.code("Rate limiter  →  Input guardrail  →  Protected model  →  Output/judge  →  HITL  →  Egress")

    st.markdown("### Audit trail")
    if snapshot["audit"]:
        st.dataframe(snapshot["audit"], use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có audit event trong phiên.")

    st.markdown("### Tải bằng chứng đã sanitize")
    exports = runtime.export_json()
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "audit_log.json",
        data=exports["audit_log.json"],
        file_name="vinbank_session_audit_log.json",
        mime="application/json",
        use_container_width=True,
    )
    d2.download_button(
        "metrics.json",
        data=exports["metrics.json"],
        file_name="vinbank_session_metrics.json",
        mime="application/json",
        use_container_width=True,
    )
    d3.download_button(
        "hitl_reviews.json",
        data=exports["hitl_reviews.json"],
        file_name="vinbank_session_hitl_reviews.json",
        mime="application/json",
        use_container_width=True,
    )


def main() -> None:
    _init_state()
    runtime = _runtime()
    _render_sidebar(runtime)

    st.title("VinBank Secure Assistant")
    st.caption("Trợ lý ngân hàng mô phỏng với defense-in-depth, HITL và audit có thể truy vết.")
    chat_tab, hitl_tab, console_tab = st.tabs(
        ["💬 Trợ lý VinBank", "🧑‍⚖️ Hành động & HITL", "🛡️ Security Console"]
    )
    with chat_tab:
        _render_chat(runtime)
    with hitl_tab:
        _render_action_review(runtime)
    with console_tab:
        _render_console(runtime)

    st.divider()
    st.caption(
        "Demo local — không phải dịch vụ ngân hàng thật. Không nhập dữ liệu khách hàng hoặc bí mật thật."
    )


if __name__ == "__main__":
    main()
