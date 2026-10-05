import smtplib
import ssl
from email.message import EmailMessage

from agent.config import NOTIFY_EMAIL_TO, SMTP_FROM, SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USER

MAX_BODY_CHARS = 8000


def email_configured() -> bool:
    return bool(SMTP_HOST and NOTIFY_EMAIL_TO)


def build_message(job: dict) -> EmailMessage:
    result = job.get("result") or {}
    failed = job["status"] != "done"
    accepted = not failed and result.get("accepted", True)
    outcome = "failed" if failed else ("done" if accepted else "rejected")
    msg = EmailMessage()
    msg["Subject"] = f"[agent] task {outcome}: {job['task'][:60]}"
    msg["From"] = SMTP_FROM or SMTP_USER
    msg["To"] = NOTIFY_EMAIL_TO
    body = job.get("error") if failed else result.get("final_output", "<no output>")
    lines = [f"Task: {job['task']}", f"Job: {job['id']}  Status: {outcome}"]
    if not failed:
        lines.append(f"Tier: {result.get('tier')}  Attempts: {result.get('attempt_count')}")
    lines += ["", (body or "")[:MAX_BODY_CHARS]]
    msg.set_content("\n".join(lines))
    return msg


def send_email(job: dict) -> None:
    """Completion hook: mail the result to NOTIFY_EMAIL_TO. Only jobs submitted with notify=True
    are mailed, and the recipient is fixed in the environment - never chosen by the caller."""
    if not job.get("notify"):
        return
    if not email_configured():
        print("[notify] notify=True but SMTP_HOST / NOTIFY_EMAIL_TO are not set; skipping email", flush=True)
        return
    msg = build_message(job)
    context = ssl.create_default_context()
    if SMTP_PORT == 465:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=30) as s:
            _login_and_send(s, msg)
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
            s.starttls(context=context)
            _login_and_send(s, msg)


def _login_and_send(s: smtplib.SMTP, msg: EmailMessage) -> None:
    if SMTP_USER:
        s.login(SMTP_USER, SMTP_PASSWORD)
    s.send_message(msg)
