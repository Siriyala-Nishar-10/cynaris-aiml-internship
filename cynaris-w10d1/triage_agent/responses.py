"""Reply text. Deterministic templates: this lesson is about graph mechanics,
not text generation, and templates keep every test exact."""
from __future__ import annotations

from typing import Optional

AUTO_REPLIES = {
    "billing": ("Thanks for contacting us about billing. We've logged your query and "
                "a billing specialist will review your account."),
    "technical": ("Sorry you're running into a problem. Please restart the app and "
                  "reply with your device, app version and the exact error message."),
    "general": "Hello! Happy to help. What would you like to know?",
}
GENERIC_FOLLOW_UP = "Thanks for your message. A specialist will follow up shortly."


def compose_response(label: str, decision: Optional[dict]) -> str:
    if decision is None:  # automatic path: no human involved
        return AUTO_REPLIES.get(label, GENERIC_FOLLOW_UP)

    # Strip a trailing period so we never produce "..", or run two sentences together.
    note = (decision.get("note") or "").strip().rstrip(".")
    if not decision.get("approved", False):
        reason = f" Reason: {note}." if note else ""
        return ("A human agent reviewed your request and did not approve automated "
                f"handling.{reason} A specialist will follow up with you.")
    if label == "refund":
        return ("A human agent approved your refund request."
                + (f" Note: {note}." if note else "") + " It will now be processed.")
    if label == "unknown":
        return note or GENERIC_FOLLOW_UP  # the human wrote the answer
    return AUTO_REPLIES.get(label, GENERIC_FOLLOW_UP) + \
        (f" (Reviewed by a human agent: {note})" if note else "")
