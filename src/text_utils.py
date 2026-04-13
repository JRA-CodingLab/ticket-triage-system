"""Text processing utilities for ticket triage.

Provides text cleaning, urgency detection, priority assignment, and
department routing for customer support tickets.
"""

import re

# Keywords that signal an urgent ticket
_URGENCY_KEYWORDS = [
    "urgent",
    "asap",
    "immediately",
    "need this resolved today",
    "really frustrated",
    "not a scam",
    "this is unacceptable",
]

# Category → department mapping
_DEPARTMENT_MAP = {
    "Payment": "Finance",
    "Refund": "Finance",
    "Technical": "Technical Support",
    "Account": "Customer Support",
    "Other": "Customer Support",
}


def clean_text(text: str) -> str:
    """Lowercase and collapse whitespace in *text*.

    Returns the cleaned string.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_urgency(text: str) -> int:
    """Return ``1`` if *text* contains any urgency keyword, else ``0``."""
    lowered = text.lower()
    for keyword in _URGENCY_KEYWORDS:
        if keyword in lowered:
            return 1
    return 0


def assign_priority(category: str, urgency: int) -> str:
    """Derive business priority from *category* and *urgency*.

    * urgency == 1 → ``"High"``
    * category in {Payment, Account} → ``"Medium"``
    * otherwise → ``"Low"``
    """
    if urgency == 1:
        return "High"
    if category in ("Payment", "Account"):
        return "Medium"
    return "Low"


def route_department(category: str) -> str:
    """Map a ticket *category* to the responsible department."""
    return _DEPARTMENT_MAP.get(category, "Customer Support")
