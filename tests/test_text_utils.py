"""Tests for src.text_utils."""

import pytest

from src.text_utils import assign_priority, clean_text, detect_urgency, route_department


# ── clean_text ──────────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_lowercases(self):
        assert clean_text("HELLO World") == "hello world"

    def test_collapses_whitespace(self):
        assert clean_text("too   many   spaces") == "too many spaces"

    def test_strips_leading_trailing(self):
        assert clean_text("  padded  ") == "padded"

    def test_tabs_and_newlines(self):
        assert clean_text("line1\n\tline2") == "line1 line2"

    def test_empty_string(self):
        assert clean_text("") == ""


# ── detect_urgency ────────────────────────────────────────────────────────────

class TestDetectUrgency:
    @pytest.mark.parametrize(
        "text",
        [
            "This is urgent!",
            "I need help ASAP",
            "Fix this immediately please",
            "I need this resolved today",
            "I am really frustrated with the service",
            "This is not a scam, I need help",
            "This is unacceptable behaviour",
        ],
    )
    def test_urgent_keywords(self, text):
        assert detect_urgency(text) == 1

    def test_no_urgency(self):
        assert detect_urgency("I have a billing question") == 0

    def test_empty_string(self):
        assert detect_urgency("") == 0

    def test_case_insensitive(self):
        assert detect_urgency("URGENT matter") == 1


# ── assign_priority ─────────────────────────────────────────────────────────────

class TestAssignPriority:
    def test_urgent_always_high(self):
        assert assign_priority("Other", 1) == "High"
        assert assign_priority("Payment", 1) == "High"

    def test_payment_medium(self):
        assert assign_priority("Payment", 0) == "Medium"

    def test_account_medium(self):
        assert assign_priority("Account", 0) == "Medium"

    def test_other_low(self):
        assert assign_priority("Other", 0) == "Low"

    def test_refund_low(self):
        assert assign_priority("Refund", 0) == "Low"

    def test_technical_low(self):
        assert assign_priority("Technical", 0) == "Low"


# ── route_department ─────────────────────────────────────────────────────────────

class TestRouteDepartment:
    @pytest.mark.parametrize(
        "category, expected",
        [
            ("Payment", "Finance"),
            ("Refund", "Finance"),
            ("Technical", "Technical Support"),
            ("Account", "Customer Support"),
            ("Other", "Customer Support"),
        ],
    )
    def test_known_categories(self, category, expected):
        assert route_department(category) == expected

    def test_unknown_category_fallback(self):
        assert route_department("SomethingElse") == "Customer Support"
