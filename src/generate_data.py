#!/usr/bin/env python3
"""Synthetic ticket dataset generator.

Usage::

    python -m src.generate_data --out data/tickets.csv

Produces 500 samples (5 categories × 10 templates × 10 variations) with
realistic noise injection.
"""

import argparse
import csv
import random
import string

# ── Templates per category ─────────────────────────────────────────────

TEMPLATES: dict[str, list[str]] = {
    "Payment": [
        "My payment was declined when I tried to purchase the subscription.",
        "I was charged twice for my last order.",
        "The payment page keeps showing an error when I enter my card.",
        "I cannot add a new payment method to my account.",
        "My bank says the charge is pending but your site says it failed.",
        "I need a receipt for the payment I made last week.",
        "The total on my invoice does not match what I was charged.",
        "I was charged a different amount than what was displayed at checkout.",
        "Can you confirm whether my payment went through successfully?",
        "I want to switch from monthly to annual billing.",
    ],
    "Refund": [
        "I would like a refund for my last purchase.",
        "I returned the item but have not received my refund yet.",
        "How long does it take for a refund to show up on my card?",
        "I cancelled my subscription and want a prorated refund.",
        "The product was defective so I need my money back.",
        "I was promised a refund by your support team but it has not arrived.",
        "Can I get a refund if I change my mind about a purchase?",
        "My refund was processed but the amount is incorrect.",
        "I never received the item so please process a full refund.",
        "I filed a refund request three weeks ago and have no update.",
    ],
    "Account": [
        "I cannot log in to my account even though my password is correct.",
        "I need to change the email address on my account.",
        "My account was locked after too many failed login attempts.",
        "I want to delete my account and all associated data.",
        "How do I enable two-factor authentication on my account?",
        "Someone else might have access to my account.",
        "I forgot my password and the reset link is not working.",
        "I need to update the billing address on my account.",
        "My profile information is not saving when I try to update it.",
        "Can I merge two accounts into one?",
    ],
    "Technical": [
        "The app crashes every time I open the settings page.",
        "I am getting a 500 error when trying to access the dashboard.",
        "The file upload feature is not working on mobile.",
        "Pages are loading extremely slowly for the past two days.",
        "I cannot export my data to CSV, the button does nothing.",
        "The search function returns no results even for known items.",
        "Notifications are not being sent even though they are enabled.",
        "The integration with the calendar service stopped syncing.",
        "The dark mode toggle does not apply the theme correctly.",
        "Videos are not playing in the embedded player.",
    ],
    "Other": [
        "I have a question about your pricing plans.",
        "Where can I find documentation for your product?",
        "Do you offer discounts for educational institutions?",
        "I would like to provide feedback on a recent feature update.",
        "How can I contact your sales team for a demo?",
        "Is there an affiliate program I can join?",
        "I would like to know more about your data privacy policy.",
        "Can I request a feature that is not currently available?",
        "What are your support hours for live chat?",
        "I am interested in a partnership opportunity.",
    ],
}

_URGENT_PHRASES = [
    "This is urgent!",
    "I need this resolved ASAP.",
    "Please fix this immediately.",
    "I am really frustrated.",
    "This is unacceptable!",
    "I need this resolved today.",
]

_CHAR_MAP = {"a": "@", "o": "0", "i": "1", "e": "3", "s": "5"}


def _inject_urgency(text: str) -> str:
    """Randomly prepend or append an urgency phrase."""
    phrase = random.choice(_URGENT_PHRASES)
    if random.random() < 0.5:
        return f"{phrase} {text}"
    return f"{text} {phrase}"


def _obfuscate_chars(text: str, probability: float = 0.15) -> str:
    """Replace certain characters with look-alikes."""
    chars = list(text)
    for idx, ch in enumerate(chars):
        if ch.lower() in _CHAR_MAP and random.random() < probability:
            chars[idx] = _CHAR_MAP[ch.lower()]
    return "".join(chars)


def _random_casing(text: str) -> str:
    """Randomly change the case of some characters."""
    return "".join(
        ch.upper() if random.random() < 0.2 else ch.lower()
        for ch in text
    )


def _add_noise(text: str) -> str:
    """Apply random noise transformations to *text*."""
    # ~30 % chance of urgency
    if random.random() < 0.3:
        text = _inject_urgency(text)
    # ~25 % chance of character obfuscation
    if random.random() < 0.25:
        text = _obfuscate_chars(text)
    # ~20 % chance of random casing
    if random.random() < 0.20:
        text = _random_casing(text)
    return text


def generate_dataset() -> list[dict]:
    """Return 500 synthetic ticket rows."""
    rows: list[dict] = []
    for category, templates in TEMPLATES.items():
        for tmpl_idx, template in enumerate(templates):
            template_id = f"{category.lower()}_{tmpl_idx}"
            for _ in range(10):
                noisy = _add_noise(template)
                rows.append(
                    {
                        "ticket_text": noisy,
                        "category": category,
                        "template_id": template_id,
                    }
                )
    random.shuffle(rows)
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    """Write *rows* to a CSV file at *path*."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ticket_text", "category", "template_id"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ticket dataset")
    parser.add_argument("--out", default="data/tickets.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    rows = generate_dataset()
    write_csv(rows, args.out)
    print(f"Generated {len(rows)} samples → {args.out}")


if __name__ == "__main__":
    main()
