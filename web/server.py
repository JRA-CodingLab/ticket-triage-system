"""Flask application for ticket triage.

Provides a web form at ``/`` and a JSON API at ``/predict``.
"""

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Ensure the project root is on sys.path so ``src`` can be imported.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.classifier import TicketClassifier  # noqa: E402

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)

# Load the classifier once at startup
_classifier: TicketClassifier | None = None


def get_classifier() -> TicketClassifier:
    """Lazy-load the classifier (allows overriding in tests)."""
    global _classifier
    if _classifier is None:
        models_dir = os.environ.get("MODELS_DIR", str(_PROJECT_ROOT / "models"))
        _classifier = TicketClassifier(models_dir=models_dir)
    return _classifier


def set_classifier(clf: TicketClassifier) -> None:
    """Replace the global classifier (used by tests)."""
    global _classifier
    _classifier = clf


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the form (GET) or classify a ticket (POST)."""
    result = None
    ticket_text = ""
    if request.method == "POST":
        ticket_text = request.form.get("ticket", "")
        if ticket_text.strip():
            clf = get_classifier()
            result = clf.classify(ticket_text)
    return render_template("index.html", result=result, ticket_text=ticket_text)


@app.route("/predict", methods=["POST"])
def predict():
    """JSON API endpoint for ticket classification."""
    data = request.get_json(silent=True) or {}
    ticket_text = data.get("ticket", "")
    if not ticket_text or not ticket_text.strip():
        return jsonify({"error": "Missing or empty 'ticket' field"}), 400

    clf = get_classifier()
    result = clf.classify(ticket_text)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
