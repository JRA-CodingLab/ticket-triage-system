"""Classification pipeline for support tickets.

Loads serialized model artefacts and exposes a single ``classify_ticket``
function that returns category, urgency, priority, and department.
"""

from pathlib import Path

import joblib
from scipy.sparse import hstack

from .text_utils import assign_priority, clean_text, detect_urgency, route_department

# Default directory for serialized artefacts
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _load_artefact(name: str, models_dir: Path | None = None):
    """Load a joblib-serialized artefact by *name*."""
    directory = models_dir or _MODELS_DIR
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(f"Model artefact not found: {path}")
    return joblib.load(path)


class TicketClassifier:
    """Wraps the full classification pipeline.

    Parameters
    ----------
    models_dir : Path | str | None
        Override for the artefact directory (useful in tests).
    """

    def __init__(self, models_dir: Path | str | None = None):
        md = Path(models_dir) if models_dir else _MODELS_DIR
        self.svm = _load_artefact("svm.pkl", md)
        self.tfidf_word = _load_artefact("tfidf_word.pkl", md)
        self.tfidf_char = _load_artefact("tfidf_char.pkl", md)
        self.label_encoder = _load_artefact("label_encoder.pkl", md)

    def classify(self, ticket_text: str) -> dict:
        """Run the full pipeline on *ticket_text*.

        Returns a dict with keys: ``category``, ``urgency``, ``priority``,
        ``department``.
        """
        cleaned = clean_text(ticket_text)

        word_features = self.tfidf_word.transform([cleaned])
        char_features = self.tfidf_char.transform([cleaned])
        features = hstack([word_features, char_features])

        prediction = self.svm.predict(features)
        category = self.label_encoder.inverse_transform(prediction)[0]

        urgency = detect_urgency(ticket_text)
        priority = assign_priority(category, urgency)
        department = route_department(category)

        return {
            "category": category,
            "urgency": urgency,
            "priority": priority,
            "department": department,
        }
