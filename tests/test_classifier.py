"""Tests for src.classifier — uses mocked model artefacts."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.classifier import TicketClassifier


@pytest.fixture()
def mock_classifier():
    """Return a TicketClassifier with all artefacts mocked."""
    with patch("src.classifier._load_artefact") as mock_load:
        # Build mock objects
        mock_svm = MagicMock()
        mock_svm.predict.return_value = np.array([2])

        mock_word = MagicMock()
        mock_word.transform.return_value = csr_matrix(np.array([[1.0, 0.5]]))

        mock_char = MagicMock()
        mock_char.transform.return_value = csr_matrix(np.array([[0.3, 0.7]]))

        mock_le = MagicMock()
        mock_le.inverse_transform.return_value = np.array(["Payment"])

        def side_effect(name, _dir):
            mapping = {
                "svm.pkl": mock_svm,
                "tfidf_word.pkl": mock_word,
                "tfidf_char.pkl": mock_char,
                "label_encoder.pkl": mock_le,
            }
            return mapping[name]

        mock_load.side_effect = side_effect

        clf = TicketClassifier(models_dir="/fake/models")
    return clf


class TestTicketClassifier:
    def test_classify_returns_all_keys(self, mock_classifier):
        result = mock_classifier.classify("My payment was declined")
        assert set(result.keys()) == {"category", "urgency", "priority", "department"}

    def test_classify_category(self, mock_classifier):
        result = mock_classifier.classify("I was charged twice")
        assert result["category"] == "Payment"

    def test_classify_department_routing(self, mock_classifier):
        result = mock_classifier.classify("I was charged twice")
        assert result["department"] == "Finance"

    def test_classify_urgency_detected(self, mock_classifier):
        result = mock_classifier.classify("This is urgent! I was charged twice")
        assert result["urgency"] == 1
        assert result["priority"] == "High"

    def test_classify_no_urgency(self, mock_classifier):
        result = mock_classifier.classify("I have a billing question")
        assert result["urgency"] == 0

    def test_classify_calls_pipeline(self, mock_classifier):
        """Verify the full pipeline is invoked in order."""
        mock_classifier.classify("test ticket")
        mock_classifier.tfidf_word.transform.assert_called_once()
        mock_classifier.tfidf_char.transform.assert_called_once()
        mock_classifier.svm.predict.assert_called_once()
        mock_classifier.label_encoder.inverse_transform.assert_called_once()
