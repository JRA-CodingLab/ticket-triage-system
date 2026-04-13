"""Tests for web.server Flask endpoints — classifier is mocked."""

import json
from unittest.mock import MagicMock, patch

import pytest

from web.server import app, set_classifier


@pytest.fixture()
def mock_clf():
    """Provide a mocked classifier and wire it into the Flask app."""
    clf = MagicMock()
    clf.classify.return_value = {
        "category": "Technical",
        "urgency": 0,
        "priority": "Low",
        "department": "Technical Support",
    }
    set_classifier(clf)
    yield clf
    set_classifier(None)


@pytest.fixture()
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── GET / ────────────────────────────────────────────────────────────────────────

class TestIndexGet:
    def test_returns_200(self, client, mock_clf):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_contains_form(self, client, mock_clf):
        resp = client.get("/")
        assert b"<form" in resp.data

    def test_no_result_on_get(self, client, mock_clf):
        resp = client.get("/")
        assert b"Classification Result" not in resp.data


# ── POST / ──────────────────────────────────────────────────────────────────────

class TestIndexPost:
    def test_returns_result(self, client, mock_clf):
        resp = client.post("/", data={"ticket": "App crashes on settings page"})
        assert resp.status_code == 200
        assert b"Classification Result" in resp.data
        assert b"Technical" in resp.data

    def test_empty_ticket_no_result(self, client, mock_clf):
        resp = client.post("/", data={"ticket": "   "})
        assert b"Classification Result" not in resp.data


# ── POST /predict ────────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_valid_ticket(self, client, mock_clf):
        resp = client.post(
            "/predict",
            data=json.dumps({"ticket": "Server is down"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["category"] == "Technical"
        assert body["department"] == "Technical Support"

    def test_missing_ticket_field(self, client, mock_clf):
        resp = client.post(
            "/predict",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_empty_ticket_field(self, client, mock_clf):
        resp = client.post(
            "/predict",
            data=json.dumps({"ticket": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_no_json_body(self, client, mock_clf):
        resp = client.post("/predict", data="not json")
        assert resp.status_code == 400
