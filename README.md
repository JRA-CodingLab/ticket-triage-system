# Ticket Triage System

[![CI](https://github.com/JRA-CodingLab/ticket-triage-system/actions/workflows/ci.yml/badge.svg)](https://github.com/JRA-CodingLab/ticket-triage-system/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)

Automatic customer support ticket classification and routing system built with scikit-learn and Flask.

## Features

- **Multi-class classification** — categorises tickets into Payment, Refund, Account, Technical, or Other
- **Urgency detection** — keyword-based binary urgency flag
- **Priority assignment** — business-rule-driven priority (High / Medium / Low)
- **Department routing** — maps categories to Finance, Technical Support, or Customer Support
- **Dual TF-IDF features** — word-level and character-level n-grams for robust text representation
- **Web UI + REST API** — classify tickets from a form or programmatically via JSON

## Quick Start

```bash
# Clone the repository
git clone https://github.com/JRA-CodingLab/ticket-triage-system.git
cd ticket-triage-system

# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic training data
python -m src.generate_data --out data/tickets.csv

# Train the model
python -m src.train --data data/tickets.csv --out models/

# Launch the web app
python web/server.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## REST API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticket": "My payment was declined and I need this fixed ASAP"}'
```

Response:

```json
{
  "category": "Payment",
  "urgency": 1,
  "priority": "High",
  "department": "Finance"
}
```

## Production

```bash
gunicorn web.server:app --bind 0.0.0.0:8000 --workers 4
```

## Testing

```bash
pip install pytest pytest-cov
pytest
```

## Project Structure

```
├── src/                # Core library
│   ├── text_utils.py   # Text cleaning, urgency, priority, routing
│   ├── classifier.py   # Load models & classify tickets
│   ├── train.py        # Training CLI
│   └── generate_data.py # Synthetic dataset generator
├── web/                # Flask application
│   ├── server.py       # Routes (form + API)
│   ├── templates/      # HTML templates
│   └── static/         # CSS
├── models/             # Serialized model artefacts
├── data/               # Datasets
└── tests/              # Test suite
```

## License

[MIT](LICENSE) © 2026 Juan Ruiz Alonso
