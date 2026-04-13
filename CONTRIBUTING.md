# Contributing

Contributions are welcome! Here's how to get started.

## Setup

```bash
git clone https://github.com/JRA-CodingLab/ticket-triage-system.git
cd ticket-triage-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov
```

## Running Tests

```bash
pytest
```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Use type hints where practical.
- Write docstrings for public functions and classes.

## Pull Requests

1. Fork the repository and create a feature branch (`git checkout -b feature/my-feature`).
2. Make your changes and add tests.
3. Run the test suite — all tests must pass.
4. Open a pull request against `main`.

## Reporting Issues

Open an issue on GitHub with:

- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behaviour

Thank you for contributing!
