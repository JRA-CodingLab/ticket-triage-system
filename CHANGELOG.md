# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-13

### Added

- Multi-class ticket classification (Payment, Refund, Account, Technical, Other)
- Dual TF-IDF feature extraction (word + character n-grams)
- Linear SVM classifier with Logistic Regression backup
- Keyword-based urgency detection
- Business-rule priority assignment and department routing
- Flask web UI with form-based classification
- REST API endpoint (`POST /predict`)
- Synthetic dataset generator with noise injection
- Training CLI with template-based train/test split
- Label shuffling leakage test
- Comprehensive test suite (utils, classifier, server)
- CI workflow via GitHub Actions
