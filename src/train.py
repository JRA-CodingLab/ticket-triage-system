#!/usr/bin/env python3
"""Training script for the ticket triage classifier.

Usage::

    python -m src.train --data data/tickets.csv --out models/

Reads a CSV with columns ``ticket_text``, ``category``, ``template_id``,
trains multiple classifiers, evaluates them, and serializes the best
one together with feature extractors and label encoder.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from .text_utils import clean_text


def _load_csv(path: str):
    """Read the dataset CSV and return lists of texts, labels, template ids."""
    import csv

    texts, labels, template_ids = [], [], []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            texts.append(row["ticket_text"])
            labels.append(row["category"])
            template_ids.append(row["template_id"])
    return texts, labels, template_ids


def _split_by_template(texts, labels, template_ids, test_size=0.25, seed=42):
    """Split data by unique template_id to prevent data leakage."""
    unique_ids = list(set(template_ids))
    train_ids, test_ids = train_test_split(
        unique_ids, test_size=test_size, random_state=seed
    )
    train_ids_set = set(train_ids)
    test_ids_set = set(test_ids)

    X_train, y_train, X_test, y_test = [], [], [], []
    for text, label, tid in zip(texts, labels, template_ids):
        if tid in train_ids_set:
            X_train.append(text)
            y_train.append(label)
        elif tid in test_ids_set:
            X_test.append(text)
            y_test.append(label)
    return X_train, y_train, X_test, y_test


def _label_shuffle_test(model, X_features, y_encoded, seed=42):
    """Sanity check: shuffled labels should yield much worse F1."""
    rng = np.random.RandomState(seed)
    y_shuffled = y_encoded.copy()
    rng.shuffle(y_shuffled)
    model_clone = type(model)(**model.get_params())
    model_clone.fit(X_features, y_shuffled)
    preds = model_clone.predict(X_features)
    f1 = f1_score(y_shuffled, preds, average="macro")
    return f1


def train(data_path: str, output_dir: str) -> None:
    """Train classifiers on *data_path* and save artefacts to *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    texts, labels, template_ids = _load_csv(data_path)
    print(f"Loaded {len(texts)} samples")

    # 2. Template-based split
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = _split_by_template(
        texts, labels, template_ids
    )
    print(f"Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")

    # 3. Clean text
    X_train_clean = [clean_text(t) for t in X_train_raw]
    X_test_clean = [clean_text(t) for t in X_test_raw]

    # 4. TF-IDF features
    tfidf_word = TfidfVectorizer(
        ngram_range=(1, 2), max_features=10000, sublinear_tf=True
    )
    tfidf_char = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), max_features=5000
    )

    X_train_word = tfidf_word.fit_transform(X_train_clean)
    X_train_char = tfidf_char.fit_transform(X_train_clean)
    X_train_features = hstack([X_train_word, X_train_char])

    X_test_word = tfidf_word.transform(X_test_clean)
    X_test_char = tfidf_char.transform(X_test_clean)
    X_test_features = hstack([X_test_word, X_test_char])

    # 5. Label encoding
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    # 6. Train models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LinearSVM": LinearSVC(),
        "MultinomialNB": MultinomialNB(),
    }

    best_name, best_f1, best_model = None, -1.0, None

    for name, model in models.items():
        model.fit(X_train_features, y_train)
        preds = model.predict(X_test_features)
        f1 = f1_score(y_test, preds, average="macro")
        print(f"\n--- {name} (macro F1: {f1:.4f}) ---")
        print(classification_report(y_test, preds, target_names=le.classes_))
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")

    # 7. Label shuffle leakage test
    shuffle_f1 = _label_shuffle_test(best_model, X_train_features, y_train)
    print(f"Label-shuffle F1: {shuffle_f1:.4f} (should be much lower than {best_f1:.4f})")

    # 8. Save artefacts
    joblib.dump(best_model, out / "svm.pkl")
    joblib.dump(models["LogisticRegression"], out / "lr.pkl")
    joblib.dump(tfidf_word, out / "tfidf_word.pkl")
    joblib.dump(tfidf_char, out / "tfidf_char.pkl")
    joblib.dump(le, out / "label_encoder.pkl")
    print(f"\nArtefacts saved to {out}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ticket triage classifiers")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument(
        "--out", default="models", help="Output directory for artefacts"
    )
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
