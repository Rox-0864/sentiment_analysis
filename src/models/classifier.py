import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import os

@dataclass
class SentimentResult:
    label: str          # "positive" | "negative" | "neutral"
    confidence: float   # 0.0 - 1.0
    lang: str           # "es" | "pt"

MODEL_DIR = "data/models"
_vectorizers = {}
_classifiers = {}


def _ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def _get_model(lang: str, force_retrain=False):
    """Load or train TF-IDF + LogReg model with proper train/test split."""
    vectorizer_path = f"{MODEL_DIR}/vectorizer_{lang}.joblib"
    classifier_path = f"{MODEL_DIR}/classifier_{lang}.joblib"

    # Try to load cached model
    if not force_retrain and lang in _vectorizers and lang in _classifiers:
        return _vectorizers[lang], _classifiers[lang]

    if not force_retrain and os.path.exists(vectorizer_path) and os.path.exists(classifier_path):
        _vectorizers[lang] = joblib.load(vectorizer_path)
        _classifiers[lang] = joblib.load(classifier_path)
        return _vectorizers[lang], _classifiers[lang]

    # Train with proper split
    print(f"Training TF-IDF model for {lang}...")
    texts, labels = _load_training_data(lang)

    # Split into train (70%) and test (30%)
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"  Train: {len(texts_train)} samples, Test: {len(texts_test)} samples")

    # Train vectorizer and classifier on training set ONLY
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), lowercase=True)
    X_train = vectorizer.fit_transform(texts_train)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, labels_train)

    # Evaluate on TEST set (unseen data)
    X_test = vectorizer.transform(texts_test)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(labels_test, predictions)

    print(f"  ✅ Test Accuracy: {accuracy*100:.1f}% (on {len(texts_test)} unseen samples)")
    print(f"\n  Classification Report:")
    print(classification_report(labels_test, predictions, target_names=["negative", "positive"]))

    # Save model
    _ensure_model_dir()
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(clf, classifier_path)

    _vectorizers[lang] = vectorizer
    _classifiers[lang] = clf
    return vectorizer, clf


def _load_training_data(lang: str):
    """Load training data from CSV files."""
    texts = []
    labels = []

    # Determine which files to load based on language
    if lang == "es":
        files = [
            ("data/spanish_amazon_reviews.csv", "sentiment"),  # 200k Amazon reviews
            ("data/ecommerce_reviews.csv", "rating"),  # Local e-commerce reviews
            ("data/es_reviews_sample.csv", "sentiment"),  # Tweet sentiment
        ]
    else:  # pt
        files = [
            ("data/portuguese_ecommerce_reviews.csv", "sentiment"),  # 73k PT reviews
            ("data/pt_reviews_sample.csv", "sentiment"),  # Tweet sentiment
        ]

    for file_path, label_source in files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df = df.dropna(subset=["sentiment"] if label_source == "sentiment" else ["rating"])

            if label_source == "rating":
                # Convert rating to sentiment
                df["true_sentiment"] = df["rating"].apply(
                    lambda x: "negative" if x <= 2 else ("neutral" if x == 3 else "positive")
                )
                # Filter only positive/negative for binary classification
                df = df[df["true_sentiment"].isin(["positive", "negative"])]
                if len(df) > 0:
                    df = df.dropna(subset=["review_text"])
                    texts.extend(df["review_text"].tolist())
                    labels.extend(df["true_sentiment"].tolist())

            else:  # sentiment column
                df = df[df["sentiment"].isin(["positive", "negative"])]
                if len(df) > 0:
                    text_col = "text" if "text" in df.columns else ("review_text" if "review_text" in df.columns else "text_clean")
                    df = df.dropna(subset=[text_col])
                    texts.extend(df[text_col].tolist())
                    labels.extend(df["sentiment"].tolist())

    # Fallback to seed data if still empty
    if len(texts) == 0:
        seed_texts, seed_labels = _get_seed_data(lang)
        texts.extend(seed_texts)
        labels.extend(seed_labels)

    print(f"  Loaded {len(texts)} training samples for {lang}")
    return texts, labels


def _get_seed_data(lang: str):
    """Minimal seed data for cold start."""
    if lang == "pt":
        texts = [
            "o atendimento foi ótimo", "gostei muito do produto", "serviço excelente",
            "o serviço é terrível", "péssimo atendimento", "detesto esse produto",
        ]
        labels = ["positive", "positive", "positive", "negative", "negative", "negative"]
    else:
        texts = [
            "el servicio es excelente", "me encanta este producto", "muy buen servicio",
            "el servicio es terrible", "péssimo atendimiento", "detesto este producto",
        ]
        labels = ["positive", "positive", "positive", "negative", "negative", "negative"]

    return texts, labels


def classify_sentiment(text: str, lang: Optional[str] = None) -> SentimentResult:
    """
    Classify text sentiment using TF-IDF + Logistic Regression.

    Args:
        text: Text to classify
        lang: Language code ("es" or "pt"). If None, tries to infer.

    Returns:
        SentimentResult with label, confidence, and language.
    """
    from src.utils.lang_detect import detect_lang

    if lang is None:
        lang = detect_lang(text)

    if lang not in ("es", "pt"):
        lang = "es"

    vectorizer, clf = _get_model(lang)
    X = vectorizer.transform([text])

    prediction = clf.predict(X)[0]
    probas = clf.predict_proba(X)[0]
    confidence = float(np.max(probas))

    return SentimentResult(
        label=prediction,
        confidence=confidence,
        lang=lang
    )
