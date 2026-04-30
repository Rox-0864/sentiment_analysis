import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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


def _get_model(lang: str):
    """Load or train TF-IDF + LogReg model."""
    vectorizer_path = f"{MODEL_DIR}/vectorizer_{lang}.joblib"
    classifier_path = f"{MODEL_DIR}/classifier_{lang}.joblib"
    
    if lang in _vectorizers and lang in _classifiers:
        return _vectorizers[lang], _classifiers[lang]
    
    if os.path.exists(vectorizer_path) and os.path.exists(classifier_path):
        _vectorizers[lang] = joblib.load(vectorizer_path)
        _classifiers[lang] = joblib.load(classifier_path)
        return _vectorizers[lang], _classifiers[lang]
    
    # Train with available data
    texts, labels = _load_training_data(lang)
    
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), lowercase=True)
    X = vectorizer.fit_transform(texts)
    
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, labels)
    
    _ensure_model_dir()
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(clf, classifier_path)
    
    _vectorizers[lang] = vectorizer
    _classifiers[lang] = clf
    return vectorizer, clf


def _load_training_data(lang: str):
    """Load training data from CSV or use seed data."""
    csv_path = f"data/{lang}_reviews_sample.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Filter only positive/negative for binary classification
        df = df[df["sentiment"].isin(["positive", "negative"])]
        if len(df) > 0:
            return df["text_clean"].tolist(), df["sentiment"].tolist()
    
    # Fallback seed data
    return _get_seed_data(lang)


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
