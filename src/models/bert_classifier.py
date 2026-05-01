import torch
from transformers import pipeline
from dataclasses import dataclass
from typing import Optional, Dict
import os

@dataclass
class SentimentResult:
    label: str          # "positive" | "negative" | "neutral"
    confidence: float   # 0.0 - 1.0
    lang: str           # "es" | "pt"
    model: str          # "bert" | "tfidf"


# Model configurations
MODEL_CONFIG = {
    "es": {
        "model_name": "pysentimiento/robertuito-sentiment-analysis",
        "display_name": "RoBERTuito"
    },
    "pt": {
        "model_name": "pysentimiento/bertimbau-sentiment",
        "display_name": "BERTimbau"
    }
}

# Cache for loaded models
_pipelines = {}


def _get_pipeline(lang: str):
    """Load or get cached transformer pipeline."""
    if lang in _pipelines:
        return _pipelines[lang]
    
    if lang not in MODEL_CONFIG:
        raise ValueError(f"Unsupported language: {lang}")
    
    config = MODEL_CONFIG[lang]
    device = 0 if torch.cuda.is_available() else -1
    
    pipe = pipeline(
        "sentiment-analysis",
        model=config["model_name"],
        tokenizer=config["model_name"],
        device=device,
        truncation=True,
        max_length=512
    )
    
    _pipelines[lang] = pipe
    return pipe


def classify_with_bert(text: str, lang: Optional[str] = None) -> SentimentResult:
    """
    Classify text sentiment using BERT models (RoBERTuito for ES, BERTimbau for PT).
    
    Args:
        text: Text to classify
        lang: Language code ("es" or "pt"). If None, tries to infer.
    
    Returns:
        SentimentResult with label, confidence, language, and model info.
    """
    from src.utils.lang_detect import detect_lang
    
    if lang is None:
        lang = detect_lang(text)
    
    if lang not in ("es", "pt"):
        lang = "es"
    
    pipe = _get_pipeline(lang)
    result = pipe(text)[0]
    
    # Map model output to standard labels
    label_map = {
        "POS": "positive",
        "NEG": "negative",
        "NEU": "neutral"
    }
    
    raw_label = result["label"]
    label = label_map.get(raw_label, raw_label.lower())
    confidence = float(result["score"])
    
    config = MODEL_CONFIG[lang]
    
    return SentimentResult(
        label=label,
        confidence=confidence,
        lang=lang,
        model=config["display_name"]
    )


def compare_models(text: str, lang: Optional[str] = None) -> Dict[str, SentimentResult]:
    """
    Compare predictions from both TF-IDF and BERT models.
    
    Args:
        text: Text to classify
        lang: Language code ("es" or "pt"). If None, tries to infer.
    
    Returns:
        Dictionary with results from both models.
    """
    from src.models.classifier import classify_sentiment as tfidf_classify
    
    results = {}
    
    # TF-IDF prediction
    tfidf_result = tfidf_classify(text, lang)
    results["TF-IDF + LogReg"] = tfidf_result
    
    # BERT prediction
    bert_result = classify_with_bert(text, lang)
    results["BERT"] = bert_result
    
    return results
