from dataclasses import dataclass
from typing import Optional
from transformers import pipeline, Pipeline


@dataclass
class SentimentResult:
    label: str          # "positive" | "negative" | "neutral"
    confidence: float   # 0.0 - 1.0
    lang: str           # "es" | "pt"


# Model cache to avoid reloading
_model_cache: dict[str, Pipeline] = {}


def _get_model(lang: str) -> Pipeline:
    """Load or retrieve cached sentiment analysis model."""
    if lang in _model_cache:
        return _model_cache[lang]
    
    if lang == "pt":
        model_name = "pysentimiento/bertweet-pt-sentiment"
    else:
        model_name = "pysentimiento/robertuito-sentiment-analysis"
    
    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        max_length=512
    )
    _model_cache[lang] = classifier
    return classifier


def classify_sentiment(text: str, lang: Optional[str] = None) -> SentimentResult:
    """
    Classify text sentiment using BETO (ES) or BERTimbau (PT).
    
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
        lang = "es"  # default to Spanish
    
    classifier = _get_model(lang)
    result = classifier(text)[0]
    
    # Map model output to standard labels
    raw_label = result["label"].lower()
    if "pos" in raw_label:
        label = "positive"
    elif "neg" in raw_label:
        label = "negative"
    else:
        label = "neutral"
    
    return SentimentResult(
        label=label,
        confidence=float(result["score"]),
        lang=lang
    )
