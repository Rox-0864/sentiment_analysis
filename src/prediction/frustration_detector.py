from dataclasses import dataclass
from typing import Optional
from src.models.sentiment_classifier import SentimentResult


@dataclass
class FrustrationResult:
    is_frustrated: bool
    intensity: float    # 0.0 - 1.0
    signals: list[str]  # keywords found


# Frustration keywords for ES/PT
FRUSTRATION_KEYWORDS = {
    "es": [
        "harto", "harta", "harto/a", "horrible", "terrible", "pésimo", "pésima",
        "no aguanto", "no soporto", "ya no quiero", "me rindo", "devolver",
        "cancelar", "molesto", "molesta", "enfadado", "enfadada", "furioso",
        "furiosa", "cabreado", "cabreada", "odio", "odiar", "pesadísimo",
        "pésimo servicio", "malísimo", "una porquería", "decepcionado",
    ],
    "pt": [
        "cansado", "cansada", "horrível", "terrível", "péssimo", "péssima",
        "não aguento", "não suporto", "já não quero", "desisto", "devolver",
        "cancelar", "irritado", "irritada", "furioso", "furiosa", "ódio",
        "odiar", "péssimo serviço", "uma porcaria", "decepcionado",
    ],
}


def detect_frustration(text: str, sentiment: Optional[SentimentResult] = None, lang: Optional[str] = None) -> FrustrationResult:
    """
    Detect frustration signals based on keyword patterns + sentiment intensity.
    
    Args:
        text: Text to analyze
        sentiment: Optional pre-computed sentiment (avoids re-computing)
        lang: Optional language code ("es" or "pt"). If None, tries to infer.
    
    Returns:
        FrustrationResult with frustration flag, intensity, and signals found.
    """
    from src.utils.lang_detect import detect_lang
    
    text_lower = text.lower()
    if lang is None:
        lang = detect_lang(text)
    
    # Get keywords for the detected language
    keywords = FRUSTRATION_KEYWORDS.get(lang, FRUSTRATION_KEYWORDS["es"])
    
    # Find matching signals
    signals = [kw for kw in keywords if kw in text_lower]
    
    # Base intensity from keyword matches
    intensity = min(len(signals) * 0.3, 1.0)
    
    # Boost intensity if sentiment is negative with high confidence
    if sentiment and sentiment.label == "negative":
        intensity = min(intensity + sentiment.confidence * 0.5, 1.0)
    
    is_frustrated = len(signals) > 0 or (sentiment is not None and 
                                         sentiment.label == "negative" and 
                                         sentiment.confidence >= 0.85)
    
    return FrustrationResult(
        is_frustrated=is_frustrated,
        intensity=intensity,
        signals=signals
    )
