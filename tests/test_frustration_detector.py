import pytest
from src.prediction.frustration_detector import FrustrationResult, detect_frustration


def test_frustration_known_keywords_es():
    result = detect_frustration("estoy harto de este servicio")
    assert result.is_frustrated is True
    assert "harto" in result.signals
    assert result.intensity > 0


def test_frustration_keywords_pt():
    result = detect_frustration("estou cansado disso", lang="pt")
    assert result.is_frustrated is True
    assert "cansado" in result.signals


def test_frustration_neutral_text():
    result = detect_frustration("el día está bonito hoy")
    assert result.is_frustrated is False
    assert len(result.signals) == 0


def test_frustration_with_negative_sentiment():
    from src.models.classifier import SentimentResult
    
    sentiment = SentimentResult(label="negative", confidence=0.90, lang="es")
    result = detect_frustration("esto no me gusta nada", sentiment)
    assert result.is_frustrated is True
    assert result.intensity >= 0.45  # 0.3 (base) + 0.9*0.5 = 0.75


def test_frustration_high_confidence_negative():
    from src.models.classifier import SentimentResult
    
    sentiment = SentimentResult(label="negative", confidence=0.95, lang="es")
    result = detect_frustration("no se qué pensar", sentiment)
    # No keyword, but high confidence negative → frustrated
    assert result.is_frustrated is True


def test_frustration_empty_text():
    result = detect_frustration("")
    assert result.is_frustrated is False
    assert result.intensity == 0.0
    assert result.signals == []
