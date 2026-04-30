import sys
import pytest
from datetime import datetime, timedelta
from src.models.classifier import classify_sentiment
from src.prediction.frustration_detector import detect_frustration
from src.prediction.churn_predictor import predict_churn


def test_end_to_end_es():
    """Process a small Spanish corpus through the full pipeline."""
    corpus = [
        "el servicio es excelente",
        "me encanta este producto",
        "el servicio es terrible",
        "estoy harto de los retrasos",
        "quiero cancelar mi suscripción",
    ]
    
    messages = []
    for text in corpus:
        sentiment = classify_sentiment(text, lang="es")
        frustration = detect_frustration(text, sentiment, lang="es")
        messages.append({
            "text": text,
            "sentiment": sentiment,
            "frustration": frustration
        })
    
    # Verify sentiment classification
    assert messages[0]["sentiment"].label == "positive"
    assert messages[2]["sentiment"].label == "negative"
    
    # Verify frustration detection
    assert messages[3]["frustration"].is_frustrated is True
    assert "harto" in messages[3]["frustration"].signals
    
    # Predict churn on last 3 messages
    churn = predict_churn(messages[-3:], window_size=3)
    assert churn.risk in ["high", "medium"]  # At least some risk


def test_end_to_end_pt():
    """Process a small Portuguese corpus through the full pipeline."""
    corpus = [
        "o atendimento foi ótimo",
        "o serviço é terrível",
        "estou cansado disso",
        "quero cancelar já",
    ]
    
    messages = []
    for text in corpus:
        sentiment = classify_sentiment(text, lang="pt")
        frustration = detect_frustration(text, sentiment, lang="pt")
        messages.append({
            "text": text,
            "sentiment": sentiment,
            "frustration": frustration
        })
    
    # Verify sentiment
    assert messages[0]["sentiment"].label == "positive"
    assert messages[1]["sentiment"].label == "negative"
    
    # Verify frustration
    assert messages[2]["frustration"].is_frustrated is True


if __name__ == "__main__":
    test_end_to_end_es()
    print("ES end-to-end: PASSED")
    test_end_to_end_pt()
    print("PT end-to-end: PASSED")
