import pytest
from src.prediction.churn_predictor import ChurnResult, predict_churn
from src.models.classifier import SentimentResult
from src.prediction.frustration_detector import FrustrationResult


def test_churn_high_risk():
    messages = [
        {"text": "terrible", "sentiment": SentimentResult("negative", 0.95, "es"), 
         "frustration": FrustrationResult(True, 0.8, ["terrible"])},
        {"text": "no sirve", "sentiment": SentimentResult("negative", 0.90, "es"),
         "frustration": FrustrationResult(True, 0.7, ["no sirve"])},
        {"text": "cancelar", "sentiment": SentimentResult("negative", 0.92, "es"),
         "frustration": FrustrationResult(True, 0.9, ["cancelar"])},
    ]
    result = predict_churn(messages)
    assert result.risk == "high"
    assert result.confidence >= 0.8


def test_churn_medium_risk():
    messages = [
        {"text": "regular", "sentiment": SentimentResult("negative", 0.70, "es"),
         "frustration": FrustrationResult(False, 0.2, [])},
        {"text": "malo", "sentiment": SentimentResult("negative", 0.85, "es"),
         "frustration": FrustrationResult(False, 0.3, [])},
    ]
    result = predict_churn(messages)
    assert result.risk == "medium"


def test_churn_low_risk():
    messages = [
        {"text": "bien", "sentiment": SentimentResult("positive", 0.85, "es"),
         "frustration": FrustrationResult(False, 0.0, [])},
        {"text": "eh", "sentiment": SentimentResult("neutral", 0.60, "es"),
         "frustration": FrustrationResult(False, 0.0, [])},
    ]
    result = predict_churn(messages, window_size=3)
    assert result.risk == "low"
    assert result.confidence >= 0.8


def test_churn_insufficient_data():
    result = predict_churn([])
    assert result.risk == "unknown"
    assert result.confidence == 0.0
    assert "insufficient" in result.reason


def test_churn_single_message():
    messages = [
        {"text": "test", "sentiment": SentimentResult("negative", 0.80, "es"),
         "frustration": FrustrationResult(False, 0.2, [])},
    ]
    result = predict_churn(messages)
    assert result.risk == "unknown"
    assert "insufficient context" in result.reason


def test_churn_window_size():
    messages = [
        {"text": "msg1", "sentiment": SentimentResult("positive", 0.90, "es"),
         "frustration": FrustrationResult(False, 0.0, [])},
        {"text": "msg2", "sentiment": SentimentResult("negative", 0.85, "es"),
         "frustration": FrustrationResult(True, 0.5, ["malo"])},
    ]
    # window_size=1: only last message (negative + frustration → high risk)
    result = predict_churn(messages, window_size=1)
    assert result.risk == "high"  # Last msg is negative with frustration
