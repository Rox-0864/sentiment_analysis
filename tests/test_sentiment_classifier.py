import pytest
from unittest.mock import patch, MagicMock
from src.models.classifier import SentimentResult, classify_sentiment


def test_sentiment_result_dataclass():
    result = SentimentResult(label="positive", confidence=0.95, lang="es")
    assert result.label == "positive"
    assert result.confidence == 0.95
    assert result.lang == "es"


@patch("src.models.classifier._get_model")
def test_classify_sentiment_es_negative(mock_get_model):
    mock_classifier = MagicMock()
    mock_classifier.return_value = [{"label": "NEG", "score": 0.97}]
    mock_get_model.return_value = mock_classifier
    
    result = classify_sentiment("el servicio es terrible")
    assert result.label == "negative"
    assert result.confidence == 0.97
    assert result.lang == "es"


@patch("src.models.classifier._get_model")
def test_classify_sentiment_pt_positive(mock_get_model):
    mock_classifier = MagicMock()
    mock_classifier.return_value = [{"label": "POS", "score": 0.99}]
    mock_get_model.return_value = mock_classifier
    
    result = classify_sentiment("o atendimento foi ótimo", lang="pt")
    assert result.label == "positive"
    assert result.confidence == 0.99
    assert result.lang == "pt"


@patch("src.models.classifier._get_model")
def test_classify_sentiment_neutral(mock_get_model):
    mock_classifier = MagicMock()
    mock_classifier.return_value = [{"label": "NEU", "score": 0.60}]
    mock_get_model.return_value = mock_classifier
    
    result = classify_sentiment("el día está nublado")
    assert result.label == "neutral"
    assert result.confidence == 0.60


def test_classify_sentiment_caches_model():
    from src.models.classifier import _model_cache
    _model_cache.clear()
    assert len(_model_cache) == 0
