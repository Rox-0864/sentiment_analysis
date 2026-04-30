from dataclasses import dataclass
from typing import Optional
from src.models.classifier import SentimentResult
from src.prediction.frustration_detector import FrustrationResult


@dataclass
class ChurnResult:
    risk: str           # "high" | "medium" | "low" | "unknown"
    confidence: float   # 0.0 - 1.0
    reason: str


def predict_churn(messages: list[dict], window_size: int = 3) -> ChurnResult:
    """
    Predict churn based on message window.
    
    Args:
        messages: List of dicts with keys:
            - "text": str
            - "sentiment": Optional[SentimentResult]
            - "frustration": Optional[FrustrationResult]
        window_size: Number of recent messages to consider (default: 3)
    
    Returns:
        ChurnResult with risk level, confidence, and reason.
    """
    if not messages or len(messages) < 1:
        return ChurnResult(risk="unknown", confidence=0.0, reason="insufficient data")
    
    # Use last window_size messages
    window = messages[-min(window_size, len(messages)):]
    
    # Count negatives and frustrations in window
    negative_count = 0
    frustration_count = 0
    total_confidence = 0.0
    
    for msg in window:
        sentiment = msg.get("sentiment")
        frustration = msg.get("frustration")
        
        if sentiment and sentiment.label == "negative":
            negative_count += 1
            total_confidence += sentiment.confidence
        
        if frustration and frustration.is_frustrated:
            frustration_count += 1
    
    # Calculate average confidence
    avg_confidence = total_confidence / negative_count if negative_count > 0 else 0.0
    
    # Decision logic
    if negative_count >= window_size and frustration_count >= 1:
        # All messages negative + at least one frustration
        return ChurnResult(
            risk="high",
            confidence=min(avg_confidence + 0.2, 1.0),
            reason=f"{negative_count}/{len(window)} msgs negative, {frustration_count} frustration signals"
        )
    elif negative_count >= 2:
        # Multiple negatives
        return ChurnResult(
            risk="medium",
            confidence=avg_confidence,
            reason=f"{negative_count}/{len(window)} msgs negative"
        )
    elif negative_count == 1 and len(window) == 1:
        # Single message, insufficient context
        return ChurnResult(
            risk="unknown",
            confidence=0.0,
            reason="insufficient context (single message)"
        )
    elif negative_count == 1:
        return ChurnResult(
            risk="low",
            confidence=0.3,
            reason="1 negative message in window"
        )
    else:
        return ChurnResult(
            risk="low",
            confidence=0.9,
            reason="no negative signals in window"
        )
