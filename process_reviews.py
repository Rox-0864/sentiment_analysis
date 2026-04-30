import sys
sys.path.insert(0, '.')

from datasets import load_dataset
from src.preprocessing.cleaner import clean_text
from src.models.classifier import classify_sentiment
from src.prediction.frustration_detector import detect_frustration, FrustrationResult
from src.prediction.churn_predictor import predict_churn, ChurnResult
from src.models.classifier import SentimentResult
import pandas as pd
import time
import argparse


# Dataset mapping for datasets that ACTUALLY WORK
DATASET_CONFIG = {
    "es": {
        "name": "pysentimiento/spanish-reviews",
        "field_text": "text",
        "field_id": "review_id",
        "description": "Spanish reviews (622M total, 100 sample)",
    },
    "pt": {
        "name": "eduagarcia/reviewsentbr_fewshot",
        "field_text": "sentence",  # Note: different field name!
        "field_id": "id",
        "description": "Portuguese reviews (75 samples in few-shot version)",
    }
}


def load_reviews_sample(lang: str = "es", n_samples: int = 100):
    """Load reviews from working HuggingFace datasets."""
    config = DATASET_CONFIG[lang]
    print(f"Loading {n_samples} {lang} reviews from {config['name']}...")
    print(f"Note: {config['description']}")
    
    dataset = load_dataset(config["name"], split="train", streaming=True)
    
    reviews = []
    for i, example in enumerate(dataset):
        if i >= n_samples:
            break
        text = example.get(config["field_text"], "")
        reviews.append({
            "text": text,
            "review_id": example.get(config["field_id"], None),
        })
    
    print(f"Loaded {len(reviews)} reviews")
    return reviews


def process_reviews(reviews: list[dict], lang: str = "es") -> list[dict]:
    """Process reviews through the full pipeline."""
    results = []
    
    for i, review in enumerate(reviews):
        text = review["text"]
        
        # Clean
        cleaned = clean_text(text, lang=lang)
        
        # Sentiment
        sentiment = classify_sentiment(cleaned, lang=lang)
        
        # Frustration
        frustration = detect_frustration(cleaned, sentiment, lang=lang)
        
        results.append({
            "review_id": review.get("review_id"),
            "text_original": text,
            "text_clean": cleaned,
            "sentiment": sentiment.label,
            "confidence": sentiment.confidence,
            "frustrated": frustration.is_frustrated,
            "frustration_intensity": frustration.intensity,
            "lang": sentiment.lang,
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(reviews)} reviews...")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Process reviews through sentiment pipeline")
    parser.add_argument("--lang", choices=["es", "pt"], default="es", help="Language: es or pt")
    parser.add_argument("--samples", type=int, default=100, help="Number of reviews to process")
    args = parser.parse_args()
    
    # Load sample
    reviews = load_reviews_sample(lang=args.lang, n_samples=args.samples)
    
    # Process through pipeline
    print("Processing through pipeline...")
    start = time.time()
    results = process_reviews(reviews, lang=args.lang)
    elapsed = time.time() - start
    print(f"Processed {len(results)} reviews in {elapsed:.2f}s ({elapsed/len(results):.2f}s per review)")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Summary stats
    print("\n=== Summary ===")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    print(f"\nFrustration rate: {df['frustrated'].mean()*100:.1f}%")
    
    # Predict churn for consecutive groups (simulating conversations)
    print("\n=== Churn Prediction (simulated conversations) ===")
    for i in range(0, len(results), 3):
        window = results[i:i+3]
        churn = predict_churn([
            {"text": r["text_clean"], "sentiment": SentimentResult(r["sentiment"], r["confidence"], r["lang"]), 
             "frustration": FrustrationResult(r["frustrated"], r["frustration_intensity"], [])}
            for r in window
        ])
        if churn.risk in ["high", "medium"]:
            print(f"  Conversation {i//3}: {churn.risk} risk - {churn.reason}")
    
    # Save to CSV for dashboard
    output_file = f"data/{args.lang}_reviews_sample.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
