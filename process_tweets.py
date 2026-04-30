import sys
sys.path.insert(0, '.')

from datasets import load_dataset
from src.preprocessing.cleaner import clean_text
from src.models.sentiment_classifier import classify_sentiment
from src.prediction.frustration_detector import detect_frustration, FrustrationResult
from src.prediction.churn_predictor import predict_churn, ChurnResult
from src.models.sentiment_classifier import SentimentResult
import pandas as pd
import time
import argparse


# Dataset mapping for datasets that ACTUALLY WORK
DATASET_CONFIG = {
    "es": {
        "name": "pysentimiento/spanish-tweets",
        "field_text": "text",
        "field_id": "tweet_id",
        "description": "Spanish tweets (622M total, 100 sample)",
    },
    "pt": {
        "name": "eduagarcia/tweetsentbr_fewshot",
        "field_text": "sentence",  # Note: different field name!
        "field_id": "id",
        "description": "TweetSentBR Portuguese (75 samples in few-shot version)",
    }
}


def load_tweets_sample(lang: str = "es", n_samples: int =100):
    """Load tweets from working HuggingFace datasets."""
    config = DATASET_CONFIG[lang]
    print(f"Loading {n_samples} {lang} tweets from {config['name']}...")
    print(f"Note: {config['description']}")
    
    dataset = load_dataset(config["name"], split="train", streaming=True)
    
    tweets = []
    for i, example in enumerate(dataset):
        if i >= n_samples:
            break
        text = example.get(config["field_text"], "")
        tweets.append({
            "text": text,
            "tweet_id": example.get(config["field_id"], None),
        })
    
    print(f"Loaded {len(tweets)} tweets")
    return tweets


def process_tweets(tweets: list[dict], lang: str = "es") -> list[dict]:
    """Process tweets through the full pipeline."""
    results = []
    
    for i, tweet in enumerate(tweets):
        text = tweet["text"]
        
        # Clean
        cleaned = clean_text(text, lang=lang)
        
        # Sentiment
        sentiment = classify_sentiment(cleaned, lang=lang)
        
        # Frustration
        frustration = detect_frustration(cleaned, sentiment, lang=lang)
        
        results.append({
            "tweet_id": tweet.get("tweet_id"),
            "text_original": text,
            "text_clean": cleaned,
            "sentiment": sentiment.label,
            "confidence": sentiment.confidence,
            "frustrated": frustration.is_frustrated,
            "frustration_intensity": frustration.intensity,
            "lang": sentiment.lang,
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(tweets)} tweets...")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Process tweets through sentiment pipeline")
    parser.add_argument("--lang", choices=["es", "pt"], default="es", help="Language: es or pt")
    parser.add_argument("--samples", type=int, default=100, help="Number of tweets to process")
    args = parser.parse_args()
    
    # Load sample
    tweets = load_tweets_sample(lang=args.lang, n_samples=args.samples)
    
    # Process through pipeline
    print("Processing through pipeline...")
    start = time.time()
    results = process_tweets(tweets, lang=args.lang)
    elapsed = time.time() - start
    print(f"Processed {len(results)} tweets in {elapsed:.2f}s ({elapsed/len(results):.2f}s per tweet)")
    
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
    output_file = f"data/{args.lang}_tweets_sample.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
