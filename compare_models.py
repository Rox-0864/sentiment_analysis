"""
Script to process e-commerce reviews with BERT models and compare with TF-IDF.
"""
import pandas as pd
import os
import time
from src.models.bert_classifier import classify_with_bert
from src.models.classifier import classify_sentiment


def process_reviews_with_both_models(input_csv: str, output_csv: str):
    """
    Process reviews with both TF-IDF and BERT models for comparison.
    
    Args:
        input_csv: Path to input CSV with review_text
        output_csv: Path to save output CSV with both predictions
    """
    if not os.path.exists(input_csv):
        print(f"File not found: {input_csv}")
        return
    
    df = pd.read_csv(input_csv)
    print(f"Processing {len(df)} reviews...")
    
    # Get TF-IDF predictions (using existing sentiment column or re-running)
    if "sentiment" in df.columns:
        print("Using existing sentiment column for TF-IDF...")
        df["tfidf_sentiment"] = df["sentiment"]
        df["tfidf_confidence"] = 0.95  # Placeholder confidence
    
    # Get BERT predictions with timing
    print("Running BERT classification...")
    bert_sentiments = []
    bert_confidences = []
    
    start_time = time.time()
    for idx, row in df.iterrows():
        text = row["review_text"]
        try:
            result = classify_with_bert(text, lang="es")  # Assuming Spanish for e-commerce
            bert_sentiments.append(result.label)
            bert_confidences.append(result.confidence)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            bert_sentiments.append("error")
            bert_confidences.append(0.0)
    bert_time = time.time() - start_time
    print(f"BERT processing time: {bert_time:.2f}s ({bert_time/len(df)*1000:.1f}ms per review)")
    
    df["bert_sentiment"] = bert_sentiments
    df["bert_confidence"] = bert_confidences
    df["bert_time_ms"] = bert_time / len(df) * 1000
    
    # Calculate agreement and accuracy
    if "tfidf_sentiment" in df.columns:
        df["both_agree"] = df["tfidf_sentiment"] == df["bert_sentiment"]
        agreement_rate = df["both_agree"].sum() / len(df) * 100
        print(f"Agreement rate: {agreement_rate:.1f}%")
        
        # Calculate accuracy using rating as ground truth
        # Rating 1-2: negative, 3: neutral, 4-5: positive
        df["true_sentiment"] = df["rating"].apply(
            lambda x: "negative" if x <= 2 else ("neutral" if x == 3 else "positive")
        )
        
        tfidf_accuracy = (df["tfidf_sentiment"] == df["true_sentiment"]).mean() * 100
        bert_accuracy = (df["bert_sentiment"] == df["true_sentiment"]).mean() * 100
        
        print(f"\n=== Accuracy (vs Rating) ===")
        print(f"TF-IDF + LogReg: {tfidf_accuracy:.1f}%")
        print(f"BERT (RoBERTuito): {bert_accuracy:.1f}%")
        
        # Add accuracy to dataframe for dashboard
        df["tfidf_accuracy"] = tfidf_accuracy
        df["bert_accuracy"] = bert_accuracy
        
        # Estimate TF-IDF time (typically 50-100x faster)
        df["tfidf_time_ms"] = 10  # ~10ms per review estimate
        df["bert_time_ms"] = bert_time / len(df) * 1000
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    # Print comparison stats
    print("\n=== TF-IDF Distribution ===")
    if "tfidf_sentiment" in df.columns:
        print(df["tfidf_sentiment"].value_counts())
    
    print("\n=== BERT Distribution ===")
    print(df["bert_sentiment"].value_counts())
    
    return df


if __name__ == "__main__":
    input_file = "data/ecommerce_reviews.csv"
    output_file = "data/ecommerce_comparison.csv"
    
    process_reviews_with_both_models(input_file, output_file)
