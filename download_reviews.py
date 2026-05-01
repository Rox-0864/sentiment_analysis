"""
Script to download real e-commerce review datasets and merge them for training.
"""
import pandas as pd
from datasets import load_dataset
import os

def download_spanish_reviews():
    """Download Spanish Amazon reviews from Hugging Face."""
    print("Downloading Spanish Amazon reviews (210k samples)...")
    try:
        dataset = load_dataset("SetFit/amazon_reviews_multi_es", split="train")
        df = dataset.to_pandas()
        
        # Keep only relevant columns
        df = df[["text", "label"]]
        df.columns = ["text", "sentiment"]
        
        # Convert labels (0=negative, 1=positive)
        df["sentiment"] = df["sentiment"].map({0: "negative", 1: "positive"})
        
        # Save as CSV
        output_path = "data/spanish_amazon_reviews.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} Spanish reviews to {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error downloading Spanish reviews: {e}")
        return None

def download_portuguese_reviews():
    """Download Portuguese e-commerce reviews from Hugging Face."""
    print("Downloading Portuguese e-commerce reviews (20k samples)...")
    try:
        # Try IberaSoft (20k reviews, verified)
        dataset = load_dataset("IberaSoft/ecommerce-reviews-sentiment", split="train")
        df = dataset.to_pandas()

        # Keep only relevant columns
        df = df[["text", "label"]]
        df.columns = ["text", "label_num"]

        # Convert labels (0=negative, 1=neutral, 2=positive)
        df["sentiment"] = df["label_num"].map({
            0: "negative",
            1: "neutral",
            2: "positive"
        })

        # Filter only positive/negative for binary classification
        df = df[df["sentiment"].isin(["positive", "negative"])]

        # Save as CSV
        output_path = "data/portuguese_ecommerce_reviews.csv"
        df[["text", "sentiment"]].to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} Portuguese reviews to {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error downloading Portuguese reviews: {e}")
        return None

if __name__ == "__main__":
    print("=== Downloading Real E-Commerce Reviews ===\n")
    
    # Download Spanish reviews
    es_path = download_spanish_reviews()
    
    print()
    
    # Download Portuguese reviews
    pt_path = download_portuguese_reviews()
    
    print("\n=== Summary ===")
    if es_path and os.path.exists(es_path):
        df = pd.read_csv(es_path)
        print(f"Spanish: {len(df)} reviews")
        print(df["sentiment"].value_counts())
    
    if pt_path and os.path.exists(pt_path):
        df = pd.read_csv(pt_path)
        print(f"\nPortuguese: {len(df)} reviews")
        print(df["sentiment"].value_counts())
