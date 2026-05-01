import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import os


st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("📊 E-Commerce Reviews Dashboard")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    
    date_range = st.date_input(
        "Select Date Range",
        value=(datetime.now() - timedelta(days=7), datetime.now()),
        max_value=datetime.now()
    )
    
    category = st.selectbox(
        "Filter by Category",
        options=["All", "Electronics", "Clothing", "Home & Kitchen", "Sports", "Office"],
        index=0
    )
    
    min_rating = st.slider("Minimum Rating", 1, 5, 1)
    
    model_option = st.selectbox(
        "Select Model",
        options=["TF-IDF + LogReg", "BERT (RoBERTuito/BERTimbau)", "Compare Both"],
        index=2
    )

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📈 TF-IDF Details", "🆚 Model Comparison", "📊 BERT Details", "⚡ Performance"])

# Load data for all tabs
data = None
comparison_data = None

# Load e-commerce reviews data
csv_path = "data/ecommerce_reviews.csv"
comparison_path = "data/ecommerce_comparison.csv"

if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
    data["date"] = pd.to_datetime(data["date"])
    
    # Apply filters
    if category != "All":
        data = data[data["category"] == category]
    data = data[data["rating"] >= min_rating]

if os.path.exists(comparison_path):
    comparison_data = pd.read_csv(comparison_path)
    comparison_data["date"] = pd.to_datetime(comparison_data["date"])
    
    # Apply same filters
    if category != "All":
        comparison_data = comparison_data[comparison_data["category"] == category]
    comparison_data = comparison_data[comparison_data["rating"] >= min_rating]

# Tab 1: Overview
with tab1:
    if data is None or len(data) == 0:
        st.warning("No data available for the selected filters.")
    else:
        st.info(f"Loaded {len(data)} reviews from e-commerce data")
        
        st.subheader("Sentiment Distribution by Category")
        sentiment_col = "sentiment" if model_option == "TF-IDF + LogReg" else "bert_sentiment"
        if sentiment_col not in data.columns:
            sentiment_col = "sentiment"
        
        category_sentiment = data.groupby(["category", sentiment_col]).size().reset_index(name="count")
        fig_category = px.bar(
            category_sentiment, x="category", y="count", color=sentiment_col,
            title="Sentiment Count by Product Category", 
            labels={"count": "Reviews", "category": "Product Category"}
        )
        st.plotly_chart(fig_category, width='stretch')
        
        st.subheader("Overall Rating Distribution")
        rating_counts = data["rating"].value_counts().reset_index()
        rating_counts.columns = ["rating", "count"]
        fig_rating = px.pie(rating_counts, values="count", names="rating", title="Rating Distribution")
        st.plotly_chart(fig_rating, width='stretch')
        
        st.subheader("Sentiment vs Rating")
        fig_scatter = px.scatter(
            data, x="rating", y=sentiment_col, color="category",
            title="Sentiment vs Rating by Category",
            labels={"rating": "Rating (1-5)", "sentiment": "Sentiment"}
        )
        st.plotly_chart(fig_scatter, width='stretch')

# Tab 2: Model Comparison
with tab2:
    if comparison_data is not None:
        st.info(f"Comparing TF-IDF vs BERT on {len(comparison_data)} reviews")
        
        # Agreement rate
        if "both_agree" in comparison_data.columns:
            agreement_rate = comparison_data["both_agree"].mean() * 100
            st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
        
        # Side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("TF-IDF + LogReg")
            tfidf_counts = comparison_data["tfidf_sentiment"].value_counts()
            fig_tfidf = px.pie(
                values=tfidf_counts.values, 
                names=tfidf_counts.index,
                title="TF-IDF Predictions"
            )
            st.plotly_chart(fig_tfidf, width='stretch')
        
        with col2:
            st.subheader("BERT (RoBERTuito/BERTimbau)")
            bert_counts = comparison_data["bert_sentiment"].value_counts()
            fig_bert = px.pie(
                values=bert_counts.values,
                names=bert_counts.index,
                title="BERT Predictions"
            )
            st.plotly_chart(fig_bert, width='stretch')
        
        # Detailed comparison table
        st.subheader("Detailed Comparison")
        display_cols = ["product_name", "rating", "review_text", "tfidf_sentiment", "bert_sentiment", "both_agree"]
        st.dataframe(comparison_data[display_cols], width='stretch')
    else:
        st.warning("Comparison data not found. Run 'python compare_models.py' first.")

# Tab 3: BERT Details
with tab3:
    if comparison_data is not None:
        st.info(f"BERT model details for {len(comparison_data)} reviews")
        
        # BERT confidence distribution
        st.subheader("BERT Confidence Distribution")
        fig_conf = px.histogram(
            comparison_data, x="bert_confidence", color="bert_sentiment",
            title="BERT Prediction Confidence",
            labels={"bert_confidence": "Confidence", "count": "Reviews"}
        )
        st.plotly_chart(fig_conf, width='stretch')
        
        # BERT sentiment by category
        st.subheader("BERT Sentiment by Category")
        bert_category = comparison_data.groupby(["category", "bert_sentiment"]).size().reset_index(name="count")
        fig_bert_cat = px.bar(
            bert_category, x="category", y="count", color="bert_sentiment",
            title="BERT Sentiment by Product Category",
            labels={"count": "Reviews", "category": "Product Category"}
        )
        st.plotly_chart(fig_bert_cat, width='stretch')
    else:
        st.warning("BERT data not found. Run 'python compare_models.py' first.")

# Tab 4: Performance Metrics
with tab4:
    if comparison_data is not None:
        st.info(f"Performance comparison for {len(comparison_data)} reviews")
        
        # Accuracy metrics
        if "tfidf_accuracy" in comparison_data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Accuracy (vs Rating)")
                tfidf_acc = comparison_data["tfidf_accuracy"].iloc[0]
                bert_acc = comparison_data["bert_accuracy"].iloc[0]
                
                metrics_df = pd.DataFrame({
                    "Model": ["TF-IDF + LogReg", "BERT (RoBERTuito)"],
                    "Accuracy": [tfidf_acc, bert_acc]
                })
                
                fig_acc = px.bar(
                    metrics_df, x="Model", y="Accuracy",
                    title="Model Accuracy Comparison",
                    text="Accuracy",
                    color="Model",
                    labels={"Accuracy": "Accuracy (%)"}
                )
                fig_acc.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_acc, width='stretch')
                
                st.metric("TF-IDF Accuracy", f"{tfidf_acc:.1f}%")
                st.metric("BERT Accuracy", f"{bert_acc:.1f}%")
            
            with col2:
                st.subheader("⚡ Processing Time")
                tfidf_time = comparison_data["tfidf_time_ms"].iloc[0]
                bert_time = comparison_data["bert_time_ms"].iloc[0]
                
                time_df = pd.DataFrame({
                    "Model": ["TF-IDF + LogReg", "BERT (RoBERTuito)"],
                    "Time (ms)": [tfidf_time, bert_time]
                })
                
                fig_time = px.bar(
                    time_df, x="Model", y="Time (ms)",
                    title="Processing Time per Review",
                    text="Time (ms)",
                    color="Model",
                    labels={"Time (ms)": "Time (milliseconds)"}
                )
                fig_time.update_traces(texttemplate='%{text:.1f}ms', textposition='outside')
                st.plotly_chart(fig_time, width='stretch')
                
                st.metric("TF-IDF Time", f"{tfidf_time:.1f} ms/review")
                st.metric("BERT Time", f"{bert_time:.1f} ms/review")
                
                speedup = bert_time / tfidf_time
                st.warning(f"BERT is {speedup:.0f}x slower than TF-IDF")
        
        # Agreement rate
        if "both_agree" in comparison_data.columns:
            st.subheader("🤝 Model Agreement")
            agreement_rate = comparison_data["both_agree"].mean() * 100
            st.progress(agreement_rate / 100)
            st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
    else:
        st.warning("Performance data not found. Run 'python compare_models.py' first.")
