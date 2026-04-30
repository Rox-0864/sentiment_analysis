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

st.header("Insights")

data = None

# Load e-commerce reviews data
csv_path = "data/ecommerce_reviews.csv"

if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
    data["date"] = pd.to_datetime(data["date"])
    
    # Apply filters
    if category != "All":
        data = data[data["category"] == category]
    data = data[data["rating"] >= min_rating]
    
    # Show data source info
    st.info(f"Loaded {len(data)} reviews from e-commerce data")
else:
    st.warning(f"File not found: {csv_path}")

if data is None or len(data) == 0:
    st.warning("No data available for the selected filters.")
else:
    st.subheader("Sentiment Distribution by Category")
    category_sentiment = data.groupby(["category", "sentiment"]).size().reset_index(name="count")
    fig_category = px.bar(
        category_sentiment, x="category", y="count", color="sentiment",
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
        data, x="rating", y="sentiment", color="category",
        title="Sentiment vs Rating by Category",
        labels={"rating": "Rating (1-5)", "sentiment": "Sentiment"}
    )
    st.plotly_chart(fig_scatter, width='stretch')
    
    st.subheader("Reviews by Date")
    daily_reviews = data.groupby(["date", "sentiment"]).size().reset_index(name="count")
    fig_daily = px.line(
        daily_reviews, x="date", y="count", color="sentiment",
        title="Daily Review Count", labels={"count": "Reviews", "date": "Date"}
    )
    st.plotly_chart(fig_daily, width='stretch')
