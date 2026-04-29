import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import os


st.set_page_config(page_title="ConversaAI Sentiment Dashboard", layout="wide")

st.title("📊 ConversaAI Sentiment Analysis Dashboard")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    
    date_range = st.date_input(
        "Select Date Range",
        value=(datetime.now() - timedelta(days=7), datetime.now()),
        max_value=datetime.now()
    )
    
    language = st.selectbox(
        "Filter by Language",
        options=["All", "es", "pt"],
        index=0
    )

st.header("Insights")

data = None

# Load real data from CSVs
csv_files = {
    "es": "data/spanish_tweets_sample.csv",
    "pt": "data/pt_tweets_sample.csv"
}

dfs = []
for lang, csv_path in csv_files.items():
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["date"] = datetime.now().date()  # No date in CSV, use today
        dfs.append(df)
    else:
        st.warning(f"File not found: {csv_path}")

if dfs:
    data = pd.concat(dfs, ignore_index=True)
    
    if language != "All":
        data = data[data["lang"] == language]
    
    # Show data source info
    st.info(f"Loaded {len(data)} messages from pipeline CSVs")
else:
    st.warning("No data available. Run `python3 process_tweets.py --lang es/pt` to generate data.")

if data is None or len(data) == 0:
    st.warning("No data available for the selected filters.")
else:
    st.subheader("Daily Sentiment Distribution")
    daily_sentiment = data.groupby(["date", "sentiment"]).size().reset_index(name="count")
    fig_daily = px.bar(
        daily_sentiment, x="date", y="count", color="sentiment",
        title="Sentiment Count by Day", labels={"count": "Messages", "date": "Date"}
    )
    st.plotly_chart(fig_daily, use_container_width=True)
    
    st.subheader("Overall Sentiment Distribution")
    sentiment_counts = data["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]
    fig_pie = px.pie(sentiment_counts, values="count", names="sentiment", title="Sentiment Proportion")
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.subheader("Frustration Detection")
    frustration_counts = data["frustrated"].value_counts().reset_index()
    frustration_counts.columns = ["frustrated", "count"]
    fig_frust = px.bar(frustration_counts, x="frustrated", y="count", title="Frustration Count")
    st.plotly_chart(fig_frust, use_container_width=True)
    
    st.subheader("Churn Risk Breakdown")
    # Simulate churn for visualization
    data["churn_risk"] = data.apply(
        lambda row: "high" if row["frustrated"] and row["sentiment"] == "negative" 
        else ("medium" if row["sentiment"] == "negative" else "low"),
        axis=1
    )
    churn_counts = data["churn_risk"].value_counts().reset_index()
    churn_counts.columns = ["risk", "count"]
    fig_churn = px.pie(churn_counts, values="count", names="risk", title="Churn Risk Distribution")
    st.plotly_chart(fig_churn, use_container_width=True)
