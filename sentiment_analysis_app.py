import streamlit as st
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns

# App Introduction
st.title("Sentiment Analysis App")
st.write("""
Welcome to the Sentiment Analysis App! 🎉

This application allows you to analyze the sentiment of text reviews. You can either:

- **Analyze a Single Review:** Enter any text review, and the app will instantly tell you if it is Positive or Negative.
- **Analyze a Dataset:** Upload a CSV file containing a column named 'review', and the app will analyze the sentiment of each review and display the results.

Get started by selecting an option from the sidebar. 👈
""")

# Sidebar Configuration
st.sidebar.title("Sentiment Analysis App")
option = st.sidebar.radio("Select an Option:", ("Analyze Single Review", "Analyze Dataset"))

# Load BERT Model
@st.cache_resource
def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    return tokenizer, model

# Load Model
tokenizer, model = load_bert_model()

# Sentiment Prediction Function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    return "Positive" if scores[1] > scores[0] else "Negative"

# Analyze Single Review
if option == "Analyze Single Review":
    st.header("Analyze Single Review")
    user_review = st.text_area("Enter your review:")
    if st.button("Analyze"):
        if user_review.strip():
            sentiment = predict_sentiment(user_review)
            color = '#4CAF50' if sentiment == 'Positive' else '#ff4c4c'
            st.markdown(
                f"<div style='background-color: {color}; padding: 15px; border-radius: 10px; text-align: center; color: white; font-weight: bold;'>{sentiment}</div>",
                unsafe_allow_html=True
            )
        else:
            st.error("Please enter a review.")

# Analyze Dataset
elif option == "Analyze Dataset":
    st.header("Analyze Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'review' in df.columns:
            df['Sentiment'] = df['review'].apply(predict_sentiment)
            st.write(df[['review', 'Sentiment']])

            st.subheader("Sentiment Distribution")
            sentiment_counts = df['Sentiment'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=["#4CAF50", "#ff4c4c"], ax=ax)
            for i, value in enumerate(sentiment_counts.values):
                ax.text(i, value, str(value), ha='center', va='bottom', fontsize=12, color='white')
            st.pyplot(fig)

            # Show counts of each sentiment
            st.write("### Review Counts:")
            for sentiment, count in sentiment_counts.items():
                st.write(f"- **{sentiment}**: {count}")
        else:
            st.error("Uploaded file must contain a 'review' column.")
