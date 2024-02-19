# Imports
import numpy as np
import pandas as pd
import pickle
import re
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to set background image
def set_bg_hack_url(image_url, width=None, height=None):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}");
            background-size: cover;
            {'width: ' + width + ';' if width else ''}
            {'height: ' + height + ';' if height else ''}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to preprocess text
def preprocess_text(text, le, custom_stopwords):
    r = re.sub('[^a-zA-Z1-9]', " ", str(text).lower())
    r = ' '.join([le.lemmatize(word) for word in r.split() if word not in custom_stopwords])
    return r

# Function to perform TF-IDF vectorization
def tfidf_vectorize(text, tfidf_vocab):
    tfidf_vectorizer = TfidfVectorizer(vocabulary=tfidf_vocab)
    text_tfidf = tfidf_vectorizer.fit_transform([text])
    return text_tfidf

# Main function
def main():
    # Set background image
    set_bg_hack_url("https://cdn.pixabay.com/photo/2017/03/09/06/30/pool-2128578_1280.jpg", width="100%", height="100%")

    # Load pre-trained SVM model
    with open('sv_d.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # Load TF-IDF Vectorizer vocabulary
    with open("tfidf_vocab.pkl", "rb") as vocab_file:
        tfidf_vocab = pickle.load(vocab_file)

    # Initialize WordNetLemmatizer and stopwords
    le = WordNetLemmatizer()
    default_stopwords = set(stopwords.words('english'))
    custom_stopwords = default_stopwords.union({'n'}) - {'not'}

    # UI
    st.title("Hotel Review Analysis App")
    st.markdown("""
    <div style ="background-color:yellow;padding:10px">
    <h1 style ="color:black;text-align:center;font-family: 'Bell MT', serif;">Sentiment Review Analysis</h1>
    </div>
    <br>
    """, unsafe_allow_html=True)

    # User input
    review = st.text_input("Text for Prediction:", placeholder="Enter the Text")

    data = st.file_uploader("Upload a CSV (or) Excel file for bulk sentiment analysis", type=["csv", "xlsx"])

    if st.button("Predict"):
        if review:
            # Preprocess input text
            processed_text = preprocess_text(review, le, custom_stopwords)

            # Sentiment Analysis
            sia = SentimentIntensityAnalyzer()
            sentiment_score = sia.polarity_scores(processed_text)['compound']

            # Perform TF-IDF vectorization
            text_tfidf = tfidf_vectorize(processed_text, tfidf_vocab)

            # Predict polarity using the pre-trained SVM model
            polarity = loaded_model.predict(text_tfidf)[0]

            # Display result
            st.write("Sentiment Score:", sentiment_score)
            st.markdown(f'<div style="background-color: {"green" if polarity == 1 else "red"}; color:white; padding: 10px">The Review is {"Positive" if
