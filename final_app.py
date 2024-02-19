import streamlit as st
import numpy as np
import pandas as pd
import pickle
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import eli5
nltk.download('stopwords')
nltk.download('wordnet')

from PIL import Image

logo=Image.open('/content/reviews_icon.png')
st.set_page_config(page_title="Hotel Review Analysis App",page_icon=logo)

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

# Example usage
set_bg_hack_url("https://cdn.pixabay.com/photo/2017/03/09/06/30/pool-2128578_1280.jpg", width="100%", height="100%")

# Load pre-trained SVM model
with open('/content/sv_d.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

html_temp = """
<div style ="background-color:yellow;padding:10px">
<h1 style ="color:black;text-align:center;font-family: 'Bell MT', serif;">Sentiment Review Analysis</h1>
</div>
<br>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# Developing custom stopwords involves modifying existing stop words by altering a few words
default_stopwords = set(stopwords.words('english'))
custom_stopwords = default_stopwords.union({'n'}) - {'not'}

# Initialize WordNetLemmatizer
le = WordNetLemmatizer()

# Preprocess input text
def preprocess_text(text):
    r = re.sub('[^a-zA-Z1-9]', " ", str(text))
    r = r.lower()
    r = r.split()
    r = [le.lemmatize(word) for word in r if word not in custom_stopwords]
    text = " ".join(r)
    return text

# Load TF-IDF Vectorizer vocabulary
with open("tfidf_vocab.pkl", "rb") as vocab_file:
    tfidf_vocab = pickle.load(vocab_file)

# Function to perform TF-IDF vectorization
def tfidf_vectorize(text):
    # Load TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(vocabulary=tfidf_vocab)
    # Transform the input text
    text_tfidf = tfidf_vectorizer.fit_transform([text])
    return text_tfidf

# User input
review = st.text_input("Text for Prediction:", placeholder="Enter the Text")

data= st.file_uploader(
    "Upload  a CSV (or) Excel file for the prediction - Upload the file and click on Predict",
    type=["csv","xlsx"]
)
if st.button("Predict"):
    if review:  # Check if review text is provided
        # Preprocess input text
        processed_text = preprocess_text(review)

        # Sentiment Analysis
        nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(processed_text)['compound']

        # Perform TF-IDF vectorization
        text_tfidf = tfidf_vectorize(processed_text)

        # Predict polarity using the pre-trained SVM model
        polarity = loaded_model.predict(text_tfidf)[0]

        # Display the result
        st.write("Sentiment Score:", sentiment_score)
        if polarity == 1:
            st.markdown('<div style="background-color:  #333333; color:green ;padding: 10px">The Review is Positive 😃</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background-color:  #333333; color:red; padding: 10px">The Review is Negative 😞</div>', unsafe_allow_html=True)
    else:
        if data is not None:
            if data.name.split('.')[-1] == 'csv':
                data = pd.read_csv(data)
            elif data.name.split('.')[-1] == 'xlsx':
                data = pd.read_excel(data)

            # Preprocess text in the 'Review' column
            data["preprocessed_text"] = data['Review'].apply(preprocess_text)

            # Sentiment Analysis
            nltk.download('vader_lexicon')
            sia = SentimentIntensityAnalyzer()
            data['sentiment_score'] = data['preprocessed_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
            data["polarity"] = np.where(data["sentiment_score"] > 0, "positive", "negative")
            # Display the preprocessed data
            data1=data.drop(["Review","Rating"],axis=1)
            st.write(data1)
        else:
            st.write("Please upload a file to perform prediction.")
