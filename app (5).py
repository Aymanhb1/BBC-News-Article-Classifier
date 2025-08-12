import streamlit as st
import joblib
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ssl
import os # Import os module

# Set NLTK data path to a writable directory
nltk_data_path = os.path.join("/tmp", "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)


# Download necessary NLTK data (if not already downloaded)
# Use a more direct approach for downloading and check if already present
try:
    _create_unverified_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_context

# List of NLTK data packages to download
nltk_packages = ['punkt', 'stopwords', 'wordnet']

for package in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{package}') # Check if already downloaded
    except LookupError: # Handle cases where the package is not found initially
        nltk.download(package, download_dir=nltk_data_path, quiet=True)
    except Exception as e: # Catch any other exceptions during download
        st.error(f"Error downloading NLTK package {package}: {e}")


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join tokens back into a string
    return ' '.join(cleaned_tokens)

# Load the trained model and vectorizer
try:
    naive_bayes_model = joblib.load('naive_bayes_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'naive_bayes_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
    st.stop()


# Define the categories (make sure this matches the order used during training)
categories = ['business', 'entertainment', 'politics', 'sport', 'tech']

st.title("BBC News Article Classifier")
st.write("Enter a news article to classify its category.")

# Text input area
article_text = st.text_area("Enter Article Text", height=300)

if st.button("Classify"):
    if article_text:
        # Clean the input text
        cleaned_article = clean_text(article_text)

        # Vectorize the cleaned text
        article_tfidf = tfidf_vectorizer.transform([cleaned_article])

        # Get prediction probabilities
        probabilities = naive_bayes_model.predict_proba(article_tfidf)[0]

        # Get the predicted category index and confidence
        predicted_category_index = naive_bayes_model.predict(article_tfidf)[0]
        confidence = probabilities[predicted_category_index]

        # Get the predicted category name
        predicted_category = categories[predicted_category_index]

        st.subheader("Classification Result:")
        st.write(f"Predicted Category: **{predicted_category.capitalize()}**")
        st.write(f"Confidence: **{confidence:.2f}**")

        # Display probabilities for all categories
        st.subheader("Category Probabilities:")
        prob_df = pd.DataFrame({'Category': categories, 'Probability': probabilities})
        prob_df = prob_df.sort_values(by='Probability', ascending=False)
        st.dataframe(prob_df)

    else:
        st.warning("Please enter some text to classify.")
