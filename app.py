import streamlit as st
import joblib
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ssl
import os

# Set NLTK data path to a writable directory
nltk_data_path = os.path.join("/tmp", "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download necessary NLTK data (if not already downloaded)
try:
    _create_unverified_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_context

# List of NLTK data packages to download
nltk_packages = ['punkt', 'stopwords', 'wordnet']

@st.cache_resource
def download_nltk_data():
    """Download NLTK data packages with caching"""
    for package in nltk_packages:
        try:
            # Check different possible paths for the package
            if package == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif package == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif package == 'wordnet':
                nltk.data.find('corpora/wordnet')
        except LookupError:
            try:
                nltk.download(package, download_dir=nltk_data_path, quiet=True)
            except Exception as e:
                st.error(f"Error downloading NLTK package {package}: {e}")
                return False
    return True

# Download NLTK data
if not download_nltk_data():
    st.error("Failed to download required NLTK data. App may not function correctly.")
    st.stop()

# Initialize NLTK components
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    st.error(f"Error initializing NLTK components: {e}")
    st.stop()

def clean_text(text):
    """Clean and preprocess text"""
    try:
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
    except Exception as e:
        st.error(f"Error cleaning text: {e}")
        return text

@st.cache_resource
def load_models():
    """Load the trained model and vectorizer with caching"""
    try:
        # Check if files exist
        if not os.path.exists('naive_bayes_model.pkl'):
            st.error("Model file 'naive_bayes_model.pkl' not found.")
            return None, None
        
        if not os.path.exists('tfidf_vectorizer.pkl'):
            st.error("Vectorizer file 'tfidf_vectorizer.pkl' not found.")
            return None, None
        
        # Load models
        naive_bayes_model = joblib.load('naive_bayes_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        
        return naive_bayes_model, tfidf_vectorizer
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("This often happens when:")
        st.error("1. Required dependencies (like scikit-learn) are missing")
        st.error("2. Model was trained with different library versions")
        st.error("3. Model files are corrupted")
        return None, None

# Load models
naive_bayes_model, tfidf_vectorizer = load_models()

if naive_bayes_model is None or tfidf_vectorizer is None:
    st.error("Failed to load models. Please check the error messages above.")
    st.stop()

# Define the categories (make sure this matches the order used during training)
categories = ['business', 'entertainment', 'politics', 'sport', 'tech']

st.title("BBC News Article Classifier")
st.write("Enter a news article to classify its category.")

# Text input area
article_text = st.text_area("Enter Article Text", height=300, 
                           placeholder="Paste your news article text here...")

if st.button("Classify", type="primary"):
    if article_text.strip():
        try:
            # Clean the input text
            cleaned_article = clean_text(article_text)
            
            if not cleaned_article.strip():
                st.warning("The text appears to be empty after cleaning. Please try with different content.")
                st.stop()

            # Vectorize the cleaned text
            article_tfidf = tfidf_vectorizer.transform([cleaned_article])

            # Get prediction probabilities
            probabilities = naive_bayes_model.predict_proba(article_tfidf)[0]

            # Get the predicted category index and confidence
            predicted_category_index = naive_bayes_model.predict(article_tfidf)[0]
            confidence = probabilities[predicted_category_index]

            # Get the predicted category name
            predicted_category = categories[predicted_category_index]

            # Display results
            st.success("Classification completed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Prediction")
                st.metric("Category", predicted_category.capitalize())
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col2:
                st.subheader("📊 All Probabilities")
                prob_df = pd.DataFrame({
                    'Category': [cat.capitalize() for cat in categories], 
                    'Probability': [f"{prob:.1%}" for prob in probabilities]
                })
                prob_df = prob_df.sort_values(by='Probability', ascending=False)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
            st.error("Please try again with different text.")

    else:
        st.warning("⚠️ Please enter some text to classify.")

# Add some example text
with st.expander("📝 Try with example text"):
    example_text = """
    Apple Inc. reported strong quarterly earnings today, beating analyst expectations. 
    The tech giant's revenue increased by 15% compared to the same period last year, 
    driven by robust iPhone sales and growing services revenue. The company's stock 
    price rose 5% in after-hours trading following the announcement.
    """
    if st.button("Use this example"):
        st.rerun()
