# BBC News Article Classification

This project demonstrates classifying BBC news articles into categories (business, entertainment, politics, sport, tech) using both a Naive Bayes model and a fine-tuned DistilBERT model. It also includes a simple Streamlit web application for interactive classification.

## How to Run the Notebook

The analysis and model training were performed in a Google Colab notebook. To run the notebook:

1.  Open the notebook file in Google Colab.
2.  Run all the code cells sequentially. The notebook will:
    *   Load the dataset from Kaggle.
    *   Preprocess the text data (cleaning, tokenization, lemmatization).
    *   Train and evaluate a Naive Bayes classifier.
    *   Train and evaluate a DistilBERT classifier.
    *   Compare the performance of the two models.
    *   Save the trained Naive Bayes model and TF-IDF vectorizer.

## How to Run the Streamlit App

A simple Streamlit web application is included to classify news articles interactively using the trained Naive Bayes model.

1.  Ensure you have Python and pip installed.
2.  Install the required dependencies by running the following command in your terminal in the project directory:
3.  Click on this link to run the Streamlit app: https://bbc-news-article-classifier-bxgevteb7rtnjjneuqokhc.streamlit.app/
