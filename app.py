import streamlit as st
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import pickle
import os

# Ensure NLTK data is available




# Load the saved Naive Bayes model
with open('nb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the CountVectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to preprocess the input text
def preprocess_text(text):
    stemmer = PorterStemmer()
    stopwords_set = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    return text

# Streamlit app
st.title('Spam Email Classifier')

st.write("""
### Model Summary

The spam mail classification model utilizes the Naive Bayes algorithm, specifically the Multinomial Naive Bayes variant. This model leverages natural language processing (NLP) techniques to preprocess email texts and classify them as either "spam" or "ham" (non-spam).

**Data Preprocessing:**
1. **Text Preprocessing**:
   - **Lowercasing**: Converts all characters to lowercase to maintain consistency.
   - **Punctuation Removal**: Eliminates punctuation marks to reduce noise in the data.
   - **Tokenization**: Splits the text into individual words (tokens).
   - **Stopwords Removal**: Removes common English stopwords (e.g., "the", "and", "is") that do not contribute to the spam/ham classification.
   - **Stemming**: Reduces words to their root forms (e.g., "running" becomes "run") to standardize the vocabulary.

2. **Vectorization**:
   - **Count Vectorizer**: Converts the preprocessed text data into a numerical format suitable for machine learning by counting the frequency of each word in the text.

**Model Training:**
- **Algorithm**: Multinomial Naive Bayes
- **Training Data**: 80% of the dataset
- **Testing Data**: 20% of the dataset

**Model Performance:**
- **Accuracy**: 97.97%
- **Precision and Recall**:
  - **Ham**: Precision = 98%, Recall = 99%, F1-Score = 99%
  - **Spam**: Precision = 97%, Recall = 96%, F1-Score = 97%

The model demonstrates high accuracy and strong performance metrics, indicating its effectiveness in distinguishing between spam and non-spam emails.
""")


# Input email text
input_email = st.text_area('Enter the email text')

if st.button('Classify'):
    if input_email:
        # Preprocess the input email
        preprocessed_email = preprocess_text(input_email)
        text_vector = vectorizer.transform([preprocessed_email]).toarray()
        
        # Predict using the Naive Bayes model
        prediction = model.predict(text_vector)[0]
        if prediction == 1:
            st.write('This email is classified as Spam.')
        else:
            st.write('This email is classified as Not Spam.')
    else:
        st.write('Please enter some text to classify.')
