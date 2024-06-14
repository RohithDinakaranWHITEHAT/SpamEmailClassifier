import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string



# Load the saved model
model = load_model('spam_classifier_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    stemmer = PorterStemmer()
    stopwords_set = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])
    words = [stemmer.stem(word) for word in words if word not in stopwords_set]
    return ' '.join(words)

def classify_email(text):
    preprocessed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=500)
    prediction = model.predict(padded_sequence)
    return 'Spam' if prediction > 0.5 else 'Not Spam'

st.title('Spam Email Classifier')
# Model description
st.header('About the Model')
st.write('''
This application uses a Convolutional Neural Network (CNN) combined with a Long Short-Term Memory (LSTM) network for classifying emails as spam or not spam. 
The model was trained on a dataset of emails, preprocessed by converting text to lowercase, removing punctuation and stopwords, and stemming the words. 
The preprocessed text was then tokenized and padded to ensure uniform input length. 

The CNN-LSTM hybrid model leverages the strengths of both CNNs (for extracting local patterns) and LSTMs (for capturing long-term dependencies) in text data, 
resulting in a high accuracy of 98% on the test data. This allows for efficient and accurate classification of emails, helping users identify spam effectively.
''')

st.write('Enter an email to check if it is spam or not.')

input_email = st.text_area('Enter the email text:')
if st.button('Classify'):
    result = classify_email(input_email)
    st.write(f'The email is: {result}')
