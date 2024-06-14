import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import nltk

# Download stopwords
nltk.download('stopwords')

# Load the saved model
model = load_model('spam_classifier_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Initialize the stemmer and stopwords
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    return text

# Streamlit app
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

user_input = st.text_area('Email text')

if st.button('Classify'):
    if user_input:
        # Preprocess the input
        preprocessed_text = preprocess_text(user_input)
        # Tokenize the input
        tokenized_text = tokenizer.texts_to_sequences([preprocessed_text])
        # Pad the sequence
        padded_text = pad_sequences(tokenized_text, maxlen=100)

        # Predict
        prediction = model.predict(padded_text)[0][0]

        # Output result
        if prediction > 0.5:
            st.write('This email is **spam**.')
        else:
            st.write('This email is **not spam**.')
    else:
        st.write('Please enter some text.')
