import streamlit as st
import pandas as pd
import joblib
import re
import os
import pickle 
import nltk
from nltk import WordNetLemmatizer
from nltk.tokenize.punkt import PunktSentenceTokenizer


# Add local nltk data path
nltk.data.path.append("./nltk_data")

try:
    with open("nltk_data/tokenizers/punkt/english.pickle", "rb") as f:
        sentence_tokenizer = pickle.load(f)
except FileNotFoundError:
    st.error("Missing punkt tokenizer file. Please ensure 'nltk_data/tokenizers/punkt/english.pickle' exists.")
    st.stop()


# Download NLTK models if not found
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


def identity_function(x):
    return x


# Load saved model artifacts
model = joblib.load("clinical_note_model.pkl")
vectorizer = joblib.load("clinical_note_vectorizer.pkl")
label_encoder = joblib.load("clinical_note_label_encoder.pkl")

lemmatizer = WordNetLemmatizer()

# Preprocessing
def clean_note(text):
    boilerplate = [
        'clinical note', 'patient name', 'chief complaint', 'history of present illness',
        'visit date', 'signature', 'date of birth', 'subjective', 'objective', 'dob'
    ]
    pattern = '|'.join([r'\b' + re.escape(p) + r'\b' for p in boilerplate])
    text = re.sub(pattern, '', str(text), flags=re.IGNORECASE)
    text = re.sub(r'-', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'[^\w\s\n]', '', text)
    text = re.sub(r'(\n)+', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # Tokenize using  Punkt tokenizer
    tokens = [word for sent in sentence_tokenizer.tokenize(text.lower()) for word in nltk.word_tokenize(sent)]
    lemmas = [lemmatizer.lemmatize(w, 'v') for w in tokens]
    return lemmas


# Streamlit UI
st.title("Clinical Note Classifier")

uploaded_file = st.file_uploader("Upload your clinical notes CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df)

    if 'Clinical Note' not in df.columns:
        st.error("File must contain a 'Clinical Note' column.")
    else:
        st.info("Processing and classifying notes...")


        # Clean and vectorize
        processed_notes = df["Clinical Note"].apply(clean_note)
        X_new = vectorizer.transform(processed_notes)
        
        # Predict
        preds = model.predict(X_new)
        preds_labels = label_encoder.inverse_transform(preds)
        df["Prediction"] = preds_labels

        st.subheader("Predictions")
        st.dataframe(df)

        # Download option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
