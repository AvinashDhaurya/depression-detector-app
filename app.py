import streamlit as st
import re
import numpy as np
import joblib
import torch
import tensorflow as tf
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#  Load NLP resources
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return " ".join([word for word in text.split() if word not in stop_words])

#  Load models
svm_model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/tokenizer.pkl")
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
lstm_tokenizer = joblib.load(open("models/lstm_tokenizer.pkl", "rb"))
bert_tokenizer = BertTokenizer.from_pretrained("models/bert_model/")
bert_model = BertForSequenceClassification.from_pretrained("models/bert_model/")
meta_model = joblib.load("meta_model.pkl")

#  Streamlit UI
st.set_page_config(page_title="Depression Detector", layout="centered")
st.title(" Depression Detection App")
st.write("Enter a social media post to check for depressive signals using SVM, LSTM, BERT, and Meta-Model Ensemble.")

user_input = st.text_area(" Input Text", height=150)

if st.button(" Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)

        #  SVM
        svm_prob = svm_model.predict_proba(vectorizer.transform([cleaned]))[0][1]

        # LSTM
        lstm_seq = lstm_tokenizer.texts_to_sequences([cleaned])
        lstm_pad = pad_sequences(lstm_seq, maxlen=50)
        lstm_prob = lstm_model.predict(lstm_pad)[0][0]

        #  BERT
        inputs = bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            output = bert_model(**inputs)
            bert_prob = torch.nn.functional.softmax(output.logits, dim=1)[0][1].item()

        #  Meta Model
        meta_features = np.array([[svm_prob, lstm_prob, bert_prob]])
        final_pred = meta_model.predict(meta_features)[0]
        final_label = "Depressed" if final_pred == 1 else "Not Depressed"

        #  Show Results
        st.subheader(" Model Confidence Scores")
        st.write(f"SVM: `{svm_prob:.2f}`")
        st.write(f"LSTM: `{lstm_prob:.2f}`")
        st.write(f"BERT: `{bert_prob:.2f}`")

        st.success(f" **Final Prediction**: {final_label}")

