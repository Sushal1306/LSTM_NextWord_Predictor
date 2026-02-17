import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🤖",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        background: -webkit-linear-gradient(#00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .subtitle {
        text-align: center;
        color: #cfcfcf;
        font-size: 18px;
        margin-bottom: 40px;
    }
    .prediction-box {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        color: #00ffcc;
        margin-top: 20px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 45px;
        font-size: 16px;
        font-weight: 600;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = load_model('next_word_lstm.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# ---------------- PREDICTION FUNCTION ----------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
        
    token_list = pad_sequences([token_list], 
                               maxlen=max_sequence_len-1, 
                               padding='pre')
    
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# ---------------- UI ----------------
st.markdown('<div class="title">Next Word Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by LSTM</div>', unsafe_allow_html=True)

input_text = st.text_input(
    "Enter your sentence:",
    "To be or not to"
)

if st.button("Predict Next Word 🚀"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    
    if next_word:
        st.markdown(
            f'<div class="prediction-box">Next Word: {next_word}</div>',
            unsafe_allow_html=True
        )
    else:
        st.warning("Could not predict a word. Try a different input.")
