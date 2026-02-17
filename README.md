# 🧠 LSTM Next Word Predictor

A Deep Learning project that predicts the next word in a sentence using an LSTM (Long Short-Term Memory) network, deployed with a Streamlit web application.

---

## 🚀 Overview

This project trains an LSTM model on text data to learn word sequences and predict the most probable next word given an input phrase.

Example:
Input: To be or not to
Output: be

The trained model is integrated into a simple and interactive **Streamlit app** for real-time predictions.

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Streamlit

---

## 📂 Project Structure
LSTM_NextWord_Predictor/
│
├── app.py # Streamlit web app
├── exp.ipynb # Model training notebook
├── next_word_lstm.h5 # Trained LSTM model
├── tokenizer.pickle # Saved tokenizer
├── hamlet.txt # Training dataset
├── requirements.txt
└── README.md

---

## ▶️ Run Locally

Clone the repository:

git clone https://github.com/Sushal1306/LSTM_NextWord_Predictor.git

cd LSTM_NextWord_Predictor

Install dependencies: pip install -r requirements.txt
Run the app: streamlit run app.py

---

## 🧠 How It Works

1. Text is tokenized into sequences.
2. Sequences are padded to fixed length.
3. LSTM learns contextual patterns.
4. Model predicts the most probable next word using Softmax.

---

## ✨ Future Improvements

- Add top-k predictions with probabilities
- Improve accuracy with more epochs
- Improve UI design  
- Train on larger datasets  
- Deploy publicly  

---
## 📌 Author
**Sushal Devasari**

<img width="1919" height="917" alt="Screenshot 2026-02-17 165402" src="https://github.com/user-attachments/assets/83f1ec7b-40ea-474d-be94-8f5fe4080ef0" />



