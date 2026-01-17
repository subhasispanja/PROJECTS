import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Page setup
st.set_page_config(page_title="LSTM IMDb Sentiment", page_icon="🎬")
st.title("🎬 LSTM IMDb Sentiment Analysis")
st.write("Enter a movie review to predict sentiment using an LSTM model.")

# Load model
model = tf.keras.models.load_model("LSTM_imdb_model.h5")

# Load IMDb word index
word_index = imdb.get_word_index()
INDEX_FROM = 3   # IMDb reserved indices

# ----------------------------
# Text → IMDb numerical conversion Function (backend only)

MAX_FEATURES = 1000   # MUST match training

def text_to_imdb_sequence(text):
    words = text.lower().split()
    sequence = []

    for word in words:
        idx = word_index.get(word)
        if idx is not None and idx < MAX_FEATURES:
            sequence.append(idx + INDEX_FROM)

    return sequence


# User input
review = st.text_area(
    "✍️ Enter movie review:",
    height=150,
    placeholder="This movie was amazing, the acting was brilliant and the story was engaging."
)

MAX_LEN = 200   # MUST match training

# Prediction
if st.button("🔍 Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        sequence = text_to_imdb_sequence(review)
        padded = pad_sequences([sequence], maxlen=MAX_LEN)

        prediction = model.predict(padded)[0][0]

        if prediction >= 0.5:
            st.success(f"😊 Positive Sentiment\n\nPrediction: {prediction:.2f}")
        else:
            st.error(f"😠 Negative Sentiment\n\nPrediction: {prediction:.2f}")
