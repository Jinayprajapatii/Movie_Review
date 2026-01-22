import re
import pickle
import streamlit as st

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analysis By Jinay",
    layout="centered"
)

# -------------------- Load Model & Vectorizer (NO CACHE) --------------------
with open("movie_sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -------------------- Text Cleaning --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------- UI --------------------
st.title("Movie Review Sentiment Analysis By Jinay")
st.write(
    "This application predicts whether a movie review expresses "
    "a positive or negative sentiment using a trained machine learning model."
)

st.divider()

review_text = st.text_input(
    "Enter a movie review",
    placeholder="Type or paste a movie review here..."
)

# -------------------- Prediction --------------------
if st.button("Analyze Sentiment"):
    if review_text.strip() == "":
        st.error("Input text cannot be empty.")
    else:
        cleaned = clean_text(review_text)
        vector = vectorizer.transform([cleaned])

        pred_label = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        negative_prob = prob[0] * 100
        positive_prob = prob[1] * 100

        st.subheader("Prediction Result")

        if pred_label == 1:
            st.success(
                f"Sentiment :- Positive\n\n"
                f"Confidence: {positive_prob:.2f}%"
            )
        else:
            st.error(
                f"Sentiment :- Negative\n\n"
                f"Confidence: {negative_prob:.2f}%"
            )

