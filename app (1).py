import re
import os
import pickle
import streamlit as st

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    layout="centered"
)

# ---------------- LOAD MODEL & VECTORIZER ---------------- #
@st.cache_resource(show_spinner=True)
def load_artifacts():
    model_path = "movie_sentiment_model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"

    # Check file existence
    if not os.path.exists(model_path):
        st.error("❌ Model file not found.")
        st.stop()

    if not os.path.exists(vectorizer_path):
        st.error("❌ Vectorizer file not found.")
        st.stop()

    # Check file size (EOFError protection)
    if os.path.getsize(model_path) == 0 or os.path.getsize(vectorizer_path) == 0:
        st.error("❌ Model or vectorizer file is empty or corrupted.")
        st.stop()

    # Load artifacts safely
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_artifacts()

# ---------------- TEXT CLEANING ---------------- #
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------- UI ---------------- #
st.title("🎬 Movie Review Sentiment Analysis")
st.write(
    "This application predicts whether a movie review expresses "
    "a **positive** or **negative** sentiment using a trained ML model."
)

st.divider()

review_text = st.text_area(
    "Enter a movie review",
    placeholder="Type or paste a movie review here...",
    height=150
)

# ---------------- PREDICTION ---------------- #
if st.button("Analyze Sentiment"):
    if review_text.strip() == "":
        st.warning("⚠️ Input text cannot be empty.")
    else:
        cleaned_text = clean_text(review_text)
        vector = vectorizer.transform([cleaned_text])

        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

        negative_prob = probabilities[0] * 100
        positive_prob = probabilities[1] * 100

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success(f"✅ **Positive Sentiment**\n\nConfidence: **{positive_prob:.2f}%**")
        else:
            st.error(f"❌ **Negative Sentiment**\n\nConfidence: **{negative_prob:.2f}%**")
