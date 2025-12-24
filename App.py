import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# ✅ Page config MUST be first Streamlit command
# -----------------------------
st.set_page_config(
    page_title="Sarcasm Detector",
    page_icon="😏",
    layout="centered"
)

# -----------------------------
# 1. Text Cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z0-9\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# 2. Load dataset & train model
# -----------------------------
@st.cache_resource  # cache model so it doesn't retrain every reload
def load_model():
    df = pd.read_csv("train.csv")
    df.columns = ["tweets", "class"]
    df["label"] = df["class"].apply(lambda x: 1 if x.lower() == "sarcasm" else 0)
    df["cleaned_tweet"] = df["tweets"].apply(clean_text)

    X = df["cleaned_tweet"]
    y = df["label"]

    vectorizer = TfidfVectorizer(max_features=20000, sublinear_tf=True, ngram_range=(1,2))
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(C=0.75, class_weight="balanced", max_iter=1000)
    model.fit(X_vec, y)

    return model, vectorizer

model, vectorizer = load_model()

# -----------------------------
# 3. Streamlit Frontend
# -----------------------------
st.title("🤖 Sarcasm Detector")
st.write("Enter a sentence below to check if it's **sarcastic** or **regular**.")

# Input box
user_input = st.text_area("Type your sentence here:", "")

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a sentence before detecting.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        # Show result
        if prediction == 1:
            st.success("### Prediction: Sarcastic 😏")
        else:
            st.info("### Prediction: Regular 🙂")

        # Confidence display
        st.write("### Confidence Levels:")
        st.progress(float(proba[0]))  # Regular
        st.write(f"Regular: **{proba[0]*100:.2f}%**")
        st.progress(float(proba[1]))  # Sarcasm
        st.write(f"Sarcastic: **{proba[1]*100:.2f}%**")
