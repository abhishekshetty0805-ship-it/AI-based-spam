"""
AI-Based Spam Message Detection System — Web App
==================================================
A simple Streamlit web interface for the trained spam/ham model.

HOW TO RUN
----------
1. Make sure you've already run spam_detector.py at least once, so that
   model.pkl and vectorizer.pkl exist in this same folder.
2. Install streamlit if you don't have it:
       pip install streamlit
3. Run:
       streamlit run app.py
4. Your browser will open automatically at http://localhost:8501
"""

import re
import string
import joblib
import streamlit as st

# ---------------------------------------------------------------------
# Must match the cleaning logic used in spam_detector.py exactly,
# so predictions stay consistent with how the model was trained.
# ---------------------------------------------------------------------
STOPWORDS = set("""
a an the is are was were be been being am i you he she it we they
this that these those to of in on at for with as by from up down
and or but if then so than too very can will just don't should now
my your his her its our their me him them us not no nor do does did
have has had having what which who whom
""".split())


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="Spam Message Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ AI Spam Message Detector")
st.write("Paste any SMS or email text below and the model will predict whether it's **Spam** or **Ham** (not spam).")

try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error(
        "Couldn't find 'model.pkl' / 'vectorizer.pkl' in this folder.\n\n"
        "Run `python spam_detector.py` first to train and save the model, "
        "then come back and run this app again."
    )
    st.stop()

message = st.text_area(
    "Message to check:",
    height=150,
    placeholder="e.g. Congratulations! You've won a free prize, click here to claim now!",
)

col1, col2 = st.columns([1, 3])
with col1:
    check_clicked = st.button("Check Message", type="primary")

if check_clicked:
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(message)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        spam_prob = proba[1] * 100
        ham_prob = proba[0] * 100

        if pred == 1:
            st.error(f"🚨 This looks like **SPAM** ({spam_prob:.1f}% confidence)")
        else:
            st.success(f"✅ This looks like **HAM** (not spam) ({ham_prob:.1f}% confidence)")

        st.progress(spam_prob / 100)
        st.caption(f"Spam probability: {spam_prob:.1f}%  |  Ham probability: {ham_prob:.1f}%")

st.divider()
st.caption("Model: TF-IDF + Naive Bayes, trained on the SMS Spam Collection dataset.")