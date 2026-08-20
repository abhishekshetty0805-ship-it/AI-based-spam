"""
AI-Based Spam Message Detection System
========================================
A complete, runnable spam/ham text classifier using TF-IDF + Naive Bayes.

HOW TO USE
----------
1. (Recommended) Download the real "SMS Spam Collection" dataset from
   Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
   Save it as "spam.csv" in this same folder (columns: 'label', 'message').
   The script will automatically use it if found.

2. If spam.csv is NOT found, the script falls back to a small built-in
   sample dataset so you can still see everything working end-to-end.
   (Accuracy will be lower with the small sample — swap in the real
   dataset for a proper model.)

3. Run it:
       python spam_detector.py

4. After training, the script will:
   - Print accuracy / precision / recall / F1 / confusion matrix
   - Let you test the model on custom messages
   - Save the trained model + vectorizer to disk (model.pkl, vectorizer.pkl)
"""

import re
import string
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------
# 1. A small built-in stopword list (avoids needing nltk downloads)
# ---------------------------------------------------------------------
STOPWORDS = set("""
a an the is are was were be been being am i you he she it we they
this that these those to of in on at for with as by from up down
and or but if then so than too very can will just don't should now
my your his her its our their me him them us not no nor do does did
have has had having what which who whom
""".split())


# ---------------------------------------------------------------------
# 2. Load dataset (real Kaggle CSV if available, else built-in sample)
# ---------------------------------------------------------------------
def load_dataset(path="spam.csv"):
    try:
        # The Kaggle SMS Spam Collection CSV is usually latin-1 encoded
        # with columns v1 (label) and v2 (message).
        df = pd.read_csv(path, encoding="latin-1")
        df = df.iloc[:, :2]
        df.columns = ["label", "message"]
        print(f"Loaded real dataset from '{path}' ({len(df)} messages).")
    except FileNotFoundError:
        print(f"'{path}' not found — using small built-in sample dataset instead.")
        print("(Download the real dataset for much better accuracy — see docstring.)\n")
        data = {
            "label": (
                ["spam"] * 20
                + ["ham"] * 20
            ),
            "message": [
                # --- spam examples ---
                "WINNER!! You have been selected to receive a $1000 cash prize. Click here now!",
                "Congratulations! You won a free iPhone. Claim your prize before it expires.",
                "URGENT: Your account has been suspended. Verify your details immediately.",
                "Get rich quick! Work from home and earn $5000 a week guaranteed.",
                "FREE entry into our weekly draw for a chance to win cash. Text WIN to 8007.",
                "You have 1 new voicemail. Call now to claim your reward, limited time offer.",
                "Congrats! You've been chosen for a free cruise vacation. Reply YES to claim.",
                "Lowest price on Viagra online, no prescription needed, buy now!",
                "Your loan of $10000 has been approved. Click the link to receive funds today.",
                "Hot singles in your area want to meet you tonight. Click here.",
                "Claim your free gift card worth $500 now, offer ends today!",
                "URGENT: We tried to contact you, your prize of 5000 pounds is waiting.",
                "Double your income working just 2 hours a day from home!",
                "Your Netflix account is on hold, update payment info now to avoid suspension.",
                "Text STOP to unsubscribe or continue to receive free daily horoscopes.",
                "You've been selected for a limited time discount, 90% off, buy now!",
                "Free ringtones, click this link to download the hottest tracks now.",
                "Act now! Your credit score qualifies you for a guaranteed cash loan.",
                "Congratulations, you have won a lottery of $1,000,000. Send your bank details.",
                "Limited offer: buy one get one free on all items, click to shop now.",
                # --- ham examples ---
                "Hey, are we still meeting for lunch tomorrow?",
                "Can you send me the report before end of day?",
                "Happy birthday! Hope you have an amazing day.",
                "Don't forget to pick up milk on your way home.",
                "The meeting has been moved to 3pm, see you then.",
                "Thanks for helping me move last weekend, really appreciate it.",
                "What time does the movie start tonight?",
                "I'll be running about 10 minutes late, sorry about that.",
                "Can you call mom, she has been trying to reach you.",
                "Let's catch up this weekend, it's been a while.",
                "The kids have a soccer game on Saturday morning.",
                "I finished the assignment, can you review it when you get a chance?",
                "Traffic is bad, might be late to dinner.",
                "Did you watch the game last night? What a finish!",
                "Reminder: dentist appointment is at 9am tomorrow.",
                "Let me know if you need anything from the grocery store.",
                "Great job on the presentation today, the client loved it.",
                "Can we reschedule our call to Thursday instead?",
                "I left the keys under the mat, let yourself in.",
                "Looking forward to seeing you at the wedding next month!",
            ],
        }
        df = pd.DataFrame(data)

    df = df.dropna()
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham"])]
    return df


# ---------------------------------------------------------------------
# 3. Clean / preprocess text
# ---------------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"\d+", " ", text)                      # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


# ---------------------------------------------------------------------
# 4. Main pipeline
# ---------------------------------------------------------------------
def main():
    df = load_dataset()
    df["clean_message"] = df["message"].apply(clean_text)

    X = df["clean_message"]
    y = df["label"].map({"ham": 0, "spam": 1})  # 0 = ham, 1 = spam

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF feature extraction
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    print("\n===== MODEL PERFORMANCE =====")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"F1 Score : {f1_score(y_test, y_pred, zero_division=0):.3f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted) [ham, spam]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nFull report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"], zero_division=0))

    # Save model + vectorizer for reuse
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Saved trained model to 'model.pkl' and vectorizer to 'vectorizer.pkl'.")

    # ------------------------------------------------------------
    # 5. Try it on your own custom messages
    # ------------------------------------------------------------
    custom_messages = [
        "Congratulations, you have won a free vacation! Click now to claim.",
        "Hey, can you send me the notes from today's class?",
        "URGENT: verify your bank account now to avoid suspension.",
        "Are we still on for coffee tomorrow morning?",
    ]

    print("\n===== TESTING ON CUSTOM MESSAGES =====")
    for msg in custom_messages:
        cleaned = clean_text(msg)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        label = "SPAM" if pred == 1 else "HAM"
        print(f"[{label}]  {msg}")


def predict_message(message, model_path="model.pkl", vectorizer_path="vectorizer.pkl"):
    """Load a saved model and classify a single new message."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return "spam" if pred == 1 else "ham"


if __name__ == "__main__":
    main()
