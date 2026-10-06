
"""
AI-Based Fake News Detection System
Streamlit Application
"""

from pathlib import Path
import re

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean news text before prediction.
    """

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model_and_vectorizer():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


# ============================================================
# CHECK MODEL FILES
# ============================================================

def model_files_exist():

    return MODEL_PATH.exists() and VECTORIZER_PATH.exists()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 Project Information")

    st.write("**Project:** AI-Based Fake News Detection System")
    st.write("**Model:** Logistic Regression")
    st.write("**Feature Extraction:** TF-IDF")
    st.write("**Dataset:** Fake.csv + True.csv")
    st.write("**Type:** College Mini Project")

    st.divider()

    st.info(
        "The prediction is based on patterns learned from the "
        "training dataset. It should not be treated as absolute proof."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("📰 AI-Based Fake News Detection System")

st.subheader(
    "Detect whether a news article is likely to be REAL or FAKE using Machine Learning."
)

st.divider()


# ============================================================
# CHECK MODEL
# ============================================================

if not model_files_exist():

    st.error("❌ Model files not found.")

    st.warning(
        "You must train the model before running the application."
    )

    st.code(
        "python train_model.py",
        language="bash"
    )

    st.info(
        "Run the above command in your project folder and then refresh this page."
    )

    st.stop()


# ============================================================
# INPUT
# ============================================================

news_text = st.text_area(
    "📝 Enter or paste news article here",
    height=250,
    placeholder="Paste the news title and article content here..."
)


# ============================================================
# DETECT BUTTON
# ============================================================

detect_clicked = st.button(
    "🔍 Detect News",
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if detect_clicked:

    # --------------------------------------------------------
    # Check empty input
    # --------------------------------------------------------

    if not news_text.strip():

        st.warning(
            "⚠️ Please enter some news text before clicking Detect."
        )

        st.stop()


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        model, vectorizer = load_model_and_vectorizer()

    except Exception as error:

        st.error(
            f"❌ Could not load model files: {error}"
        )

        st.info(
            "Please train the model again using:"
        )

        st.code(
            "python train_model.py",
            language="bash"
        )

        st.stop()


    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    cleaned_text = clean_text(news_text)


    if not cleaned_text:

        st.warning(
            "⚠️ No usable text was found. Please enter a proper news article."
        )

        st.stop()


    # --------------------------------------------------------
    # Short text warning
    # --------------------------------------------------------

    if len(cleaned_text.split()) < 15:

        st.warning(
            "⚠️ The article is very short. "
            "Please enter a longer article for a more reliable prediction."
        )


    # --------------------------------------------------------
    # Convert text to TF-IDF
    # --------------------------------------------------------

    try:

        features = vectorizer.transform([cleaned_text])

    except Exception as error:

        st.error(
            f"❌ Text processing failed: {error}"
        )

        st.stop()


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        prediction = int(model.predict(features)[0])

        probabilities = model.predict_proba(features)[0]

    except Exception as error:

        st.error(
            f"❌ Prediction failed: {error}"
        )

        st.stop()


    # ========================================================
    # PROBABILITIES
    # ========================================================

    # Model labels:
    # 0 = FAKE
    # 1 = REAL

    fake_prob = float(probabilities[0]) * 100
    real_prob = float(probabilities[1]) * 100

    confidence = max(fake_prob, real_prob)


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader("📊 Prediction Result")


    if prediction == 0:

        st.error(
            "⚠️ This news is predicted as FAKE"
        )

    else:

        st.success(
            "✅ This news is predicted as REAL"
        )


    st.write(
        f"### Confidence: {confidence:.2f}%"
    )

    st.progress(
        int(confidence)
    )


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    col2.metric(
        "REAL Probability",
        f"{real_prob:.2f}%"
    )

    col3.metric(
        "FAKE Probability",
        f"{fake_prob:.2f}%"
    )


    # ========================================================
    # PROBABILITY CHART
    # ========================================================

    st.subheader("📈 Probability Chart")

    chart_data = pd.DataFrame(
        {
            "Probability (%)": [
                real_prob,
                fake_prob
            ]
        },
        index=[
            "REAL",
            "FAKE"
        ]
    )

    st.bar_chart(chart_data)


    # ========================================================
    # INFORMATION
    # ========================================================

    st.info(
        "This prediction is generated by a machine-learning model "
        "using TF-IDF features and Logistic Regression. "
        "Always verify important news using reliable sources."
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

if CONFUSION_MATRIX_PATH.exists():

    with st.expander(
        "📊 View Model Performance"
    ):

        st.image(
            str(CONFUSION_MATRIX_PATH),
            caption="Confusion Matrix"
        )

