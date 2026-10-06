"""
train_model.py
--------------
Trains the Fake News Detection model and saves it to the 'models' folder.

Run this ONCE (and again only if you want to retrain):
    python train_model.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # lets matplotlib save images without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils.preprocessing import clean_text

# ----------------------------------------------------------------------
# 1. Paths (these work on Windows because we use pathlib)
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"

RANDOM_STATE = 42  # makes results reproducible
TEST_SIZE = 0.20   # 20% of the data is used for testing


def find_csv(filename):
    """Look for the CSV in data/ first, then in the project root."""
    for folder in (DATA_DIR, BASE_DIR):
        candidate = folder / filename
        if candidate.exists():
            return candidate

    print(f"[ERROR] Could not find {filename}.")
    print(f"        Put it in: {DATA_DIR}")
    print(f"        (or directly in: {BASE_DIR})")
    sys.exit(1)


def read_csv_safely(path):
    """Read a CSV file. Try UTF-8 first; if that fails, try latin-1."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        print(f"[INFO] {path.name} is not UTF-8. Retrying with latin-1 encoding...")
        return pd.read_csv(path, encoding="latin-1")


def check_columns(df, filename):
    """Make sure the columns we need exist."""
    missing = [col for col in ("title", "text") if col not in df.columns]
    if missing:
        print(f"[ERROR] {filename} is missing column(s): {missing}")
        print(f"        Columns found: {list(df.columns)}")
        sys.exit(1)


def main():
    print("=" * 60)
    print(" AI-Based Fake News Detection - Model Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 2. Load the two CSV files
    # ------------------------------------------------------------------
    fake_path = find_csv("Fake.csv")
    true_path = find_csv("True.csv")
    print(f"[OK] Found Fake.csv at: {fake_path}")
    print(f"[OK] Found True.csv at: {true_path}")

    fake_df = read_csv_safely(fake_path)
    true_df = read_csv_safely(true_path)
    check_columns(fake_df, "Fake.csv")
    check_columns(true_df, "True.csv")
    print(f"[OK] Fake articles loaded: {len(fake_df)}")
    print(f"[OK] True articles loaded: {len(true_df)}")

    # ------------------------------------------------------------------
    # 3. Add labels: FAKE = 0, REAL = 1
    # ------------------------------------------------------------------
    fake_df["label"] = 0
    true_df["label"] = 1

    # ------------------------------------------------------------------
    # 4. Combine the two datasets into one
    # ------------------------------------------------------------------
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # ------------------------------------------------------------------
    # 5. Handle missing values and combine title + text
    # ------------------------------------------------------------------
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["content"] = df["title"] + " " + df["text"]

    # Remove rows that have no content at all
    df = df[df["content"].str.strip() != ""]

    # ------------------------------------------------------------------
    # 6. Remove duplicates
    # ------------------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset="content")
    print(f"[OK] Duplicates removed: {before - len(df)}")

    # ------------------------------------------------------------------
    # 7. Shuffle the data (so FAKE and REAL rows are mixed)
    # ------------------------------------------------------------------
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 8. Clean the text (this can take a minute on the full dataset)
    # ------------------------------------------------------------------
    print("[INFO] Cleaning text, please wait...")
    df["clean_content"] = df["content"].apply(clean_text)
    df = df[df["clean_content"].str.strip() != ""]
    print(f"[OK] Total articles used: {len(df)}")
    print(f"     FAKE: {(df['label'] == 0).sum()}  |  REAL: {(df['label'] == 1).sum()}")

    # ------------------------------------------------------------------
    # 9. Split into training and testing data
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_content"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],  # keeps the FAKE/REAL ratio equal in both sets
    )
    print(f"[OK] Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # ------------------------------------------------------------------
    # 10. TF-IDF vectorizer (learn from TRAINING data only)
    # ------------------------------------------------------------------
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"[OK] TF-IDF created with {len(vectorizer.vocabulary_)} features")

    # ------------------------------------------------------------------
    # 11. Train Logistic Regression
    # ------------------------------------------------------------------
    print("[INFO] Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    print("[OK] Model training finished")

    # ------------------------------------------------------------------
    # 12. Evaluate: accuracy, classification report, confusion matrix
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + "-" * 60)
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")
    print("-" * 60)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows = actual, columns = predicted):")
    print(cm)

    # ------------------------------------------------------------------
    # 13. Save the confusion matrix as an image
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["FAKE", "REAL"],
        yticklabels=["FAKE", "REAL"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(f"Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    print(f"\n[OK] Confusion matrix image saved: {CONFUSION_MATRIX_PATH}")

    # ------------------------------------------------------------------
    # 14. Save the model and the vectorizer
    # ------------------------------------------------------------------
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[OK] Model saved:      {MODEL_PATH}")
    print(f"[OK] Vectorizer saved: {VECTORIZER_PATH}")

    print("\n" + "=" * 60)
    print(" SUCCESS! Now run:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
