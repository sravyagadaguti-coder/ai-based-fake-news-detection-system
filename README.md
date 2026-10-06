# AI-Based Fake News Detection System

Detects whether a news article is likely to be REAL or FAKE using TF-IDF + Logistic Regression, with a Streamlit web interface.

## Setup (Windows CMD)

```bat
cd AI_Fake_News_Detection
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

Copy `Fake.csv` and `True.csv` into the `data` folder (the project root also works).

## Train the model (once)

```bat
python train_model.py
```

This creates `models\fake_news_model.pkl`, `models\tfidf_vectorizer.pkl` and `confusion_matrix.png`.

## Run the app

```bat
streamlit run app.py
```

Then open http://localhost:8501 if the browser does not open automatically.

## Note

Predictions come from writing-style patterns learned from the dataset. They are not proof that an article is true or false.
"# ai-based-fake-news-detection-system" 
