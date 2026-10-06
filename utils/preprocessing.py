"""
preprocessing.py
----------------
Text cleaning used by BOTH train_model.py and app.py.
Using the same function in both places is important: the model must see
new text cleaned the same way as the training text.
"""

import re


def clean_text(text):
    """
    Clean a news article:
      1. lowercase
      2. remove HTML tags
      3. remove web links
      4. remove the word 'reuters' (see note below)
      5. keep only letters
      6. remove extra spaces
    Stop words (the, is, and ...) are removed later by TfidfVectorizer.
    """
    # Handle missing / non-text values safely
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)                   # HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)     # URLs
    text = re.sub(r"\breuters\b", " ", text)               # dataset shortcut (see note)
    text = re.sub(r"[^a-z\s]", " ", text)                  # keep letters only
    text = re.sub(r"\s+", " ", text).strip()               # extra spaces
    return text
