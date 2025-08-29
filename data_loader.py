# data_loader.py
from __future__ import annotations
import re
from typing import Tuple, Optional, Literal

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from textblob import Word

# Ensure required NLTK assets exist (safe to call repeatedly)
def _ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")

class DataLoader:
    """
    Loads and preprocesses Reviews.csv for sentiment tasks.

    Label options:
      - label_source='score': derive binary labels from Score (1/2=Negative, 4/5=Positive), drop Score==3
      - label_source='vader': create labels using VADER compound >= 0 => Positive else Negative
    """

    def __init__(
        self,
        text_col: str = "Text",
        score_col: str = "Score",
        label_source: Literal["score", "vader"] = "score",
        lowercase: bool = True,
        remove_punct: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        sample_size: Optional[int] = None,  # sample for fast experiments
        random_state: int = 42,
    ):
        self.text_col = text_col
        self.score_col = score_col
        self.label_source = label_source
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.sample_size = sample_size
        self.random_state = random_state

        _ensure_nltk()
        self._stop = set(stopwords.words("english"))

        # Lazy import for vader only when needed
        self._vader = None

    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if self.sample_size:
            df = df.sample(self.sample_size, random_state=self.random_state).reset_index(drop=True)
        # Keep only relevant columns if they exist
        keep = [c for c in [self.text_col, self.score_col, "Summary"] if c in df.columns]
        return df[keep].dropna(subset=[self.text_col]).copy()

    # --- Text cleaning steps ---
    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        if self.lowercase:
            text = " ".join(text.lower().split())

        if self.remove_punct:
            text = re.sub(r"[^\w\s]", "", text)

        if self.remove_stopwords:
            text = " ".join(w for w in text.split() if w not in self._stop)

        if self.lemmatize:
            text = " ".join(Word(w).lemmatize() for w in text.split())

        return text

    def preprocess_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.text_col] = df[self.text_col].astype(str).apply(self._clean_text)
        return df

    # --- Label creation ---
    def _labels_from_score(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.score_col not in df.columns:
            raise KeyError(f"Score column '{self.score_col}' not found for label_source='score'.")
        df = df.copy()
        # Drop neutral (3) and map to binary
        df = df[df[self.score_col].isin([1, 2, 4, 5])]
        df["Sentiment"] = np.where(df[self.score_col].isin([1, 2]), "Negative", "Positive")
        return df

    def _labels_from_vader(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._vader is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()

        df = df.copy()
        scores = df[self.text_col].astype(str).apply(lambda t: self._vader.polarity_scores(t)["compound"])
        df["compound"] = scores
        df["Sentiment"] = np.where(df["compound"] >= 0.0, "Positive", "Negative")
        return df

    def build_dataset(self, path: str) -> pd.DataFrame:
        df = self.load(path)
        df = self.preprocess_text(df)

        if self.label_source == "score":
            df = self._labels_from_score(df)
        elif self.label_source == "vader":
            df = self._labels_from_vader(df)
        else:
            raise ValueError("label_source must be 'score' or 'vader'.")

        df = df.dropna(subset=[self.text_col, "Sentiment"])
        df = df.reset_index(drop=True)
        return df
