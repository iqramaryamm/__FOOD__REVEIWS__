# data_analyser.py
from __future__ import annotations
from typing import Optional, Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

sns.set(style="whitegrid")

class DataAnalyser:
    def __init__(self, text_col: str = "Text", label_col: str = "Sentiment", score_col: str = "Score"):
        self.text_col = text_col
        self.label_col = label_col
        self.score_col = score_col

    def basic_info(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        return df.describe(include="all").transpose(), df.dtypes

    def class_balance(self, df: pd.DataFrame, normalize: bool = True) -> pd.Series:
        return df[self.label_col].value_counts(normalize=normalize)

    def score_distribution(self, df: pd.DataFrame) -> None:
        if self.score_col not in df.columns:
            print(f"'{self.score_col}' not in DataFrame; skipping score distribution.")
            return
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=self.score_col)
        plt.title("Score Distribution")
        plt.tight_layout()
        plt.show()

    def wordcloud(self, text: str, background_color: str = "white", title: Optional[str] = None) -> None:
        if not isinstance(text, str) or not text.strip():
            print("No text for wordcloud.")
            return
        wc = WordCloud(background_color=background_color, width=1000, height=600).generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.show()

    def wordcloud_overall(self, df: pd.DataFrame, source_col: str = "Summary") -> None:
        if source_col not in df.columns:
            print(f"'{source_col}' not in DataFrame; skipping overall wordcloud.")
            return
        self.wordcloud(df[source_col].astype(str).str.cat(sep=" "), title="Overall WordCloud")

    def wordcloud_by_label(self, df: pd.DataFrame, source_col: str = "Summary") -> None:
        if source_col not in df.columns:
            print(f"'{source_col}' not in DataFrame; skipping label wordclouds.")
            return
        for label, bg in [("Negative", "white"), ("Positive", "black")]:
            subset = df[df[self.label_col] == label]
            text = subset[source_col].astype(str).str.cat(sep=" ")
            self.wordcloud(text, background_color=bg, title=f"{label} Reviews WordCloud")
