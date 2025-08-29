# model_trainer.py
from __future__ import annotations
from typing import Dict, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def split_data(
    df: pd.DataFrame,
    text_col: str = "Text",
    label_col: str = "Sentiment",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    X = df[text_col]
    y = df[label_col]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

def model_zoo(random_state: int = 42) -> Dict[str, Pipeline]:
    return {
        "LogReg": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, random_state=random_state))
        ]),
        "LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
            ("clf", LinearSVC(random_state=random_state))
        ]),
       
        
    }

def default_param_grids() -> Dict[str, dict]:
    return {
        "LogReg": {
            "tfidf__ngram_range": [(1,1), (1,2)],
            "tfidf__min_df": [1, 2, 5],
            "clf__C": [0.5, 1.0, 2.0],
        },
        "LinearSVC": {
            "tfidf__ngram_range": [(1,1), (1,2)],
            "tfidf__min_df": [1, 2, 5],
            "clf__C": [0.5, 1.0, 2.0],
        },
       
    }

def train_models(
    X_train: pd.Series,
    y_train: pd.Series,
    use_grid_search: bool = False,
    cv: int = 3,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Dict[str, Pipeline]:
    models = model_zoo(random_state=random_state)
    fitted = {}

    if use_grid_search:
        grids = default_param_grids()

    for name, pipe in models.items():
        if use_grid_search and name in grids:
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=grids[name],
                cv=cv,
                n_jobs=n_jobs,
                scoring="f1_macro",
                refit=True,
            )
            gs.fit(X_train, y_train)
            fitted[name] = gs.best_estimator_
            print(f"[{name}] best params: {gs.best_params_}")
        else:
            pipe.fit(X_train, y_train)
            fitted[name] = pipe

    return fitted
