# model_evaluator.py
from __future__ import annotations
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

sns.set(style="whitegrid")

def evaluate_models(models: Dict[str, object], X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        })

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Positive"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {name}")
        plt.grid(False)
        plt.tight_layout()
        plt.show()

        print(f"\n=== Classification Report: {name} ===")
        print(classification_report(y_test, y_pred, digits=3))

    results_df = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

    # Accuracy bar
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=results_df, x="Model", y="Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    return results_df
