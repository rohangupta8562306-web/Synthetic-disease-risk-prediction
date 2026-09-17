"""
evaluate_model.py
------------------
Utility functions to load and present the evaluation metrics produced by
train_model.py, and to re-run evaluation on demand (e.g. for reporting or
a sanity re-check) without needing to retrain the model.
"""

import os
import json

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

try:
    from src.data_preprocessing import load_and_prepare
except ImportError:
    from data_preprocessing import load_and_prepare

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Synthetic_disease_risk_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_risk_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessing_pipeline.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")

RANDOM_STATE = 42


def load_saved_metrics() -> dict:
    """
    Load the pre-computed metrics summary produced during training.
    This is what the Streamlit "Model Performance" page uses, so the app
    never needs to retrain the model just to display numbers.
    """
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            "model_metrics.json not found. Please run `python src/train_model.py` first."
        )
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def recompute_test_metrics():
    """
    Re-run the exact same train/test split (fixed random_state) used during
    training and recompute metrics for the currently saved model. Useful as
    an independent sanity check that the saved model/metrics are consistent.
    """
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    X, y, _ = load_and_prepare(DATA_PATH)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    X_test_transformed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_transformed)
    y_proba = model.predict_proba(X_test_transformed)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


if __name__ == "__main__":
    summary = load_saved_metrics()
    print(f"Best model: {summary['best_model']}")
    print(json.dumps(summary["all_model_results"][summary["best_model"]], indent=2)[:500])

    print("\nRe-computed metrics (independent check):")
    recomputed = recompute_test_metrics()
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        print(f"  {key}: {recomputed[key]:.4f}")
