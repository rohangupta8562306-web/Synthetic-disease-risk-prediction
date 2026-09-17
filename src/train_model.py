"""
train_model.py
---------------
Trains and compares several classification algorithms on the Synthetic
Disease Risk dataset, selects the best-performing model based on
evaluation metrics (not arbitrarily), and persists the fitted
preprocessing pipeline and model to disk with joblib.

Run directly with:
    python src/train_model.py
"""

import os
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# Allow running this script both as `python src/train_model.py` (script)
# and as part of the package (`from src.train_model import ...`).
try:
    from src.data_preprocessing import (
        load_and_prepare,
        build_preprocessing_pipeline,
        get_feature_names,
    )
except ImportError:
    from data_preprocessing import (
        load_and_prepare,
        build_preprocessing_pipeline,
        get_feature_names,
    )

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reproducibility & paths
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Synthetic_disease_risk_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "disease_risk_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")


def get_candidate_models():
    """
    Define the candidate classification algorithms to compare.

    class_weight="balanced" (where supported) is used because the target
    variable is imbalanced (~14.5% positive class), so the models don't
    simply learn to predict the majority class.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=RANDOM_STATE
        ),
    }


def evaluate_model(model, X_test_transformed, y_test):
    """
    Compute the full evaluation metric suite for a fitted model.

    Returns a dictionary of metrics plus the raw data needed to redraw the
    confusion matrix and ROC curve later in the Streamlit app (so the app
    never needs to retrain or re-run predictions itself).
    """
    y_pred = model.predict(X_test_transformed)

    # Not every classifier trivially supports predict_proba, but all four
    # candidates used here do.
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_transformed)[:, 1]
    else:
        y_proba = model.decision_function(X_test_transformed)

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
    }
    return metrics


def select_best_model(results: dict) -> str:
    """
    Select the best model based on ROC-AUC first (a robust metric for
    imbalanced classification problems), using F1-score as a tie-breaker.
    This avoids picking a model arbitrarily.
    """
    ranked = sorted(
        results.items(),
        key=lambda item: (item[1]["roc_auc"], item[1]["f1_score"]),
        reverse=True,
    )
    return ranked[0][0]


def train_and_select_best_model():
    """
    Full training pipeline:
    1. Load and clean data.
    2. Train/test split (stratified, reproducible).
    3. Fit preprocessing on training data ONLY (no leakage).
    4. Train each candidate model.
    5. Evaluate every model on the held-out test set.
    6. Select the best model based on evaluation metrics.
    7. Persist the fitted preprocessor and the best model to disk.
    8. Persist all models' metrics to a JSON file for the dashboard.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading and cleaning dataset...")
    X, y, cleaned_df = load_and_prepare(DATA_PATH)

    print("Splitting into train/test sets (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Fitting preprocessing pipeline on training data only...")
    preprocessor = build_preprocessing_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    candidates = get_candidate_models()
    results = {}
    fitted_models = {}

    for name, model in candidates.items():
        print(f"Training: {name} ...")
        start = time.time()
        model.fit(X_train_transformed, y_train)
        elapsed = time.time() - start

        metrics = evaluate_model(model, X_test_transformed, y_test)
        metrics["training_time_seconds"] = round(elapsed, 3)
        results[name] = metrics
        fitted_models[name] = model

        print(
            f"  -> Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )

    best_model_name = select_best_model(results)
    best_model = fitted_models[best_model_name]

    print(f"\nSelected best model based on ROC-AUC (tie-break F1): {best_model_name}")

    # Persist artifacts
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    summary = {
        "best_model": best_model_name,
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "feature_columns": list(X.columns),
        "all_model_results": results,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

    return summary


if __name__ == "__main__":
    train_and_select_best_model()
