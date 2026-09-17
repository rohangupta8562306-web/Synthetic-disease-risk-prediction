"""
data_preprocessing.py
----------------------
Handles all data loading, cleaning, and preprocessing logic for the
Synthetic Disease Risk Prediction project.

This module is intentionally kept independent of any specific ML model so
that it can be reused by both the training script (train_model.py) and the
Streamlit application (app.py) for a consistent, leakage-free pipeline.
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
# These lists describe the dataset's schema. Keeping them centralized here
# means the rest of the project never has to hard-code column names.

ID_COLUMN = "Patient_ID"
TARGET_COLUMN = "Disease_Risk"

NUMERIC_FEATURES = [
    "Age",
    "BMI",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "Cholesterol_Level",
    "Glucose_Level",
    "Genetic_Risk_Score",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Smoking_Status",
    "Alcohol_Consumption",
    "Physical_Activity_Level",
    "Family_History",
    "Previous_Diagnosis",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Categorical columns where a missing value has a real-world meaning
# (e.g. no alcohol consumption reported, no previous diagnosis) rather than
# being a genuinely unknown/erroneous value. We fill these with an explicit
# "None" category instead of dropping records or using the column mode,
# so we don't discard useful rows or invent information that isn't there.
MEANINGFUL_MISSING_CATEGORICAL = {
    "Alcohol_Consumption": "None",
    "Previous_Diagnosis": "None",
}


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the raw dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the dataset CSV file.

    Returns
    -------
    pd.DataFrame
        The raw dataset, completely unmodified.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset without mutating the original file on disk.

    Steps performed:
    1. Work on a copy (never touch the caller's / original DataFrame in place).
    2. Remove exact duplicate rows, if any.
    3. Fill "meaningful missing" categorical values (e.g. Alcohol_Consumption,
       Previous_Diagnosis) with an explicit "None" category rather than
       dropping rows.
    4. Clip/validate obviously impossible numeric values (defensive only —
       this dataset did not contain any in practice, but guards against
       future/real-world data quality issues).
    5. Ensure the target column contains only the two expected classes.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset as loaded by `load_dataset`.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset, ready for feature/target separation.
    """
    data = df.copy()

    # 1. Drop exact duplicate rows (keep the first occurrence)
    before_rows = len(data)
    data = data.drop_duplicates()
    duplicates_removed = before_rows - len(data)

    # 2. Fill meaningful missing categorical values with an explicit label
    for column, fill_value in MEANINGFUL_MISSING_CATEGORICAL.items():
        if column in data.columns:
            data[column] = data[column].fillna(fill_value)

    # 3. Defensive numeric sanity checks (do not drop rows; only guard against
    #    physically impossible values that would indicate data entry errors).
    numeric_bounds = {
        "Age": (0, 120),
        "BMI": (5, 80),
        "Blood_Pressure_Systolic": (50, 260),
        "Blood_Pressure_Diastolic": (30, 180),
        "Cholesterol_Level": (50, 500),
        "Glucose_Level": (30, 500),
        "Genetic_Risk_Score": (0, 1),
    }
    for column, (low, high) in numeric_bounds.items():
        if column in data.columns:
            data[column] = data[column].clip(lower=low, upper=high)

    # 4. Keep only rows with a valid target label
    if TARGET_COLUMN in data.columns:
        data = data[data[TARGET_COLUMN].isin(["Yes", "No"])]

    data = data.reset_index(drop=True)

    print(f"[data_preprocessing] Duplicates removed: {duplicates_removed}")
    print(f"[data_preprocessing] Final cleaned dataset shape: {data.shape}")

    return data


def split_features_target(df: pd.DataFrame):
    """
    Separate the cleaned dataset into model-ready features (X) and target (y).

    The Patient_ID column is dropped because it is a unique identifier with
    no predictive value and would otherwise leak information / add noise.
    The target column is label-encoded: "Yes" -> 1 (at risk), "No" -> 0.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.

    Returns
    -------
    (pd.DataFrame, pd.Series)
        X (features only, no ID, no target) and y (binary target).
    """
    feature_columns = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_columns].copy()

    y = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)

    return X, y


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that:
      - Imputes and scales numeric features.
      - Imputes and one-hot encodes categorical features.

    Wrapping imputation + encoding inside sklearn transformers (rather than
    doing it manually with pandas before the train/test split) means the
    transformer can be `fit` only on the training data and safely applied
    to the test data / new patient inputs afterward, preventing data
    leakage.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer):
    """
    Retrieve human-readable feature names after preprocessing/encoding.
    Useful for feature-importance style analysis and debugging.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return None


def load_and_prepare(csv_path: str):
    """
    Convenience wrapper: load -> clean -> split features/target.

    Returns
    -------
    (pd.DataFrame, pd.Series, pd.DataFrame)
        X, y, and the full cleaned dataframe (useful for EDA / dashboards).
    """
    raw_df = load_dataset(csv_path)
    cleaned_df = clean_dataset(raw_df)
    X, y = split_features_target(cleaned_df)
    return X, y, cleaned_df


if __name__ == "__main__":
    # Quick manual test when running this file directly
    default_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "Synthetic_disease_risk_dataset.csv"
    )
    X, y, full_df = load_and_prepare(default_path)
    print("Features shape:", X.shape)
    print("Target distribution:\n", y.value_counts())
