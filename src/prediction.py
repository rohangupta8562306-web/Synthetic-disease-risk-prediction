"""
prediction.py
--------------
Loads the trained model and preprocessing pipeline from disk and exposes a
simple function to predict disease risk for a single new patient, without
ever needing to retrain the model.

This module is used directly by the Streamlit application (app.py).
"""

import os
import sys
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
for p in [BASE_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.data_preprocessing import ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
except ImportError:
    from data_preprocessing import ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES

# Ensure both module namespaces exist for pickle deserialization compatibility
try:
    import data_preprocessing as _dp
    sys.modules.setdefault("src.data_preprocessing", _dp)
    sys.modules.setdefault("data_preprocessing", _dp)
except Exception:
    pass

MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_risk_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessing_pipeline.pkl")


class ModelNotFoundError(Exception):
    """Raised when the trained model/preprocessor files cannot be found."""
    pass


def load_artifacts():
    """
    Load the trained model and preprocessing pipeline from disk.
    If artifacts are missing or fail to unpickle due to version/module
    discrepancies in the hosting environment, automatically trains them
    on the fly.
    """
    need_train = not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH)
    if not need_train:
        try:
            model = joblib.load(MODEL_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            return model, preprocessor
        except Exception as err:
            print(f"[prediction] Warning: Artifact loading failed ({err}). Retraining model on the fly...")

    try:
        try:
            from src.train_model import train_and_select_best_model
        except ImportError:
            from train_model import train_and_select_best_model
        train_and_select_best_model()
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except Exception as e:
        raise ModelNotFoundError(f"Failed to load or generate model artifacts: {e}")


def validate_patient_input(patient_data: dict):
    """
    Validate a single patient's raw input dictionary before prediction.

    Returns
    -------
    list[str]
        A list of human-readable validation error messages. Empty list
        means the input is valid.
    """
    errors = []

    def _check_range(field, low, high):
        value = patient_data.get(field)
        if value is None:
            errors.append(f"{field} is required.")
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number.")
            return
        if not (low <= value <= high):
            errors.append(f"{field} should be between {low} and {high}.")

    _check_range("Age", 18, 120)
    _check_range("BMI", 10, 70)
    _check_range("Blood_Pressure_Systolic", 70, 250)
    _check_range("Blood_Pressure_Diastolic", 40, 150)
    _check_range("Cholesterol_Level", 80, 400)
    _check_range("Glucose_Level", 50, 400)
    _check_range("Genetic_Risk_Score", 0.0, 1.0)

    if patient_data.get("Blood_Pressure_Diastolic") and patient_data.get("Blood_Pressure_Systolic"):
        try:
            if float(patient_data["Blood_Pressure_Diastolic"]) >= float(patient_data["Blood_Pressure_Systolic"]):
                errors.append("Diastolic blood pressure should be lower than systolic blood pressure.")
        except (TypeError, ValueError):
            pass

    for field in CATEGORICAL_FEATURES:
        if not patient_data.get(field):
            errors.append(f"{field} is required.")

    return errors


def predict_risk(patient_data: dict, model=None, preprocessor=None):
    """
    Predict disease risk for a single patient.

    Parameters
    ----------
    patient_data : dict
        Dictionary with keys matching ALL_FEATURES (Age, Gender, BMI, ...).
    model, preprocessor : optional
        Pre-loaded artifacts. If not provided, they are loaded from disk
        (useful for the Streamlit app to cache them once with st.cache_resource
        and pass them in on every call, avoiding repeated disk reads).

    Returns
    -------
    dict
        {
            "prediction_label": "High Risk" | "Low Risk",
            "prediction_value": 1 | 0,
            "probability_high_risk": float (0-1),
            "probability_low_risk": float (0-1),
        }
    """
    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts()

    # Build a single-row DataFrame in the exact column order the
    # preprocessor was fitted on.
    row = {feature: patient_data.get(feature) for feature in ALL_FEATURES}
    input_df = pd.DataFrame([row])

    transformed = preprocessor.transform(input_df)

    prediction = int(model.predict(transformed)[0])
    probabilities = model.predict_proba(transformed)[0]

    probability_low_risk = float(probabilities[0])
    probability_high_risk = float(probabilities[1])

    return {
        "prediction_label": "High Risk" if prediction == 1 else "Low Risk",
        "prediction_value": prediction,
        "probability_high_risk": probability_high_risk,
        "probability_low_risk": probability_low_risk,
    }


if __name__ == "__main__":
    # Simple manual smoke test
    sample_patient = {
        "Age": 55,
        "Gender": "Male",
        "BMI": 29.5,
        "Smoking_Status": "Current",
        "Alcohol_Consumption": "Moderate",
        "Physical_Activity_Level": "Low",
        "Blood_Pressure_Systolic": 145,
        "Blood_Pressure_Diastolic": 92,
        "Cholesterol_Level": 230,
        "Glucose_Level": 160,
        "Family_History": "Yes",
        "Genetic_Risk_Score": 0.68,
        "Previous_Diagnosis": "Pre-disease",
    }
    errors = validate_patient_input(sample_patient)
    print("Validation errors:", errors)
    result = predict_risk(sample_patient)
    print("Prediction result:", result)
