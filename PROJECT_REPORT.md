# PROJECT REPORT

## Synthetic Disease Risk Prediction System

---

## Chapter 1 — Introduction

### 1.1 Background

Chronic and lifestyle-related diseases are influenced by a combination of
demographic factors (age, gender), lifestyle choices (smoking, alcohol use,
physical activity), clinical measurements (blood pressure, cholesterol,
glucose), and hereditary factors (family history, genetic predisposition).
Machine Learning offers a way to model the complex, non-linear relationships
between such factors and disease outcomes, potentially supporting early
awareness and preventive action.

### 1.2 Problem Statement

Given a set of patient health attributes, the goal is to build a
classification model that predicts whether a patient is at risk of disease
(`Disease_Risk` = Yes/No), and to present this capability through an
accessible, interactive web application.

### 1.3 Motivation

Health-risk prediction is one of the most common and instructive real-world
applications of machine learning, combining data cleaning, exploratory
analysis, classification modeling, and interactive deployment into a single
project — making it well suited to demonstrate a full end-to-end ML
workflow.

### 1.4 Objectives

- Load, clean, and explore a synthetic disease-risk dataset.
- Design a leakage-free preprocessing pipeline.
- Train and objectively compare multiple classification algorithms.
- Select the best-performing model using proper evaluation metrics.
- Persist the trained model for reuse without retraining.
- Build an interactive, multi-page Streamlit web application for predictions
  and data exploration.

### 1.5 Scope

The scope of this project is strictly educational. It is trained on a
**synthetic** dataset of ~4,000 records and is **not** validated for, or
intended for, real clinical or diagnostic use.

---

## Chapter 2 — Dataset

### 2.1 Dataset Description

The dataset (`data/Synthetic_disease_risk_dataset.csv`) contains synthetic
patient records with demographic, lifestyle, clinical, and hereditary
attributes, along with a binary disease-risk label.

### 2.2 Number of Records

4,000 rows (patients).

### 2.3 Number of Features

15 columns total: 1 identifier (`Patient_ID`), 13 predictive features, and 1
target column (`Disease_Risk`).

### 2.4 Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| Patient_ID | Identifier | Unique patient ID (excluded from modeling) |
| Age | Numeric | Age in years (18–89 in this dataset) |
| Gender | Categorical | Female / Male / Other |
| BMI | Numeric | Body Mass Index |
| Smoking_Status | Categorical | Never / Former / Current |
| Alcohol_Consumption | Categorical | None / Moderate / High |
| Physical_Activity_Level | Categorical | Low / Moderate / High |
| Blood_Pressure_Systolic | Numeric | Systolic BP (mmHg) |
| Blood_Pressure_Diastolic | Numeric | Diastolic BP (mmHg) |
| Cholesterol_Level | Numeric | Cholesterol (mg/dL) |
| Glucose_Level | Numeric | Glucose (mg/dL) |
| Family_History | Categorical | Yes / No |
| Genetic_Risk_Score | Numeric | Continuous score, 0.2–0.93 in this dataset |
| Previous_Diagnosis | Categorical | None / Pre-disease / Diagnosed |
| Disease_Risk | Categorical (Target) | Yes / No |

### 2.5 Target Variable

`Disease_Risk` is binary: **3,419 "No"** records and **581 "Yes"** records
(≈14.5% positive class) — a moderately imbalanced classification problem.

### 2.6 Missing Values

An inspection of the raw dataset found:

- `Alcohol_Consumption`: 1,639 missing values
- `Previous_Diagnosis`: 2,027 missing values
- No missing values in any other column
- **0 duplicate rows**

These two columns' missing values were determined to represent a meaningful
real-world state (i.e. "no alcohol consumption reported" / "no previous
diagnosis") rather than random or erroneous missingness, so they were filled
with an explicit `"None"` category rather than being dropped or imputed with
the column mode — preserving all 4,000 records.

### 2.7 Data Preprocessing

1. Duplicate row removal (0 rows removed — none were present).
2. Explicit `"None"` category fill for `Alcohol_Consumption` and
   `Previous_Diagnosis`.
3. Defensive numeric range clipping for physically implausible values
   (none triggered in practice on this dataset, but included for robustness).
4. Removal of the `Patient_ID` column prior to modeling (identifier column,
   no predictive value, risk of leakage/overfitting if included).
5. Label encoding of the target (`Yes` → 1, `No` → 0).
6. Numeric features: median imputation + standardization (`StandardScaler`).
7. Categorical features: most-frequent imputation + one-hot encoding
   (`OneHotEncoder(handle_unknown="ignore")`).
8. All imputation/scaling/encoding steps were fit **only on the training
   split** and then applied to the test split and to new patient inputs,
   preventing data leakage.

---

## Chapter 3 — Technologies

### 3.1 Python
The core programming language used throughout the project (data processing,
modeling, and the web application).

### 3.2 Pandas
Used for loading, cleaning, filtering, grouping, and summarizing the tabular
dataset throughout preprocessing, EDA, and the dashboard.

### 3.3 NumPy
Used for numerical array operations underlying Pandas and Scikit-learn, and
for direct numeric operations such as building the confusion matrix array
for display.

### 3.4 Scikit-learn
Used for the preprocessing pipeline (`ColumnTransformer`, `Pipeline`,
`SimpleImputer`, `StandardScaler`, `OneHotEncoder`), the four classification
algorithms, the train/test split, and all evaluation metrics.

### 3.5 Streamlit
Used to build the entire interactive, multi-page web application without
writing any HTML/CSS/JavaScript framework code (light custom CSS is injected
for card styling only).

### 3.6 Plotly / Matplotlib / Seaborn
Plotly powers the interactive charts inside the Streamlit app (histograms,
bar charts, pie charts, gauge indicator, ROC curve, confusion matrix heatmap).
Matplotlib and Seaborn are used in the standalone Jupyter notebook for static
exploratory data analysis.

### 3.7 Joblib
Used to serialize (`joblib.dump`) and deserialize (`joblib.load`) the fitted
preprocessing pipeline and the trained model, so the Streamlit app can make
predictions instantly without retraining on every run.

---

## Chapter 4 — Methodology

### 4.1 Data Collection
The dataset was provided as a single CSV file of synthetic patient records
(no external data collection was performed).

### 4.2 Data Preprocessing
See Chapter 2.7 above — cleaning, meaningful missing-value handling, and a
scikit-learn `ColumnTransformer` for imputation, scaling, and encoding.

### 4.3 Exploratory Data Analysis
Conducted in `notebooks/disease_risk_analysis.ipynb` and mirrored
interactively in the Streamlit "Dashboard / Analytics" page. It covers:
target distribution, numeric feature distributions split by risk class,
categorical feature relationships with risk, and a correlation heatmap of
numeric features against the target.

### 4.4 Feature Engineering / Preprocessing
No synthetic feature engineering was added beyond the pipeline described in
2.7, in order to keep the pipeline transparent and avoid introducing
unjustified assumptions about the underlying (synthetic) data-generating
process. `Patient_ID` was explicitly excluded as a non-predictive identifier.

### 4.5 Train-Test Split
An 80/20 stratified split (`train_test_split(..., stratify=y,
random_state=42)`) was used to preserve the ~14.5% positive class ratio in
both sets and to ensure reproducibility.

- Training samples: 3,200
- Test samples: 800

### 4.6 Model Training
Four classification algorithms were trained on the identical, preprocessed
training data:

1. **Logistic Regression** (`class_weight="balanced"`)
2. **Decision Tree** (`max_depth=6`, `class_weight="balanced"`)
3. **Random Forest** (`n_estimators=300`, `max_depth=10`, `class_weight="balanced"`)
4. **Gradient Boosting** (`n_estimators=200`, `max_depth=3`)

`class_weight="balanced"` was applied where supported to counteract the
class imbalance in the target variable. A fixed `random_state=42` was used
throughout for reproducibility.

### 4.7 Model Selection
Models were ranked by **ROC-AUC** on the held-out test set (a metric robust
to class imbalance), with **F1-score** as a tie-breaker. This is an
objective, metric-driven selection process rather than an arbitrary choice.
**Gradient Boosting** was selected (see Chapter 6 for full results).

### 4.8 Evaluation
Each model was evaluated using Accuracy, Precision, Recall, F1-score,
ROC-AUC, and a Confusion Matrix, all computed on the 800-sample held-out
test set that the models never saw during training.

### 4.9 Prediction
The selected model and fitted preprocessor are persisted with `joblib`. The
`src/prediction.py` module loads these artifacts once and exposes a
`predict_risk()` function that transforms a single new patient's raw input
through the same fitted preprocessing pipeline and returns a risk label plus
class probabilities.

---

## Chapter 5 — Implementation

### `src/data_preprocessing.py`
Defines the dataset schema (numeric/categorical feature lists), and provides
`load_dataset()`, `clean_dataset()`, `split_features_target()`, and
`build_preprocessing_pipeline()` — the single source of truth for all data
handling, reused identically by training, evaluation, and the app.

### `src/train_model.py`
Orchestrates the full training pipeline: loads and cleans data, splits into
train/test, fits the preprocessing pipeline on the training data only,
trains all four candidate models, evaluates each on the test set, selects
the best model by ROC-AUC/F1, and saves the model, preprocessor, and a
`model_metrics.json` summary to the `models/` directory.

### `src/evaluate_model.py`
Provides `load_saved_metrics()` (used by the Streamlit "Model Performance"
page to display results instantly without retraining) and
`recompute_test_metrics()` (an independent sanity check that re-derives the
same test-set metrics from the saved model/preprocessor).

### `src/prediction.py`
Loads the saved model and preprocessor (`load_artifacts()`), validates a raw
patient input dictionary (`validate_patient_input()`), and produces a
prediction with class probabilities (`predict_risk()`) for use by the
Streamlit prediction form.

### `app.py`
The Streamlit application entry point. Implements sidebar navigation across
five pages (Home, Disease Risk Prediction, Dashboard / Analytics, Model
Performance, About Project), caches the dataset and model artifacts with
`st.cache_data` / `st.cache_resource`, and renders all interactive UI
elements, forms, and Plotly charts.

### `notebooks/disease_risk_analysis.ipynb`
A standalone Jupyter notebook covering data quality checks and exploratory
visualizations using Matplotlib/Seaborn, independent of the Streamlit app.

---

## Chapter 6 — Results

All metrics below were computed on the 800-sample held-out test set
(20% of the cleaned dataset, `random_state=42`) and are taken directly from
`models/model_metrics.json`, generated by actually running
`src/train_model.py` on the provided dataset. **No values in this section
were invented or estimated.**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8500 | 0.4905 | 0.8879 | 0.6319 | 0.9412 |
| Decision Tree | 0.8725 | 0.5389 | 0.8362 | 0.6554 | 0.8960 |
| Random Forest | 0.9288 | 0.7921 | 0.6897 | 0.7373 | 0.9641 |
| **Gradient Boosting** | **0.9375** | **0.8173** | **0.7328** | **0.7727** | **0.9737** |

**Selected model: Gradient Boosting** (highest ROC-AUC and highest F1-score
among all four candidates).

### Confusion Matrix — Gradient Boosting (test set, n=800)

|  | Predicted: No Risk | Predicted: At Risk |
|---|---|---|
| **Actual: No Risk** | 665 (True Negative) | 19 (False Positive) |
| **Actual: At Risk** | 31 (False Negative) | 85 (True Positive) |

### Interpretation

- Logistic Regression achieved the highest **recall** (0.888) but the lowest
  **precision** (0.491) — it catches most at-risk patients but with many
  false alarms.
- Gradient Boosting achieved the best overall balance, with the highest
  ROC-AUC (0.974), the highest precision (0.817), and a strong F1-score
  (0.773), making it the most reliable overall performer on this dataset.
- The full ROC curves and confusion matrices for all four models are
  available interactively in the "Model Performance" page of the Streamlit
  app, and in raw form in `models/model_metrics.json`.

---

## Chapter 7 — Application

### Home
Displays the project title, a short description, an explanation of how to
navigate the app, key dataset statistics (total records, at-risk rate,
average age, average BMI) shown as metric cards, and the medical disclaimer.

### Disease Risk Prediction
An interactive form collecting all 13 patient features (Age, Gender, BMI,
Smoking Status, Alcohol Consumption, Physical Activity Level, Systolic/
Diastolic Blood Pressure, Cholesterol, Glucose, Family History, Genetic Risk
Score, Previous Diagnosis). On submission, inputs are validated, passed
through the same fitted preprocessing pipeline used in training, and a
prediction (Low Risk / High Risk) is returned with a probability-based
confidence gauge and a plain-language list of contributing risk indicators
(explicitly noted as a simplified summary, not a model-internal explanation
or medical advice).

### Dashboard / Analytics
Provides age-range, gender, and smoking-status filters, followed by 12
interactive Plotly charts: Disease Risk distribution, Age distribution,
BMI distribution, Gender vs Disease Risk, Smoking Status vs Disease Risk,
Physical Activity vs Disease Risk, Family History vs Disease Risk, Previous
Diagnosis vs Disease Risk, Systolic Blood Pressure distribution, Cholesterol
distribution, Glucose distribution, and Genetic Risk Score distribution —
all recalculated live from the filtered data.

### Model Performance
Displays the selected model's name and rationale, its five headline metrics
as cards (Accuracy, Precision, Recall, F1-score, ROC-AUC), an interactive
confusion matrix heatmap, an interactive ROC curve, and a full side-by-side
comparison table and bar chart of all four trained models.

### About Project
Explains the problem statement, dataset, technologies, ML methodology,
application features, limitations, and future improvements, along with the
medical disclaimer.

---

## Chapter 8 — Limitations

- **Synthetic data:** the dataset does not represent real patients and may
  not reflect true real-world clinical relationships, correlations, or
  population distributions.
- **No external clinical validation:** the model's predictions have not been
  checked against real diagnostic outcomes or reviewed by medical
  professionals.
- **Class imbalance:** only ~14.5% of records are labeled "at risk", which
  constrains achievable precision/recall trade-offs and means metrics like
  raw accuracy can be misleading on their own (hence the emphasis on
  ROC-AUC and F1-score in model selection).
- **Uncalibrated probabilities:** the predicted probabilities have not been
  passed through a calibration step (e.g. Platt scaling), so they should be
  read as relative risk indicators, not precise/calibrated medical
  probabilities.
- **Potential bias:** as with any model trained on a fixed dataset, results
  may not generalize to populations or feature distributions not
  represented in the training data.
- **Not a diagnostic tool:** under no circumstances should this
  application's output be interpreted as, or substituted for, a medical
  diagnosis.

---

## Chapter 9 — Future Scope

- Train and validate on larger, real-world, properly consented and
  de-identified clinical datasets.
- Perform external validation against independent patient cohorts.
- Add Explainable AI techniques (SHAP, LIME) for transparent, per-prediction
  feature attribution.
- Apply probability calibration (e.g. Platt scaling, isotonic regression) so
  that predicted probabilities better reflect true observed frequencies.
- Incorporate additional clinical features (lab panels, medication history,
  comorbidities).
- Add model monitoring and scheduled retraining pipelines for production
  settings.
- Add secure deployment practices (HTTPS, containerization, access logging).
- Add authentication and privacy controls (e.g. role-based access, data
  encryption) before any handling of real patient data.

---

## Chapter 10 — Conclusion

This project implemented a complete, leakage-free machine learning pipeline
for disease risk prediction on a synthetic dataset, from raw data cleaning
through model comparison to a fully interactive Streamlit web application.
Four classification algorithms were trained and objectively compared using
Accuracy, Precision, Recall, F1-score, and ROC-AUC on a held-out test set,
with **Gradient Boosting** selected as the best-performing model
(ROC-AUC = 0.974, F1-score = 0.773, Accuracy = 93.75%). The resulting
application provides an accessible interface for exploring the dataset and
generating illustrative risk predictions, while being explicit throughout —
in the UI and in this report — that it is an educational tool built on
synthetic data and is not a substitute for professional medical diagnosis or
advice.
