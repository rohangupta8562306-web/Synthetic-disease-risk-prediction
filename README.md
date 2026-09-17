# 🩺 Synthetic Disease Risk Prediction

An end-to-end Machine Learning web application, built entirely in Python, that
predicts whether a patient is at risk of disease based on demographic,
lifestyle, and clinical indicators.

> ⚠️ **Disclaimer:** This project uses a **synthetic** dataset and is intended
> for **educational / portfolio purposes only**. It is **not a medical
> device**, does **not** provide a medical diagnosis, and must never be used
> as a substitute for professional medical advice. Always consult a qualified
> healthcare professional for real health decisions.

---

## 1. Project Overview

This project demonstrates a complete, production-style ML workflow:

1. Load and clean a synthetic patient health dataset.
2. Explore the data (EDA) to understand feature distributions and relationships.
3. Preprocess the data with a leakage-free scikit-learn pipeline.
4. Train and compare four classification algorithms.
5. Select the best model based on evaluation metrics.
6. Persist the trained model and preprocessing pipeline with `joblib`.
7. Serve everything through an interactive **Streamlit** web application.

## 2. Problem Statement

Given a set of demographic, lifestyle, and clinical attributes for a patient,
predict whether that patient falls into a "high risk" or "low risk" category
for disease, and present that prediction through an accessible, interactive
web interface.

## 3. Objectives

- Build a clean, reproducible ML pipeline (no data leakage).
- Compare multiple classification algorithms using proper evaluation metrics.
- Provide an interactive dashboard for both predictions and data exploration.
- Follow good software engineering practice: modular code, error handling,
  input validation, and clear documentation.

## 4. Dataset Information

- **File:** `data/Synthetic_disease_risk_dataset.csv`
- **Records:** ~4,000 synthetic patients
- **Columns (15):**

| Column | Description |
|---|---|
| `Patient_ID` | Unique patient identifier (excluded from modeling) |
| `Age` | Patient age in years |
| `Gender` | Female / Male / Other |
| `BMI` | Body Mass Index |
| `Smoking_Status` | Never / Former / Current |
| `Alcohol_Consumption` | None / Moderate / High |
| `Physical_Activity_Level` | Low / Moderate / High |
| `Blood_Pressure_Systolic` | Systolic blood pressure (mmHg) |
| `Blood_Pressure_Diastolic` | Diastolic blood pressure (mmHg) |
| `Cholesterol_Level` | Cholesterol level (mg/dL) |
| `Glucose_Level` | Glucose level (mg/dL) |
| `Family_History` | Family history of disease (Yes/No) |
| `Genetic_Risk_Score` | Continuous genetic risk score (0–1) |
| `Previous_Diagnosis` | None / Pre-disease / Diagnosed |
| `Disease_Risk` | **Target** — Yes / No |

The original dataset file is never modified by the code; all cleaning happens
on an in-memory copy at runtime.

## 5. Features

- **Home** — project overview, dataset snapshot, and key statistics.
- **Disease Risk Prediction** — interactive form → instant prediction with a
  confidence score, a risk gauge chart, and a plain-language explanation.
- **Dashboard / Analytics** — 12 interactive charts covering distributions and
  relationships across the dataset, with age/gender/smoking filters.
- **Model Performance** — accuracy, precision, recall, F1, ROC-AUC, confusion
  matrix, ROC curve, and a side-by-side comparison of all trained models.
- **About Project** — methodology, technologies, limitations, and future work.
- Light/Dark mode toggle, sidebar navigation, input validation, and error handling.

## 6. Technologies Used

| Category | Technology |
|---|---|
| Language | Python 3.9+ |
| Data handling | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Web app / frontend | Streamlit |
| Visualization | Plotly, Matplotlib, Seaborn |
| Model persistence | Joblib |
| File formats | CSV, Jupyter Notebook |

## 7. Machine Learning Algorithms

Four classification algorithms were trained and compared on an identical,
leakage-free preprocessed dataset:

- Logistic Regression
- Decision Tree
- Random Forest
- **Gradient Boosting** ← selected as the best model

Selection was based on **ROC-AUC** (robust to class imbalance), with
**F1-score** as a tie-breaker — not chosen arbitrarily. See
[`PROJECT_REPORT.md`](PROJECT_REPORT.md) for full results.

## 8. Project Structure

```text
synthetic-disease-risk-prediction/
│
├── app.py                        # Streamlit application (entry point)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── PROJECT_REPORT.md             # Detailed project report
│
├── .streamlit/
│   └── config.toml               # Streamlit theme configuration
│
├── data/
│   └── Synthetic_disease_risk_dataset.csv   # Original dataset (unmodified)
│
├── models/
│   ├── disease_risk_model.pkl           # Trained best model
│   ├── preprocessing_pipeline.pkl       # Fitted preprocessing pipeline
│   └── model_metrics.json               # Saved evaluation metrics
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Loading, cleaning, preprocessing pipeline
│   ├── train_model.py            # Model training, comparison & selection
│   ├── evaluate_model.py         # Metrics loading / re-evaluation utilities
│   └── prediction.py             # Single-patient prediction logic
│
├── notebooks/
│   └── disease_risk_analysis.ipynb   # Exploratory Data Analysis notebook
│
└── assets/
    └── screenshots/              # App screenshots (see below)
```

## 9. Installation Instructions

**Requirements:** Python 3.9 or later (works on Windows, macOS, and Linux).

1. Clone or download this project folder.
2. (Recommended) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 10. How to Run the Application

### Step 1 — Train the model (first time only)

```bash
python src/train_model.py
```

This loads the dataset, cleans it, trains and compares four models, selects
the best one, and saves `models/disease_risk_model.pkl` and
`models/preprocessing_pipeline.pkl`. Re-run this any time you want to retrain
(e.g. after changing the dataset).

### Step 2 — Launch the web app

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) — open
it in your browser.

## 11. Example Usage

1. Open the app and go to **Disease Risk Prediction** in the sidebar.
2. Fill in a patient's details, e.g.:
   - Age: 55, Gender: Male, BMI: 29.5, Smoking Status: Current
   - Blood Pressure: 145/92, Cholesterol: 230, Glucose: 160
   - Family History: Yes, Genetic Risk Score: 0.68, Previous Diagnosis: Pre-disease
3. Click **Predict Disease Risk**.
4. View the risk label (High Risk / Low Risk), confidence gauge, and the
   plain-language list of contributing indicators.

## 12. Model Evaluation (from the held-out test set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8500 | 0.4905 | 0.8879 | 0.6319 | 0.9412 |
| Decision Tree | 0.8725 | 0.5389 | 0.8362 | 0.6554 | 0.8960 |
| Random Forest | 0.9288 | 0.7921 | 0.6897 | 0.7373 | 0.9641 |
| **Gradient Boosting (selected)** | **0.9375** | **0.8173** | **0.7328** | **0.7727** | **0.9737** |

*(800 held-out test samples, 80/20 stratified split, `random_state=42`. Full
details, confusion matrices, and ROC curves in `PROJECT_REPORT.md` and the
in-app "Model Performance" page.)*

## 13. Screenshots

*Add screenshots of the running application here after your first local run:*

- `assets/screenshots/home.png`
- `assets/screenshots/prediction.png`
- `assets/screenshots/dashboard.png`
- `assets/screenshots/model_performance.png`

## 14. Limitations

- The dataset is **synthetic** and does not represent real patients.
- The model has not been validated against real-world clinical outcomes.
- The target class is imbalanced (~14.5% positive), which affects precision/recall trade-offs.
- Predicted probabilities are **not calibrated** medical probabilities.
- This tool must **not** be used for real medical decision-making.

## 15. Future Improvements

- Train on larger, real-world, properly consented clinical datasets.
- Add explainability (SHAP / LIME) for per-prediction feature attribution.
- Apply probability calibration (Platt scaling / isotonic regression).
- Add more clinical features (lab panels, medication history).
- Add model monitoring, scheduled retraining, authentication, and privacy controls.

## 16. Disclaimer

This application and all its predictions are provided for **educational and
research purposes only**. It is built on synthetic data and is **not a
certified medical device**. It does not diagnose, treat, cure, or prevent any
disease. Always consult a qualified healthcare professional for medical
concerns.

## 17. Author

Built as a portfolio / educational Machine Learning project demonstrating a
complete data-to-deployment workflow in Python.
