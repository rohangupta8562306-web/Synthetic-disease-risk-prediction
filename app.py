"""
app.py
------
Synthetic Disease Risk Prediction - Streamlit Web Application.

This is the single entry point for the application. It provides a
multi-page dashboard (via sidebar navigation) covering:
    - Home
    - Disease Risk Prediction (interactive form)
    - Dashboard / Analytics (EDA visualizations with filters)
    - Model Performance (evaluation metrics)
    - About Project

Run with:
    streamlit run app.py

IMPORTANT: This application is an educational / research prediction tool.
It is NOT a medical device and does NOT provide medical diagnoses. Always
consult a qualified healthcare professional for medical advice.
"""

import os
import sys
import json

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Make sure `src` and project root are importable regardless of the working directory the
# app is launched from (keeps things cross-platform / Windows & Streamlit Cloud friendly).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
for p in [BASE_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Register namespace aliases so pickle / joblib unpickling succeeds under both root and package modes
try:
    from src import data_preprocessing as _dp
    sys.modules.setdefault("data_preprocessing", _dp)
    sys.modules.setdefault("src.data_preprocessing", _dp)
except Exception:
    pass

from src.data_preprocessing import (
    load_dataset,
    clean_dataset,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.prediction import load_artifacts, predict_risk, validate_patient_input, ModelNotFoundError
from src.evaluate_model import load_saved_metrics

DATA_PATH = os.path.join(BASE_DIR, "data", "Synthetic_disease_risk_dataset.csv")

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Synthetic Disease Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def get_dataset():
    """Load and clean the dataset once, cached across reruns."""
    raw_df = load_dataset(DATA_PATH)
    cleaned_df = clean_dataset(raw_df)
    return cleaned_df


@st.cache_resource(show_spinner="Loading trained model...")
def get_model_artifacts():
    """Load the trained model + preprocessing pipeline once, cached."""
    return load_artifacts()


@st.cache_data(show_spinner=False)
def get_metrics():
    """Load pre-computed evaluation metrics once, cached."""
    return load_saved_metrics()


# ---------------------------------------------------------------------------
# Theme handling (lightweight light/dark toggle)
# ---------------------------------------------------------------------------
def inject_custom_css(dark_mode: bool):
    """
    Inject custom CSS for a professional look, plus a lightweight dark-mode
    approximation for the custom card components used on the Home and
    Prediction pages. Native Streamlit widgets follow the base theme set in
    .streamlit/config.toml; this toggle additionally re-styles the custom
    HTML cards so the whole page feels consistent.
    """
    if dark_mode:
        bg_color = "#0e1117"
        card_bg = "#1c1f26"
        text_color = "#f0f2f6"
        accent = "#4da6ff"
        border_color = "#2d3138"
    else:
        bg_color = "#ffffff"
        card_bg = "#f8f9fb"
        text_color = "#1a1a1a"
        accent = "#0066cc"
        border_color = "#e6e6e6"

    st.markdown(
        f"""
        <style>
            .metric-card {{
                background-color: {card_bg};
                border: 1px solid {border_color};
                border-radius: 12px;
                padding: 1.2rem;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            }}
            .metric-card h2 {{
                color: {accent};
                margin-bottom: 0.2rem;
                font-size: 1.8rem;
            }}
            .metric-card p {{
                color: {text_color};
                margin: 0;
                font-size: 0.9rem;
                opacity: 0.8;
            }}
            .result-card-high {{
                background-color: #ffe8e8;
                border-left: 6px solid #d9534f;
                border-radius: 10px;
                padding: 1.5rem;
                color: #7a1f1a;
            }}
            .result-card-low {{
                background-color: #e8f9ee;
                border-left: 6px solid #28a745;
                border-radius: 10px;
                padding: 1.5rem;
                color: #155724;
            }}
            .disclaimer-box {{
                background-color: #fff8e1;
                border-left: 6px solid #f0ad4e;
                border-radius: 8px;
                padding: 1rem 1.2rem;
                color: #6b5300;
                font-size: 0.92rem;
            }}
            .section-header {{
                border-bottom: 2px solid {accent};
                padding-bottom: 0.3rem;
                margin-bottom: 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer():
    st.markdown(
        """
        <div class="disclaimer-box">
        ⚠️ <strong>Disclaimer:</strong> This application is an <strong>educational / research
        prediction tool</strong> built on a <strong>synthetic dataset</strong>. It is
        <strong>not a medical device</strong> and does <strong>not</strong> provide a medical
        diagnosis. Predictions and probabilities shown here must not be used to make real
        healthcare decisions. Always consult a qualified healthcare professional.
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <h2>{value}</h2>
            <p>{label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# PAGE: Home
# ---------------------------------------------------------------------------
def render_home(df: pd.DataFrame):
    st.title("🩺 Synthetic Disease Risk Prediction")
    st.subheader("An educational Machine Learning project for disease risk estimation")

    st.markdown(
        """
        This application demonstrates a complete, end-to-end machine learning workflow —
        from raw data to an interactive prediction dashboard — applied to a **synthetic
        disease risk dataset**. It walks through data cleaning, exploratory data analysis,
        model training and comparison, and a live prediction interface, all built entirely
        in Python.

        Use the sidebar to navigate between:
        - **Disease Risk Prediction** — enter patient details and get an instant risk estimate.
        - **Dashboard / Analytics** — explore the dataset visually, with interactive filters.
        - **Model Performance** — see how the underlying model was evaluated and selected.
        - **About Project** — technical details, methodology, and limitations.
        """
    )

    render_disclaimer()

    st.markdown("---")
    st.markdown("### 📊 Dataset Snapshot")

    total_records = len(df)
    risk_rate = (df["Disease_Risk"] == "Yes").mean() * 100
    avg_age = df["Age"].mean()
    avg_bmi = df["BMI"].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Patient Records", f"{total_records:,}")
    with col2:
        metric_card("At-Risk Rate", f"{risk_rate:.1f}%")
    with col3:
        metric_card("Average Age", f"{avg_age:.1f} yrs")
    with col4:
        metric_card("Average BMI", f"{avg_bmi:.1f}")

    st.markdown("")
    with st.expander("Preview raw dataset (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Dataset column summary statistics"):
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: Disease Risk Prediction
# ---------------------------------------------------------------------------
def render_prediction_page():
    st.title("🔮 Disease Risk Prediction")
    st.write(
        "Fill in the patient's health information below and click **Predict Disease Risk** "
        "to get an instant, model-based risk estimate."
    )
    render_disclaimer()
    st.markdown("---")

    try:
        model, preprocessor = get_model_artifacts()
    except ModelNotFoundError as e:
        st.error(str(e))
        st.info("Run `python src/train_model.py` from the project root, then refresh this page.")
        return

    with st.form("patient_input_form"):
        st.markdown("#### Patient Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=120, value=45, step=1)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
            smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

        with col2:
            alcohol_consumption = st.selectbox(
                "Alcohol Consumption", ["None", "Moderate", "High"]
            )
            physical_activity = st.selectbox(
                "Physical Activity Level", ["Low", "Moderate", "High"]
            )
            systolic_bp = st.number_input(
                "Systolic Blood Pressure (mmHg)", min_value=70, max_value=250, value=120
            )
            diastolic_bp = st.number_input(
                "Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, value=80
            )

        with col3:
            cholesterol = st.number_input(
                "Cholesterol Level (mg/dL)", min_value=80, max_value=400, value=190
            )
            glucose = st.number_input(
                "Glucose Level (mg/dL)", min_value=50, max_value=400, value=100
            )
            family_history = st.selectbox("Family History of Disease", ["No", "Yes"])
            genetic_risk = st.slider(
                "Genetic Risk Score (0 = low, 1 = high)", min_value=0.0, max_value=1.0,
                value=0.5, step=0.01
            )
            previous_diagnosis = st.selectbox(
                "Previous Diagnosis", ["None", "Pre-disease", "Diagnosed"]
            )

        submitted = st.form_submit_button("🧪 Predict Disease Risk", use_container_width=True)

    if not submitted:
        return

    patient_data = {
        "Age": age,
        "Gender": gender,
        "BMI": bmi,
        "Smoking_Status": smoking_status,
        "Alcohol_Consumption": alcohol_consumption,
        "Physical_Activity_Level": physical_activity,
        "Blood_Pressure_Systolic": systolic_bp,
        "Blood_Pressure_Diastolic": diastolic_bp,
        "Cholesterol_Level": cholesterol,
        "Glucose_Level": glucose,
        "Family_History": family_history,
        "Genetic_Risk_Score": genetic_risk,
        "Previous_Diagnosis": previous_diagnosis,
    }

    # Input validation
    errors = validate_patient_input(patient_data)
    if errors:
        st.error("Please fix the following before predicting:")
        for err in errors:
            st.markdown(f"- {err}")
        return

    with st.spinner("Running prediction model..."):
        try:
            result = predict_risk(patient_data, model=model, preprocessor=preprocessor)
        except Exception as e:
            st.error(f"An error occurred while generating the prediction: {e}")
            return

    st.markdown("---")
    st.markdown("### 🧾 Prediction Result")

    is_high_risk = result["prediction_value"] == 1
    confidence = result["probability_high_risk"] if is_high_risk else result["probability_low_risk"]

    if is_high_risk:
        st.markdown(
            f"""
            <div class="result-card-high">
                <h2>⚠️ High Risk</h2>
                <p>The model estimates this patient profile shows patterns associated with
                elevated disease risk, with a model confidence of <strong>{confidence*100:.1f}%</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card-low">
                <h2>✅ Low Risk</h2>
                <p>The model estimates this patient profile shows patterns associated with
                lower disease risk, with a model confidence of <strong>{confidence*100:.1f}%</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["probability_high_risk"] * 100,
            title={"text": "Estimated Risk Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#d9534f" if is_high_risk else "#28a745"},
                "steps": [
                    {"range": [0, 33], "color": "#e8f9ee"},
                    {"range": [33, 66], "color": "#fff8e1"},
                    {"range": [66, 100], "color": "#ffe8e8"},
                ],
            },
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Why this result?")
        explanation_points = []
        if bmi >= 30:
            explanation_points.append("BMI is in the obese range (≥ 30).")
        if systolic_bp >= 140 or diastolic_bp >= 90:
            explanation_points.append("Blood pressure readings are in a high range.")
        if cholesterol >= 240:
            explanation_points.append("Cholesterol level is elevated.")
        if glucose >= 126:
            explanation_points.append("Glucose level is in a diabetic-range reading.")
        if smoking_status == "Current":
            explanation_points.append("Patient is a current smoker.")
        if family_history == "Yes":
            explanation_points.append("Patient has a family history of the disease.")
        if previous_diagnosis in ("Diagnosed", "Pre-disease"):
            explanation_points.append("Patient has a previous diagnosis or pre-disease record.")
        if genetic_risk >= 0.6:
            explanation_points.append("Genetic risk score is relatively high.")
        if physical_activity == "Low":
            explanation_points.append("Physical activity level is low.")

        if explanation_points:
            for point in explanation_points:
                st.markdown(f"- {point}")
        else:
            st.markdown(
                "- No major individual risk indicators stood out; the result reflects the "
                "combined pattern learned by the model across all input features."
            )

        st.caption(
            "This explanation lists common clinical risk indicators present in the input "
            "and is provided for context only — it is a simplified, rule-of-thumb summary, "
            "not a breakdown of the model's internal decision logic, and not medical advice."
        )


# ---------------------------------------------------------------------------
# PAGE: Dashboard / Analytics
# ---------------------------------------------------------------------------
def render_dashboard(df: pd.DataFrame):
    st.title("📈 Dashboard & Analytics")
    st.write("Explore patterns in the dataset using the filters below.")

    with st.expander("🔍 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            age_range = st.slider(
                "Age range", int(df["Age"].min()), int(df["Age"].max()),
                (int(df["Age"].min()), int(df["Age"].max()))
            )
        with col2:
            genders = st.multiselect(
                "Gender", options=sorted(df["Gender"].unique()),
                default=sorted(df["Gender"].unique())
            )
        with col3:
            smoking_options = st.multiselect(
                "Smoking Status", options=sorted(df["Smoking_Status"].unique()),
                default=sorted(df["Smoking_Status"].unique())
            )

    filtered_df = df[
        (df["Age"].between(age_range[0], age_range[1]))
        & (df["Gender"].isin(genders))
        & (df["Smoking_Status"].isin(smoking_options))
    ]

    if filtered_df.empty:
        st.warning("No records match the selected filters. Please broaden your filter selection.")
        return

    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} records based on current filters.")
    st.markdown("---")

    # Row 1: Disease Risk distribution + Age distribution
    col1, col2 = st.columns(2)
    with col1:
        risk_counts = filtered_df["Disease_Risk"].value_counts().reset_index()
        risk_counts.columns = ["Disease_Risk", "Count"]
        fig = px.pie(
            risk_counts, names="Disease_Risk", values="Count",
            title="Disease Risk Distribution", hole=0.4,
            color="Disease_Risk", color_discrete_map={"No": "#28a745", "Yes": "#d9534f"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            filtered_df, x="Age", color="Disease_Risk", nbins=30,
            title="Age Distribution by Disease Risk", barmode="overlay",
            color_discrete_map={"No": "#28a745", "Yes": "#d9534f"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: BMI distribution + Gender vs Disease Risk
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            filtered_df, x="BMI", color="Disease_Risk", nbins=30,
            title="BMI Distribution by Disease Risk", barmode="overlay",
            color_discrete_map={"No": "#28a745", "Yes": "#d9534f"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_risk = (
            filtered_df.groupby(["Gender", "Disease_Risk"]).size().reset_index(name="Count")
        )
        fig = px.bar(
            gender_risk, x="Gender", y="Count", color="Disease_Risk", barmode="group",
            title="Gender vs Disease Risk",
            color_discrete_map={"No": "#28a745", "Yes": "#d9534f"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Smoking vs Risk + Physical Activity vs Risk
    col1, col2 = st.columns(2)
    with col1:
        smoking_risk = (
            filtered_df.groupby(["Smoking_Status", "Disease_Risk"]).size().reset_index(name="Count")
        )
        fig = px.bar(
            smoking_risk, x="Smoking_Status", y="Count", color="Disease_Risk", barmode="group",
            title="Smoking Status vs Disease Risk",
            color_discrete_map={"No": "#28a745", "Yes": "#d9534f"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        activity_risk = (
            filtered_df.groupby(["Physical_Activity_Level", "Disease_Risk"]).size().reset_index(name="Count")
        )
        fig = px.bar(
            activity_risk, x="Physical_Activity_Level", y="Count", color="Disease_Risk", barmode="group",
            title="Physical Activity Level vs Disease Risk",
            category_orders={"Physical_Activity_Level": ["Low", "Moderate", "High"]},
            color_discrete_map={"No": "#28a745", "Yes": "#d9534f"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 4: Family History vs Risk + Previous Diagnosis vs Risk
    col1, col2 = st.columns(2)
    with col1:
        family_risk = (
            filtered_df.groupby(["Family_History", "Disease_Risk"]).size().reset_index(name="Count")
        )
        fig = px.bar(
            family_risk, x="Family_History", y="Count", color="Disease_Risk", barmode="group",
            title="Family History vs Disease Risk",
            color_discrete_map={"No": "#28a745", "Yes": "#d9534f"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        diagnosis_risk = (
            filtered_df.groupby(["Previous_Diagnosis", "Disease_Risk"]).size().reset_index(name="Count")
        )
        fig = px.bar(
            diagnosis_risk, x="Previous_Diagnosis", y="Count", color="Disease_Risk", barmode="group",
            title="Previous Diagnosis vs Disease Risk",
            color_discrete_map={"No": "#28a745", "Yes": "#d9534f"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 5: Blood Pressure + Cholesterol distributions
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            filtered_df, x="Blood_Pressure_Systolic", color="Disease_Risk", nbins=30,
            title="Systolic Blood Pressure Distribution", barmode="overlay",
            color_discrete_map={"No": "#28a745", "Yes": "#d9534f"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            filtered_df, x="Cholesterol_Level", color="Disease_Risk", nbins=30,
            title="Cholesterol Level Distribution", barmode="overlay",
            color_discrete_map={"No": "#28a745", "Yes": "#d9534f"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 6: Glucose + Genetic Risk Score distributions
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            filtered_df, x="Glucose_Level", color="Disease_Risk", nbins=30,
            title="Glucose Level Distribution", barmode="overlay",
            color_discrete_map={"No": "#28a745", "Yes": "#d9534f"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            filtered_df, x="Genetic_Risk_Score", color="Disease_Risk", nbins=30,
            title="Genetic Risk Score Distribution", barmode="overlay",
            color_discrete_map={"No": "#28a745", "Yes": "#d9534f"},
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: Model Performance
# ---------------------------------------------------------------------------
def render_model_performance():
    st.title("🧠 Model Performance")

    try:
        summary = get_metrics()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Run `python src/train_model.py` from the project root, then refresh this page.")
        return

    best_model_name = summary["best_model"]
    best_metrics = summary["all_model_results"][best_model_name]

    st.markdown(f"### Selected Model: **{best_model_name}**")
    st.caption(
        "Selected automatically based on ROC-AUC (with F1-score as a tie-breaker) on a "
        "held-out test set — not chosen arbitrarily."
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Accuracy", f"{best_metrics['accuracy']*100:.1f}%")
    with col2:
        metric_card("Precision", f"{best_metrics['precision']*100:.1f}%")
    with col3:
        metric_card("Recall", f"{best_metrics['recall']*100:.1f}%")
    with col4:
        metric_card("F1-Score", f"{best_metrics['f1_score']*100:.1f}%")
    with col5:
        metric_card("ROC-AUC", f"{best_metrics['roc_auc']*100:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Confusion Matrix")
        cm = np.array(best_metrics["confusion_matrix"])
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Risk", "At Risk"], y=["No Risk", "At Risk"],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### ROC Curve")
        fpr = best_metrics["roc_curve"]["fpr"]
        tpr = best_metrics["roc_curve"]["tpr"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{best_model_name} (AUC = {best_metrics['roc_auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline", line=dict(dash="dash")))
        fig.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Comparison Across All Trained Models")

    comparison_rows = []
    for model_name, metrics in summary["all_model_results"].items():
        comparison_rows.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1_score"],
            "ROC-AUC": metrics["roc_auc"],
        })
    comparison_df = pd.DataFrame(comparison_rows).sort_values("ROC-AUC", ascending=False)
    st.dataframe(
        comparison_df.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1-Score": "{:.4f}", "ROC-AUC": "{:.4f}",
        }).highlight_max(subset=["ROC-AUC"], color="#d4f5dd"),
        use_container_width=True,
    )

    fig = px.bar(
        comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model", y="Score", color="Metric", barmode="group",
        title="Model Comparison Across Metrics",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Training used {summary['n_train_samples']:,} samples; evaluation used "
        f"{summary['n_test_samples']:,} held-out samples not seen during training "
        f"(random_state = {summary['random_state']} for reproducibility)."
    )


# ---------------------------------------------------------------------------
# PAGE: About Project
# ---------------------------------------------------------------------------
def render_about():
    st.title("ℹ️ About This Project")

    st.markdown("""
### Problem Statement
Chronic diseases are often linked to a combination of lifestyle, clinical, and genetic
factors. This project explores whether a machine learning model can learn patterns from
patient-level health indicators to estimate disease risk, using a synthetic dataset
designed to resemble realistic clinical data.

### Dataset
A synthetic dataset of ~4,000 patient records with 15 columns, covering demographic
information (Age, Gender), lifestyle factors (Smoking Status, Alcohol Consumption,
Physical Activity Level), clinical measurements (BMI, Blood Pressure, Cholesterol,
Glucose), and history-based indicators (Family History, Genetic Risk Score, Previous
Diagnosis), with `Disease_Risk` (Yes/No) as the target variable.

### Technologies Used
- **Python** — core programming language
- **Pandas / NumPy** — data manipulation and numerical computation
- **Scikit-learn** — preprocessing, model training, and evaluation
- **Streamlit** — interactive web application framework
- **Plotly** — interactive charts and visualizations
- **Joblib** — model persistence

### Machine Learning Methodology
1. Data cleaning (duplicate removal, meaningful missing-value handling).
2. Feature/target separation, with the patient identifier excluded to avoid leakage.
3. An 80/20 stratified train/test split with a fixed random seed for reproducibility.
4. A `ColumnTransformer`-based preprocessing pipeline (median/most-frequent imputation,
   standard scaling for numeric features, one-hot encoding for categorical features),
   fit **only** on the training data to prevent data leakage.
5. Training and comparing four classifiers: Logistic Regression, Decision Tree,
   Random Forest, and Gradient Boosting — all with a fixed `random_state`.
6. Model selection based on ROC-AUC (tie-broken by F1-score) on the held-out test set —
   not chosen arbitrarily.
7. Persisting the fitted preprocessor and best model with `joblib` for reuse without
   retraining.

### Features of the Application
- **Home** — project overview and dataset snapshot.
- **Disease Risk Prediction** — interactive form producing an instant risk estimate with
  a confidence score and a plain-language explanation of contributing indicators.
- **Dashboard / Analytics** — filterable, interactive visualizations of the dataset.
- **Model Performance** — evaluation metrics, confusion matrix, ROC curve, and a
  side-by-side comparison of all trained models.

### Limitations
- The dataset is **synthetic**; it does not represent real patient data and may not
  reflect real-world clinical relationships or population distributions.
- The model has not been externally validated against real clinical outcomes.
- Class imbalance (~14.5% positive class) means recall/precision trade-offs should be
  interpreted carefully.
- The model provides a statistical estimate based on patterns in this dataset only — it
  is **not** a diagnostic tool.
- No formal probability calibration (e.g. Platt scaling / isotonic regression) has been
  applied, so predicted probabilities should be treated as relative risk indicators
  rather than precise/calibrated medical probabilities.

### Future Improvements
- Train and validate on larger, real-world (properly consented and de-identified)
  clinical datasets.
- Add explainability tooling such as SHAP or LIME for per-prediction feature attribution.
- Apply probability calibration techniques.
- Incorporate additional clinical features (e.g. lab panels, medication history).
- Add model monitoring and periodic retraining in a production setting.
- Add authentication and privacy controls before any real-world deployment.
""")

    render_disclaimer()


# ---------------------------------------------------------------------------
# Main app entry point
# ---------------------------------------------------------------------------
def main():
    st.sidebar.title("🩺 Navigation")

    dark_mode = st.sidebar.toggle("🌙 Dark mode", value=False)
    inject_custom_css(dark_mode)

    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Disease Risk Prediction",
            "Dashboard / Analytics",
            "Model Performance",
            "About Project",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Synthetic Disease Risk Prediction v1.0\n\n"
        "Educational project — not for real medical use."
    )

    try:
        df = get_dataset()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return

    if page == "Home":
        render_home(df)
    elif page == "Disease Risk Prediction":
        render_prediction_page()
    elif page == "Dashboard / Analytics":
        render_dashboard(df)
    elif page == "Model Performance":
        render_model_performance()
    elif page == "About Project":
        render_about()


if __name__ == "__main__":
    main()
