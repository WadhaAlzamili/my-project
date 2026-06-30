# =========================
# IMPORT LIBRARIES
# =========================
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prophet import Prophet

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Hospital Clinical Decision Support System",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# USERS & ROLES
# =========================
USERS = {
    "nurse": {"password": "1234", "role": "Nurse"},
    "doctor": {"password": "1234", "role": "Doctor"},
    "admin": {"password": "1234", "role": "Admin"},
    "developer": {"password": "1234", "role": "Developer"}
}


# =========================
# SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None


# =========================
# FILE PATHS
# =========================
RISK_PATH = "final_risk_dataset.csv"
FORECAST_PATH = "healthcare_analytics_patient_flow_data.csv"
VITALS_PATH = "vitals_demo.csv"


# =========================
# GLOBAL PROFESSIONAL THEME
# =========================
st.markdown("""
<style>
:root {
  --primary: #0f4c81;
  --primary-soft: #dceef8;
  --blue-soft: #eaf8ff;
  --card: #ffffff;
  --text: #0b1f3a;
  --muted: #5f6f82;
  --border: #d9eaf4;
  --shadow: 0 8px 22px rgba(15, 76, 129, 0.10);
}

.stApp {
  background: linear-gradient(135deg, #f5fcff, #e8f8ff);
}

.block-container {
  max-width: 1320px;
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}

html, body, [class*="css"] {
  font-family: "Segoe UI", Arial, sans-serif;
  color: var(--text);
}

h1, h2, h3, h4 {
  color: var(--text);
  font-weight: 800;
}

.app-header {
  background: linear-gradient(135deg, #ffffff, #e8f9ff);
  border: 1px solid var(--border);
  border-radius: 28px;
  padding: 24px 28px;
  box-shadow: var(--shadow);
  margin-bottom: 22px;
}

.app-header h2 {
  margin: 0;
  color: var(--primary);
  font-size: 30px;
}

.app-subtitle {
  color: var(--muted);
  margin-top: 8px;
  font-size: 15px;
}

.card {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 22px;
  box-shadow: var(--shadow);
  margin-bottom: 18px;
}

.card-title {
  font-size: 22px;
  font-weight: 900;
  color: var(--text);
  margin-bottom: 6px;
}

.muted {
  color: var(--muted);
  font-size: 15px;
}

.section-title {
  margin-top: 18px;
  margin-bottom: 14px;
  color: var(--primary);
  font-size: 24px;
  font-weight: 900;
}

.metric-card {
  background: #ffffff;
  border-radius: 20px;
  padding: 20px 22px;
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
  margin-bottom: 18px;
  min-height: 128px;
}

.metric-label {
  font-size: 14px;
  color: var(--muted);
  font-weight: 800;
  margin-bottom: 10px;
}

.metric-value {
  font-size: 34px;
  font-weight: 950;
  color: var(--text);
  line-height: 1.1;
}

.metric-sub {
  font-size: 13px;
  color: var(--muted);
  margin-top: 10px;
}

.metric-blue {
  border-left: 7px solid #0f4c81;
}

.metric-low {
  border-left: 7px solid #1f9d55;
  background: linear-gradient(135deg, #ffffff, #f0fff6);
}

.metric-medium {
  border-left: 7px solid #f59e0b;
  background: linear-gradient(135deg, #ffffff, #fff8e8);
}

.metric-high {
  border-left: 7px solid #dc2626;
  background: linear-gradient(135deg, #ffffff, #fff1f1);
}

.metric-purple {
  border-left: 7px solid #7c3aed;
  background: linear-gradient(135deg, #ffffff, #f5f3ff);
}

.dashboard-panel {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 18px;
  box-shadow: var(--shadow);
  margin-bottom: 18px;
}

.side-card {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 20px;
  box-shadow: var(--shadow);
  margin-bottom: 18px;
}

.side-title {
  color: var(--muted);
  font-size: 14px;
  font-weight: 800;
  margin-bottom: 8px;
}

.side-highlight {
  color: #1593a5;
  font-size: 20px;
  font-weight: 950;
  margin-bottom: 8px;
}

.alert-card {
  background: #fff1f1;
  border-left: 7px solid #dc2626;
  border-radius: 18px;
  padding: 18px;
  margin-bottom: 16px;
}

.warn-card {
  background: #fff8e8;
  border-left: 7px solid #f59e0b;
  border-radius: 18px;
  padding: 18px;
  margin-bottom: 16px;
}

.ok-card {
  background: #f0fff6;
  border-left: 7px solid #1f9d55;
  border-radius: 18px;
  padding: 18px;
  margin-bottom: 16px;
}

.stButton > button {
  background: linear-gradient(90deg, #0f4c81, #1593a5);
  color: white;
  border: none;
  border-radius: 14px;
  padding: 10px 22px;
  font-weight: 800;
}

.stButton > button:hover {
  background: linear-gradient(90deg, #0c3d67, #0f7f8e);
  color: white;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 10px;
}

.stTabs [data-baseweb="tab"] {
  background: #ffffff;
  border-radius: 14px;
  padding: 10px 22px;
  box-shadow: var(--shadow);
  font-weight: 800;
}

.stTabs [aria-selected="true"] {
  background: var(--primary-soft) !important;
  color: var(--primary) !important;
}

section[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid var(--border);
}

[data-testid="stDataFrame"] {
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid var(--border);
}

.login-wrapper {
  max-width: 540px;
  margin: 90px auto;
}

.login-box {
  background: #ffffff;
  border-radius: 26px;
  padding: 34px;
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)


# =========================
# HELPER FUNCTIONS
# =========================
def render_metric_card(label, value, subtext="", level="blue"):
    st.markdown(f"""
    <div class="metric-card metric-{level}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)


def render_side_card(title, highlight, text):
    st.markdown(f"""
    <div class="side-card">
        <div class="side-title">{title}</div>
        <div class="side-highlight">{highlight}</div>
        <div class="muted">{text}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# LOGIN PAGE
# =========================
if not st.session_state.logged_in:
    st.markdown("""
    <div class="login-wrapper">
        <div class="login-box">
            <div class="card-title">Hospital Clinical Decision Support System</div>
            <div class="muted">Secure access for nurses, doctors, admins, and developers.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign In"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.role = USERS[username]["role"]
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()


# =========================
# MAIN HEADER
# =========================
st.markdown("""
<div class="app-header">
    <h2>HealthCare Dashboard</h2>
    <div class="app-subtitle">
        Clinical decision support system for risk screening, ICU monitoring, and admissions forecasting.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# SIDEBAR
# =========================
st.sidebar.title("System Settings")
st.sidebar.markdown(f"**Logged in as:** {st.session_state.role}")

if st.session_state.role == "Developer":
    developer_mode = st.sidebar.toggle("Developer Mode", value=False)
else:
    developer_mode = False

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.role = None
    st.rerun()


# =========================
# TABS BY ROLE
# =========================
role = st.session_state.role

if role == "Admin":
    tab2, = st.tabs(["Admissions Forecast"])

elif role in ["Nurse", "Doctor"]:
    tab1, tab3 = st.tabs([
        "Risk Screening",
        "ICU Patient Monitoring"
    ])

elif role == "Developer":
    tab1, tab2, tab3 = st.tabs([
        "Risk Screening",
        "Admissions Forecast",
        "ICU Patient Monitoring"
    ])


# =========================
# TAB 2: ADMIN ADMISSIONS FORECAST
# =========================
if role in ["Admin", "Developer"]:
    with tab2:

        st.markdown("""
        <div class="card">
            <div class="card-title">Admissions Forecast</div>
            <div class="muted">
                Admin dashboard for hospital capacity, admissions trends, ICU load, and operational recommendations.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # =========================
        # CREATE / LOAD DATA
        # =========================
        if os.path.exists(FORECAST_PATH):
            try:
                df_admin = pd.read_csv(FORECAST_PATH)
            except Exception:
                df_admin = pd.DataFrame()
        else:
            df_admin = pd.DataFrame()

        # Sample data if dataset is missing
        if df_admin.empty or "Patient Admission Date" not in df_admin.columns:
            np.random.seed(42)

            dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
            admission_types = ["Elective", "Emergency", "Urgent"]
            genders = ["Female", "Male"]
            diagnoses = ["Diabetes", "Hypertension", "Obesity", "Cancer", "Asthma", "Arthritis"]
            medications = ["Metformin", "Aspirin", "Insulin", "Amoxicillin", "Lipresin"]

            rows = []

            for date in dates:
                daily_count = np.random.randint(8, 22)

                for _ in range(daily_count):
                    rows.append({
                        "Patient Admission Date": date,
                        "Admission Type": np.random.choice(
                            admission_types,
                            p=[0.40, 0.45, 0.15]
                        ),
                        "Gender": np.random.choice(
                            genders,
                            p=[0.52, 0.48]
                        ),
                        "Diagnosis": np.random.choice(diagnoses),
                        "Medication": np.random.choice(medications)
                    })

            df_admin = pd.DataFrame(rows)

        # =========================
        # CLEAN DATA
        # =========================
        df_admin["Patient Admission Date"] = pd.to_datetime(
            df_admin["Patient Admission Date"],
            errors="coerce"
        )

        df_admin = df_admin.dropna(subset=["Patient Admission Date"]).copy()

        if "Admission Type" not in df_admin.columns:
            df_admin["Admission Type"] = np.random.choice(
                ["Elective", "Emergency", "Urgent"],
                size=len(df_admin)
            )

        if "Gender" not in df_admin.columns:
            df_admin["Gender"] = np.random.choice(
                ["Female", "Male"],
                size=len(df_admin)
            )
        else:
            df_admin["Gender"] = df_admin["Gender"].astype(str).str.title()
            df_admin = df_admin[df_admin["Gender"].isin(["Female", "Male"])].copy()

        if "Diagnosis" not in df_admin.columns:
            df_admin["Diagnosis"] = np.random.choice(
                ["Diabetes", "Hypertension", "Obesity", "Cancer", "Asthma", "Arthritis"],
                size=len(df_admin)
            )

        if "Medication" not in df_admin.columns:
            df_admin["Medication"] = np.random.choice(
                ["Metformin", "Aspirin", "Insulin", "Amoxicillin", "Lipresin"],
                size=len(df_admin)
            )

        # =========================
        # KPI VALUES
        # =========================
        current_patients = 124
        total_beds = 150
        available_beds = total_beds - current_patients

        icu_patients = 14
        icu_total_beds = 20

        er_today = 18
        discharged_today = 11
        high_risk_today = 6
        doctors_available = 40

        bed_occupancy = (current_patients / total_beds) * 100
        icu_occupancy = (icu_patients / icu_total_beds) * 100

        bed_level = "high" if bed_occupancy >= 85 else "medium" if bed_occupancy >= 65 else "low"
        icu_level = "high" if icu_occupancy >= 80 else "medium" if icu_occupancy >= 60 else "low"

        # =========================
        # HOSPITAL OVERVIEW CARDS
        # =========================
        st.markdown('<div class="section-title">Hospital Overview</div>', unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            render_metric_card(
                "Current Patients",
                current_patients,
                "Currently admitted",
                bed_level
            )

        with k2:
            render_metric_card(
                "ER Admissions Today",
                er_today,
                "Emergency patients today",
                "medium"
            )

        with k3:
            render_metric_card(
                "Available Beds",
                available_beds,
                "Remaining capacity",
                "low"
            )

        with k4:
            render_metric_card(
                "ICU Occupancy",
                f"{icu_occupancy:.1f}%",
                "ICU capacity",
                icu_level
            )

        k5, k6, k7, k8 = st.columns(4)

        with k5:
            render_metric_card(
                "Doctors Available",
                doctors_available,
                "Available doctors",
                "blue"
            )

        with k6:
            render_metric_card(
                "Discharged Today",
                discharged_today,
                "Patients discharged",
                "low"
            )

        with k7:
            render_metric_card(
                "High-Risk Patients",
                high_risk_today,
                "Need attention",
                "high"
            )

        with k8:
            render_metric_card(
                "Bed Occupancy",
                f"{bed_occupancy:.1f}%",
                "Hospital capacity",
                bed_level
            )

        # =========================
        # ADMISSIONS ANALYTICS
        # =========================
        st.markdown('<div class="section-title">Admissions Analytics</div>', unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns([2, 1])

        with chart_col1:
            st.markdown("""
            <div class="card">
                <div class="card-title">Admitted Patients Trend by Month</div>
                <div class="muted">Monthly admissions grouped by admission type.</div>
            </div>
            """, unsafe_allow_html=True)

            monthly_type = (
                df_admin
                .assign(Month=df_admin["Patient Admission Date"].dt.strftime("%b"))
                .groupby(["Month", "Admission Type"])
                .size()
                .reset_index(name="Admissions")
            )

            month_order = [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
            ]

            monthly_type["Month"] = pd.Categorical(
                monthly_type["Month"],
                categories=month_order,
                ordered=True
            )

            monthly_type = monthly_type.sort_values("Month")

            fig_line = px.line(
                monthly_type,
                x="Month",
                y="Admissions",
                color="Admission Type",
                markers=True,
                color_discrete_map={
                    "Emergency": "#6ec6ff",
                    "Urgent": "#0f4c81",
                    "Elective": "#ff8a8a"
                }
            )

            fig_line.update_layout(
                height=390,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="white",
                plot_bgcolor="white",
                legend_title_text="Admission Type"
            )

            st.plotly_chart(fig_line, use_container_width=True)

        with chart_col2:
            st.markdown("""
            <div class="card">
                <div class="card-title">Admitted Patients by Gender</div>
                <div class="muted">Gender distribution for admitted patients.</div>
            </div>
            """, unsafe_allow_html=True)

            gender_df = df_admin["Gender"].value_counts().reset_index()
            gender_df.columns = ["Gender", "Count"]

            fig_donut = px.pie(
                gender_df,
                names="Gender",
                values="Count",
                hole=0.55,
                color="Gender",
                color_discrete_map={
                    "Female": "#6ec6ff",
                    "Male": "#0f4c81"
                }
            )

            fig_donut.update_layout(
                height=390,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="white"
            )

            st.plotly_chart(fig_donut, use_container_width=True)

        # =========================
        # DIAGNOSIS + SIDE CARDS
        # =========================
        st.markdown('<div class="section-title">Clinical Summary</div>', unsafe_allow_html=True)

        bar_col, side_col = st.columns([2, 1])

        with bar_col:
            st.markdown("""
            <div class="card">
                <div class="card-title">Patients by Diagnosis</div>
                <div class="muted">Most common diagnoses among admitted patients.</div>
            </div>
            """, unsafe_allow_html=True)

            diagnosis_bar = df_admin["Diagnosis"].value_counts().reset_index()
            diagnosis_bar.columns = ["Diagnosis", "Patients"]

            fig_bar = px.bar(
                diagnosis_bar,
                x="Patients",
                y="Diagnosis",
                orientation="h",
                text="Patients",
                color="Diagnosis",
                color_discrete_sequence=[
                    "#8ac6d1", "#6ec6ff", "#0f4c81",
                    "#ffb703", "#90be6d", "#f28482"
                ]
            )

            fig_bar.update_layout(
                height=430,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="white",
                plot_bgcolor="white",
                showlegend=False,
                yaxis=dict(categoryorder="total ascending")
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        with side_col:
            top_med = df_admin["Medication"].value_counts().idxmax()
            top_med_count = df_admin["Medication"].value_counts().max()

            top_condition = df_admin["Diagnosis"].value_counts().idxmax()
            top_condition_count = df_admin["Diagnosis"].value_counts().max()

            render_side_card(
                "Top Medication",
                top_med,
                f"{top_med_count} prescriptions were recorded for {top_med}."
            )

            render_side_card(
                "Critical ICU Cases",
                icu_patients,
                f"{icu_patients} patients currently require ICU monitoring."
            )

            render_side_card(
                "Most Common Condition",
                top_condition,
                f"{top_condition_count} patients were diagnosed with {top_condition}."
            )

        # =========================
        # PROPHET FORECAST
        # =========================
        st.markdown('<div class="section-title">Patient Admissions Forecast</div>', unsafe_allow_html=True)

        daily_admissions = (
            df_admin
            .groupby("Patient Admission Date")
            .size()
            .reset_index(name="Total_Admissions")
            .sort_values("Patient Admission Date")
        )

        prophet_df = daily_admissions.rename(
            columns={
                "Patient Admission Date": "ds",
                "Total_Admissions": "y"
            }
        )

        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_df["y"] = prophet_df["y"].astype(float)

        forecast_days = st.slider(
            "Select forecast period in days",
            min_value=7,
            max_value=90,
            value=30
        )

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )

        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        forecast_future = forecast.tail(forecast_days)

        avg_forecast = forecast_future["yhat"].mean()
        peak_forecast = forecast_future["yhat"].max()

        if avg_forecast >= 18:
            demand_level = "High"
            demand_color = "high"
        elif avg_forecast >= 12:
            demand_level = "Medium"
            demand_color = "medium"
        else:
            demand_level = "Low"
            demand_color = "low"

        f1, f2, f3 = st.columns(3)

        with f1:
            render_metric_card(
                "Expected Demand Level",
                demand_level,
                "Based on forecast",
                demand_color
            )

        with f2:
            render_metric_card(
                "Average Predicted Admissions",
                f"{avg_forecast:.1f}",
                "Per day",
                demand_color
            )

        with f3:
            render_metric_card(
                "Peak Predicted Admissions",
                f"{peak_forecast:.1f}",
                "Highest expected day",
                demand_color
            )

        forecast_plot = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

        fig_forecast = go.Figure()

        fig_forecast.add_trace(go.Scatter(
            x=forecast_plot["ds"],
            y=forecast_plot["yhat"],
            mode="lines",
            name="Predicted Admissions",
            line=dict(color="#0f4c81", width=3)
        ))

        fig_forecast.add_trace(go.Scatter(
            x=forecast_plot["ds"],
            y=forecast_plot["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ))

        fig_forecast.add_trace(go.Scatter(
            x=forecast_plot["ds"],
            y=forecast_plot["yhat_lower"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="Confidence Range"
        ))

        fig_forecast.update_layout(
            height=420,
            title="Patient Admissions Forecast",
            xaxis_title="Date",
            yaxis_title="Admissions",
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=50, b=10)
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        # =========================
        # OPERATIONAL RECOMMENDATION
        # =========================
        st.markdown('<div class="section-title">Operational Recommendation Summary</div>', unsafe_allow_html=True)

        if demand_level == "High":
            st.markdown("""
            <div class="alert-card">
                <b>High Demand Recommendation</b><br>
                Predicted admissions are high. Increase staffing, prepare additional beds,
                monitor ICU capacity, and activate surge planning.
            </div>
            """, unsafe_allow_html=True)

        elif demand_level == "Medium":
            st.markdown("""
            <div class="warn-card">
                <b>Medium Demand Recommendation</b><br>
                Maintain current staffing, monitor ER admissions, and keep backup resources ready.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="ok-card">
                <b>Low Demand Recommendation</b><br>
                Normal staffing is sufficient. Continue monitoring patient flow and bed availability.
            </div>
            """, unsafe_allow_html=True)

        # =========================
        # FORECAST DATA TABLE + DOWNLOAD
        # =========================
        with st.expander("View Forecast Data"):
            display_forecast = forecast_future[[
                "ds",
                "yhat",
                "yhat_lower",
                "yhat_upper"
            ]].copy()

            display_forecast.columns = [
                "Date",
                "Predicted Admissions",
                "Lower Estimate",
                "Upper Estimate"
            ]

            st.dataframe(display_forecast, use_container_width=True)

        csv_report = display_forecast.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Forecast Report",
            data=csv_report,
            file_name="admissions_forecast_report.csv",
            mime="text/csv"
        )

# =========================
# RISK MODEL FUNCTIONS
# =========================
@st.cache_resource
def train_risk_models(csv_path: str):
    df = pd.read_csv(csv_path)

    required_cols = ["Age", "Gender", "Symptoms", "Risk_Level"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in risk dataset: {missing}")

    df = df.dropna(subset=required_cols).copy()

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()
    df["Symptoms"] = df["Symptoms"].astype(str).str.strip().str.lower()
    df["Risk_Level"] = df["Risk_Level"].astype(str).str.strip().str.title()

    df = df.dropna(subset=["Age"]).copy()
    df = df[df["Gender"].isin(["male", "female"])].copy()
    df = df[df["Risk_Level"].isin(["Low", "High"])].copy()

    if df.empty:
        raise ValueError("Risk dataset is empty after preprocessing.")

    df["Gender_Encoded"] = df["Gender"].map({"female": 0, "male": 1})

    symptom_keywords = [
        "fever",
        "chest pain",
        "shortness of breath",
        "vomiting",
        "dizziness",
        "swelling",
        "fatigue",
        "blurred vision"
    ]

    for symptom in symptom_keywords:
        col_name = symptom.replace(" ", "_")
        df[col_name] = df["Symptoms"].apply(lambda x: 1 if symptom in x else 0)

    symptom_cols = [s.replace(" ", "_") for s in symptom_keywords]

    df["Symptom_Count"] = df[symptom_cols].sum(axis=1)
    df["Risk_Label"] = df["Risk_Level"].map({"Low": 0, "High": 1})

    feature_cols = [
        "Age",
        "Gender_Encoded",
        "Symptom_Count",
        "fever",
        "chest_pain",
        "shortness_of_breath",
        "vomiting",
        "dizziness",
        "swelling",
        "fatigue",
        "blurred_vision"
    ]

    X = df[feature_cols]
    y = df["Risk_Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }

    model_results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        model_results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True)
        }

        trained_models[name] = model

    best_model_name = max(
        model_results,
        key=lambda x: model_results[x]["accuracy"]
    )

    best_model = trained_models[best_model_name]

    return best_model, best_model_name, model_results, df


def build_input_features(age, gender, selected_symptoms):
    symptom_flags = {
        "fever": 1 if "fever" in selected_symptoms else 0,
        "chest_pain": 1 if "chest pain" in selected_symptoms else 0,
        "shortness_of_breath": 1 if "shortness of breath" in selected_symptoms else 0,
        "vomiting": 1 if "vomiting" in selected_symptoms else 0,
        "dizziness": 1 if "dizziness" in selected_symptoms else 0,
        "swelling": 1 if "swelling" in selected_symptoms else 0,
        "fatigue": 1 if "fatigue" in selected_symptoms else 0,
        "blurred_vision": 1 if "blurred vision" in selected_symptoms else 0,
    }

    symptom_count = sum(symptom_flags.values())
    gender_encoded = 1 if gender.lower() == "male" else 0

    input_df = pd.DataFrame([{
        "Age": float(age),
        "Gender_Encoded": gender_encoded,
        "Symptom_Count": symptom_count,
        "fever": symptom_flags["fever"],
        "chest_pain": symptom_flags["chest_pain"],
        "shortness_of_breath": symptom_flags["shortness_of_breath"],
        "vomiting": symptom_flags["vomiting"],
        "dizziness": symptom_flags["dizziness"],
        "swelling": symptom_flags["swelling"],
        "fatigue": symptom_flags["fatigue"],
        "blurred_vision": symptom_flags["blurred_vision"]
    }])

    return input_df, symptom_count


def generate_risk_actions(level: str) -> str:
    if level == "High":
        return """
        <div class="alert-card">
            <b>Recommended Nursing Actions</b><br>
            <ul>
                <li>Notify physician or rapid response team immediately.</li>
                <li>Reassess vital signs within 5 minutes.</li>
                <li>Prepare oxygen support and monitor saturation closely.</li>
                <li>Review medication schedule and verify urgent medications.</li>
            </ul>
        </div>
        """

    if level == "Medium":
        return """
        <div class="warn-card">
            <b>Recommended Nursing Actions</b><br>
            <ul>
                <li>Repeat vital signs within 15 minutes.</li>
                <li>Monitor symptoms closely and document any deterioration.</li>
                <li>Ensure medications are administered on time.</li>
                <li>Escalate if symptoms worsen.</li>
            </ul>
        </div>
        """

    return """
    <div class="ok-card">
        <b>Recommended Nursing Actions</b><br>
        <ul>
            <li>Continue standard monitoring and routine reassessment.</li>
            <li>Document current symptoms and observations.</li>
            <li>Follow normal ward protocol unless the condition changes.</li>
        </ul>
    </div>
    """


# =========================
# TAB 1: RISK SCREENING
# =========================
if role in ["Nurse", "Doctor", "Developer"]:
    with tab1:

        st.markdown("""
        <div class="card">
            <div class="card-title">Patient Risk Screening</div>
            <div class="muted">
                Machine learning-based patient risk assessment using age, gender, symptoms, and vital signs.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not os.path.exists(RISK_PATH):
            st.error("final_risk_dataset.csv not found. Please place it in the project folder.")
            st.stop()

        try:
            best_model, best_model_name, model_results, df_risk = train_risk_models(RISK_PATH)
        except Exception as e:
            st.error(f"Failed to train risk models: {e}")
            st.stop()

        st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)

        p1, p2 = st.columns(2)

        with p1:
            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=30,
                step=1
            )

        with p2:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

        st.markdown('<div class="section-title">Symptoms</div>', unsafe_allow_html=True)

        selected_symptoms = []

        s1, s2 = st.columns(2)

        with s1:
            fever = st.checkbox("Fever")
            chest_pain = st.checkbox("Chest Pain")
            short_breath = st.checkbox("Shortness of Breath")
            vomiting = st.checkbox("Vomiting")

        with s2:
            dizziness = st.checkbox("Dizziness")
            swelling = st.checkbox("Swelling")
            fatigue = st.checkbox("Fatigue")
            blurred_vision = st.checkbox("Blurred Vision")

        if fever:
            selected_symptoms.append("fever")
        if chest_pain:
            selected_symptoms.append("chest pain")
        if short_breath:
            selected_symptoms.append("shortness of breath")
        if vomiting:
            selected_symptoms.append("vomiting")
        if dizziness:
            selected_symptoms.append("dizziness")
        if swelling:
            selected_symptoms.append("swelling")
        if fatigue:
            selected_symptoms.append("fatigue")
        if blurred_vision:
            selected_symptoms.append("blurred vision")

        st.markdown('<div class="section-title">Vital Signs</div>', unsafe_allow_html=True)

        v1, v2, v3, v4 = st.columns(4)

        with v1:
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=30.0,
                max_value=45.0,
                value=37.0,
                step=0.1
            )

        with v2:
            heart_rate = st.number_input(
                "Heart Rate (bpm)",
                min_value=40,
                max_value=220,
                value=80,
                step=1
            )

        with v3:
            spo2 = st.number_input(
                "Oxygen Saturation (SpO2 %)",
                min_value=50,
                max_value=100,
                value=98,
                step=1
            )

        with v4:
            systolic_bp = st.number_input(
                "Systolic Blood Pressure (mmHg)",
                min_value=60,
                max_value=260,
                value=120,
                step=1
            )

        if st.button("Generate Risk Assessment"):

            if not selected_symptoms:
                st.warning("Please select at least one symptom.")

            else:
                X_input, symptom_count = build_input_features(
                    age,
                    gender,
                    selected_symptoms
                )

                prediction = best_model.predict(X_input)[0]
                probabilities = best_model.predict_proba(X_input)[0]
                high_risk_probability = float(probabilities[1]) * 100

                if high_risk_probability < 40:
                    level = "Low"
                    risk_color = "low"
                elif high_risk_probability < 70:
                    level = "Medium"
                    risk_color = "medium"
                else:
                    level = "High"
                    risk_color = "high"

                escalation_reason = None

                if (
                    spo2 < 92
                    or systolic_bp >= 180
                    or temperature >= 39.0
                    or heart_rate >= 130
                ):
                    if level != "High":
                        level = "High"
                        risk_color = "high"
                        escalation_reason = "Risk was escalated to High due to critical vital signs."

                symptom_text = ", ".join(selected_symptoms)

                r1, r2 = st.columns(2)

                with r1:
                    render_metric_card(
                        "Risk Level",
                        level,
                        "Final clinical risk level",
                        risk_color
                    )

                with r2:
                    render_metric_card(
                        "High Risk Probability",
                        f"{high_risk_probability:.1f}%",
                        "Predicted probability",
                        risk_color
                    )

                if developer_mode:
                    render_metric_card(
                        "Best Model Used",
                        best_model_name,
                        "Visible only in Developer Mode",
                        "purple"
                    )

                st.markdown(f"""
                <div class="card">
                    <div class="card-title">Assessment Summary</div>
                    <div class="muted">
                        Final result generated using machine learning prediction and clinical safety rules.
                    </div>
                    <hr>
                    <p><b>Model Prediction Class:</b> {"High" if prediction == 1 else "Low"}</p>
                    <p><b>Age:</b> {age}</p>
                    <p><b>Gender:</b> {gender}</p>
                    <p><b>Symptom Count:</b> {symptom_count}</p>
                    <p><b>Symptoms Entered:</b> {symptom_text}</p>
                    <p><b>Temperature:</b> {temperature} °C</p>
                    <p><b>Heart Rate:</b> {heart_rate} bpm</p>
                    <p><b>SpO2:</b> {spo2}%</p>
                    <p><b>Systolic BP:</b> {systolic_bp} mmHg</p>
                </div>
                """, unsafe_allow_html=True)

                if escalation_reason:
                    st.markdown(f"""
                    <div class="alert-card">
                        <b>Clinical Safety Escalation</b><br>
                        {escalation_reason}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(
                    generate_risk_actions(level),
                    unsafe_allow_html=True
                )

        if developer_mode:
            st.markdown(
                '<div class="section-title">Model Evaluation and Comparison</div>',
                unsafe_allow_html=True
            )

            metrics_rows = []

            for model_name, result in model_results.items():
                high_recall = result["report"]["1"]["recall"] if "1" in result["report"] else 0
                high_precision = result["report"]["1"]["precision"] if "1" in result["report"] else 0

                metrics_rows.append({
                    "Model": model_name,
                    "Accuracy": round(result["accuracy"], 4),
                    "High Risk Recall": round(high_recall, 4),
                    "High Risk Precision": round(high_precision, 4)
                })

            metrics_df = pd.DataFrame(metrics_rows).sort_values(
                "Accuracy",
                ascending=False
            )

            st.dataframe(metrics_df, use_container_width=True)
            st.success(f"Best performing model: {best_model_name}")

            selected_model_report = st.selectbox(
                "View Model Evaluation Report",
                list(model_results.keys())
            )

            report_df = pd.DataFrame(
                model_results[selected_model_report]["report"]
            ).transpose()

            st.dataframe(report_df, use_container_width=True)

            st.markdown(
                '<div class="section-title">Risk Dataset Preview</div>',
                unsafe_allow_html=True
            )

            st.dataframe(df_risk.head(10), use_container_width=True)

# =========================
# TAB 3: ICU PATIENT MONITORING
# =========================
if role in ["Nurse", "Doctor", "Developer"]:
    with tab3:

        st.markdown("""
        <div class="card">
            <div class="card-title">ICU Patient Monitoring</div>
            <div class="muted">
                Monitor ICU patient status, detect early deterioration, and identify medication timing issues.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not os.path.exists(VITALS_PATH):
            st.error("vitals_demo.csv not found. Please place it in the project folder.")
            st.stop()

        try:
            vitals = pd.read_csv(VITALS_PATH)
        except Exception as e:
            st.error(f"Failed to load vitals data: {e}")
            st.stop()

        required_cols = [
            "Patient_ID",
            "Timestamp",
            "HR",
            "Systolic_BP",
            "SpO2",
            "Temp",
            "Resp_Rate",
            "Last_Med_Time",
            "Next_Med_Time"
        ]

        missing = [c for c in required_cols if c not in vitals.columns]

        if missing:
            st.error(f"Missing columns in vitals_demo.csv: {missing}")
            st.stop()

        vitals["Timestamp"] = pd.to_datetime(vitals["Timestamp"], errors="coerce")
        vitals["Last_Med_Time"] = pd.to_datetime(vitals["Last_Med_Time"], errors="coerce")
        vitals["Next_Med_Time"] = pd.to_datetime(vitals["Next_Med_Time"], errors="coerce")

        vitals = vitals.dropna(subset=["Timestamp"]).copy()

        if vitals.empty:
            st.error("No valid ICU monitoring timestamps found.")
            st.stop()

        min_ts = vitals["Timestamp"].min()
        max_ts = vitals["Timestamp"].max()

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Available Monitoring Period</div>
            <div class="muted">{min_ts.date()} to {max_ts.date()}</div>
        </div>
        """, unsafe_allow_html=True)

        selected_day = st.date_input(
            "Select monitoring day",
            value=min_ts.date(),
            min_value=min_ts.date(),
            max_value=max_ts.date(),
            key="icu_day"
        )

        day_data = vitals[vitals["Timestamp"].dt.date == selected_day].copy()

        if day_data.empty:
            st.warning("No monitoring data available for the selected day.")
            st.stop()

        latest = (
            day_data
            .sort_values("Timestamp")
            .groupby("Patient_ID")
            .tail(1)
            .copy()
        )

        def patient_status(row):
            critical = (
                (row["SpO2"] < 90)
                or (row["HR"] > 130)
                or (row["Systolic_BP"] > 180)
                or (row["Temp"] > 39.0)
                or (row["Resp_Rate"] > 26)
            )

            warning = (
                (row["SpO2"] < 94)
                or (row["HR"] > 110)
                or (row["Systolic_BP"] > 160)
                or (row["Temp"] > 38.0)
                or (row["Resp_Rate"] > 22)
            )

            if critical:
                return "Critical"
            elif warning:
                return "Warning"
            else:
                return "Stable"

        latest["Status"] = latest.apply(patient_status, axis=1)

        critical_patients = latest[latest["Status"] == "Critical"]
        warning_patients = latest[latest["Status"] == "Warning"]
        stable_patients = latest[latest["Status"] == "Stable"]

        now_ref = day_data["Timestamp"].max()

        overdue_patients = []
        due_soon_patients = []

        for _, row in latest.iterrows():
            next_med = row["Next_Med_Time"]

            if pd.notna(next_med):
                diff_minutes = int((next_med - now_ref).total_seconds() / 60)

                if diff_minutes < 0:
                    overdue_patients.append((row, abs(diff_minutes)))
                elif 0 <= diff_minutes <= 30:
                    due_soon_patients.append((row, diff_minutes))

        st.markdown('<div class="section-title">ICU Overview</div>', unsafe_allow_html=True)

        i1, i2, i3, i4 = st.columns(4)

        with i1:
            render_metric_card(
                "Total ICU Patients",
                len(latest),
                "Current monitored patients",
                "blue"
            )

        with i2:
            render_metric_card(
                "Critical Cases",
                len(critical_patients),
                "Require urgent review",
                "high"
            )

        with i3:
            render_metric_card(
                "Warning Cases",
                len(warning_patients),
                "Need close monitoring",
                "medium"
            )

        with i4:
            render_metric_card(
                "Stable Cases",
                len(stable_patients),
                "Normal condition",
                "low"
            )

        st.markdown('<div class="section-title">Current Patient Status</div>', unsafe_allow_html=True)

        st.dataframe(
            latest[[
                "Patient_ID",
                "Timestamp",
                "HR",
                "Systolic_BP",
                "SpO2",
                "Temp",
                "Resp_Rate",
                "Status"
            ]],
            use_container_width=True
        )

        st.markdown('<div class="section-title">Clinical Alerts</div>', unsafe_allow_html=True)

        if critical_patients.empty and warning_patients.empty:
            st.markdown("""
            <div class="ok-card">
                <b>No active clinical alerts</b><br>
                Current patients are stable based on the latest available readings.
            </div>
            """, unsafe_allow_html=True)

        else:
            for _, row in critical_patients.iterrows():
                st.markdown(f"""
                <div class="alert-card">
                    <b>Critical Alert:</b> Patient {row['Patient_ID']}<br>
                    HR: {row['HR']} bpm |
                    SpO2: {row['SpO2']}% |
                    SBP: {row['Systolic_BP']} mmHg |
                    Temp: {row['Temp']}°C |
                    RR: {row['Resp_Rate']} /min
                </div>
                """, unsafe_allow_html=True)

            for _, row in warning_patients.iterrows():
                st.markdown(f"""
                <div class="warn-card">
                    <b>Warning Alert:</b> Patient {row['Patient_ID']}<br>
                    HR: {row['HR']} bpm |
                    SpO2: {row['SpO2']}% |
                    SBP: {row['Systolic_BP']} mmHg |
                    Temp: {row['Temp']}°C |
                    RR: {row['Resp_Rate']} /min
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Medication Administration Alerts</div>', unsafe_allow_html=True)

        if not overdue_patients and not due_soon_patients:
            st.markdown("""
            <div class="ok-card">
                <b>No medication timing issues detected.</b>
            </div>
            """, unsafe_allow_html=True)

        else:
            for row, mins in overdue_patients:
                st.markdown(f"""
                <div class="alert-card">
                    <b>Medication Overdue:</b> Patient {row['Patient_ID']}<br>
                    Overdue by {mins} minutes |
                    Scheduled time: {row['Next_Med_Time']}
                </div>
                """, unsafe_allow_html=True)

            for row, mins in due_soon_patients:
                st.markdown(f"""
                <div class="warn-card">
                    <b>Medication Due Soon:</b> Patient {row['Patient_ID']}<br>
                    Due in {mins} minutes |
                    Scheduled time: {row['Next_Med_Time']}
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">SBAR Auto Note</div>', unsafe_allow_html=True)

        patient_ids = latest["Patient_ID"].astype(str).tolist()

        selected_patient = st.selectbox(
            "Select patient for SBAR note",
            patient_ids
        )

        patient_row = latest[
            latest["Patient_ID"].astype(str) == selected_patient
        ].iloc[0]

        med_alert_text = "No medication timing issue detected."

        for row, mins in overdue_patients:
            if str(row["Patient_ID"]) == selected_patient:
                med_alert_text = f"Medication overdue by {mins} minutes."

        for row, mins in due_soon_patients:
            if str(row["Patient_ID"]) == selected_patient:
                med_alert_text = f"Medication due in {mins} minutes."

        situation = f"Patient {selected_patient} is currently classified as {patient_row['Status']}."

        background = (
            f"Latest observation recorded at {patient_row['Timestamp']}. "
            f"Medication status: {med_alert_text}"
        )

        assessment = (
            f"Current vital signs: HR {patient_row['HR']} bpm, "
            f"SBP {patient_row['Systolic_BP']} mmHg, "
            f"SpO2 {patient_row['SpO2']}%, "
            f"Temperature {patient_row['Temp']}°C, "
            f"Respiratory Rate {patient_row['Resp_Rate']} /min."
        )

        if patient_row["Status"] == "Critical":
            recommendation_text = (
                "Immediate physician review is recommended. "
                "Repeat vital signs within 5 minutes and prepare escalation if instability persists."
            )

        elif patient_row["Status"] == "Warning":
            recommendation_text = (
                "Repeat vital signs within 15 minutes and continue close monitoring. "
                "Escalate if condition worsens."
            )

        else:
            recommendation_text = (
                "Continue routine monitoring and follow standard ICU protocol."
            )

        sbar_note = f"""
S - Situation:
{situation}

B - Background:
{background}

A - Assessment:
{assessment}

R - Recommendation:
{recommendation_text}
        """.strip()

        st.text_area(
            "SBAR Clinical Note",
            sbar_note,
            height=260
        )

        st.markdown('<div class="section-title">Recommended Actions</div>', unsafe_allow_html=True)

        if (
            critical_patients.empty
            and warning_patients.empty
            and not overdue_patients
            and not due_soon_patients
        ):
            st.markdown("""
            <div class="ok-card">
                <b>General Clinical Action</b><br>
                <ul>
                    <li>Continue routine ICU monitoring.</li>
                    <li>Maintain scheduled medication administration.</li>
                    <li>Reassess according to standard clinical protocol.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        else:
            if not critical_patients.empty:
                st.markdown("""
                <div class="alert-card">
                    <b>Critical Patient Actions</b><br>
                    <ul>
                        <li>Notify physician immediately.</li>
                        <li>Repeat vital signs within 5 minutes.</li>
                        <li>Prepare oxygen support and verify airway status.</li>
                        <li>Review medication schedule and confirm urgent administration.</li>
                        <li>Consider escalation if instability persists.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            if not warning_patients.empty:
                st.markdown("""
                <div class="warn-card">
                    <b>Warning Patient Actions</b><br>
                    <ul>
                        <li>Repeat vital signs within 15 minutes.</li>
                        <li>Monitor patient trend closely.</li>
                        <li>Review recent medications and nursing notes.</li>
                        <li>Escalate if deterioration continues.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            if overdue_patients:
                st.markdown("""
                <div class="alert-card">
                    <b>Overdue Medication Actions</b><br>
                    <ul>
                        <li>Confirm whether the scheduled dose was administered.</li>
                        <li>Notify the responsible nurse immediately if the dose was missed.</li>
                        <li>Document the delay and reassess patient condition.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            if due_soon_patients:
                st.markdown("""
                <div class="warn-card">
                    <b>Due Soon Medication Actions</b><br>
                    <ul>
                        <li>Prepare the upcoming medication dose in advance.</li>
                        <li>Ensure administration occurs on schedule.</li>
                        <li>Continue monitoring the patient after administration.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)