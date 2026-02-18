import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Attrition Risk Dashboard")
st.markdown("Upload employee data to predict attrition risk.")

# -------------------------------------------------
# Load Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_model.pkl")

model = load_model()

# Get exact feature count expected by model
expected_features = model.n_features_in_

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

threshold = st.sidebar.slider(
    "Risk Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

# -------------------------------------------------
# Main Logic
# -------------------------------------------------
if uploaded_file is not None:

    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📁 Uploaded Data Preview")
    st.dataframe(df.head())

    # -------------------------------------------------
    # Convert all columns to numeric safely
    # -------------------------------------------------
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # -------------------------------------------------
    # Feature Shape Correction (CRITICAL FIX)
    # -------------------------------------------------
    current_features = df.shape[1]

    if current_features < expected_features:
        missing_count = expected_features - current_features

        st.warning(
            f"Model expects {expected_features} features but received {current_features}. "
            f"Adding {missing_count} default columns automatically."
        )

        for i in range(missing_count):
            df[f"dummy_{i}"] = 0

    elif current_features > expected_features:
        extra_count = current_features - expected_features

        st.info(
            f"Model expects {expected_features} features. "
            f"Ignoring extra {extra_count} columns."
        )

        df = df.iloc[:, :expected_features]

    # Final safe dataframe
    df_model = df.iloc[:, :expected_features]

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    try:
        probabilities = model.predict_proba(df_model)[:, 1]

        df["Attrition_Risk_Probability"] = probabilities
        df["Risk_Level"] = np.where(
            df["Attrition_Risk_Probability"] >= threshold,
            "High Risk",
            "Low Risk"
        )

        # -------------------------------------------------
        # KPI Metrics
        # -------------------------------------------------
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Employees", len(df))
        col2.metric("High Risk Employees",
                    (df["Risk_Level"] == "High Risk").sum())
        col3.metric("Average Risk %",
                    f"{df['Attrition_Risk_Probability'].mean():.2f}")

        # -------------------------------------------------
        # Charts
        # -------------------------------------------------
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("📊 Risk Distribution")
            fig, ax = plt.subplots()
            df["Risk_Level"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

        with chart_col2:
            st.subheader("📈 Probability Distribution")
            fig2, ax2 = plt.subplots()
            sns.histplot(df["Attrition_Risk_Probability"], bins=10)
            st.pyplot(fig2)

        # -------------------------------------------------
        # Feature Importance
        # -------------------------------------------------
        if hasattr(model, "feature_importances_"):
            st.subheader("🔥 Feature Importance")

            importance_df = pd.DataFrame({
                "Feature": [f"Feature_{i}" for i in range(expected_features)],
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.barplot(
                x="Importance",
                y="Feature",
                data=importance_df.head(10)
            )
            st.pyplot(fig3)

        # -------------------------------------------------
        # Results Table
        # -------------------------------------------------
        st.subheader("🏆 Prediction Results")
        st.dataframe(df)

        # -------------------------------------------------
        # Download Button
        # -------------------------------------------------
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Predictions",
            data=csv,
            file_name="attrition_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

else:
    st.info("👈 Upload a CSV or Excel file from the sidebar to begin.")
