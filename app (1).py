%%writefile app.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ks_2samp

st.set_page_config(page_title="Responsible AI Dashboard", layout="wide")

st.title("🤖 Responsible AI Dashboard")
st.write("Includes SHAP Explainability, Model Metrics, and Drift Checks")

uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Data preview:")
    st.dataframe(df.head())

    # Select target
    target = st.selectbox("🎯 Select target column", df.columns)
    features = [c for c in df.columns if c != target]

    # Convert categorical columns to numeric
    df_encoded = pd.get_dummies(df, drop_first=True)
    st.info(f"Data automatically encoded. New shape: {df_encoded.shape}")

    # Adjust features after encoding
    new_features = [c for c in df_encoded.columns if c != target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df_encoded[new_features], df_encoded[target], test_size=0.3, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # -------------------------------
    # Metrics
    # -------------------------------
    st.subheader("📊 Model Metrics")
    acc = accuracy_score(y_test, preds)
    st.metric("Accuracy", f"{acc:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, preds))

    # -------------------------------
    # SHAP Explainability
    # -------------------------------
    st.subheader("🌈 SHAP Feature Importance")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)

    # -------------------------------
    # Drift Check (KS Test)
    # -------------------------------
    st.subheader("📉 Data Drift Check (KS Test)")
    drift_results = {}
    for col in new_features:
        stat, p = ks_2samp(X_train[col], X_test[col])
        drift_results[col] = p
    drift_df = pd.DataFrame.from_dict(drift_results, orient="index", columns=["p_value"])
    drift_df["Drift_Flag"] = drift_df["p_value"] < 0.05
    st.write("✅ Lower p-value (<0.05) indicates drift:")
    st.dataframe(drift_df)
