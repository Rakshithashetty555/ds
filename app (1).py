# Save as app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ks_2samp

st.set_page_config(page_title="Rating Regression Dashboard", layout="wide")
st.title("🤖 Rating Regression Dashboard")
st.write("Upload dataset, predict ratings, view metrics, SHAP explanations, and data drift checks.")

# -------------------------------
# Step 1: Upload CSV
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Dataset preview:")
    st.dataframe(df.head())

    # Clean column names
    df.columns = df.columns.str.strip()

    # -------------------------------
    # Step 2: Select target
    # -------------------------------
    target = st.selectbox(
        "🎯 Select target column",
        df.columns,
        index=list(df.columns).index('rating') if 'rating' in df.columns else 0
    )
    features = [c for c in df.columns if c != target]

    # -------------------------------
    # Step 3: Convert target to numeric if needed
    # -------------------------------
    if not np.issubdtype(df[target].dtype, np.number):
        st.warning(f"Target column '{target}' is not numeric. Converting to numeric.")
        df[target] = df[target].astype(str).str.extract('(\d+)').astype(float)
    y = df[target]

    # -------------------------------
    # Step 4: Encode features
    # -------------------------------
    X = pd.get_dummies(df[features], drop_first=True)
    st.info(f"Features auto-encoded. Shape: {X.shape}")

    # -------------------------------
    # Step 5: Train/Test split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # -------------------------------
    # Step 6: Train RandomForest Regressor
    # -------------------------------
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # -------------------------------
    # Step 7: Predictions Section
    # -------------------------------
    st.subheader("🖥️ Predictions")
    pred_df = X_test.copy()
    pred_df["Actual"] = y_test
    pred_df["Predicted"] = preds
    st.dataframe(pred_df.head())

    # -------------------------------
    # Step 8: Metrics Section
    # -------------------------------
    st.subheader("📊 Model Metrics")
    st.metric("R² Score", f"{r2_score(y_test, preds):.2f}")
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, preds)):.2f}")
    st.metric("MAE", f"{mean_absolute_error(y_test, preds):.2f}")

    # -------------------------------
    # Step 9: SHAP Explanations
    # -------------------------------
    st.subheader("🌈 SHAP Explanations")
    try:
        # Sample for robust SHAP
        shap_sample_X = X_test.sample(min(50, len(X_test)), random_state=42)
        shap_background = X_train.sample(min(50, len(X_train)), random_state=42)

        explainer = shap.KernelExplainer(model.predict, shap_background)
        shap_values = explainer.shap_values(shap_sample_X, nsamples=100)

        # Beeswarm Plot
        st.write("**Beeswarm Plot (Local Feature Effects)**")
        fig, ax = plt.subplots(figsize=(8,5))
        shap.summary_plot(shap_values, shap_sample_X, plot_type="dot", show=False)
        st.pyplot(fig)

        # Global Feature Importance
        st.write("**Global Feature Importance (Bar Chart)**")
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "Feature": shap_sample_X.columns,
            "Importance": mean_abs_shap
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8,5))
        ax.barh(feature_importance["Feature"], feature_importance["Importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Global Feature Importance")
        st.pyplot(fig)

    except Exception as e:
        st.warning("⚠️ SHAP could not be computed.")
        st.write(e)

    # -------------------------------
    # Step 10: Data Drift Checks
    # -------------------------------
    st.subheader("📉 Data Drift Check (KS Test)")
    drift_results = {}
    for col in X_train.columns:
        stat, p = ks_2samp(X_train[col], X_test[col])
        drift_results[col] = p
    drift_df = pd.DataFrame.from_dict(drift_results, orient="index", columns=["p_value"])
    drift_df["Drift_Flag"] = drift_df["p_value"] < 0.05
    st.write("✅ Lower p-value (<0.05) indicates drift:")
    st.dataframe(drift_df)
