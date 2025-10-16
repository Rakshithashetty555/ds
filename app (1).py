import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ks_2samp
import numpy as np

st.set_page_config(page_title="Responsible AI Dashboard (Regression)", layout="wide")
st.title("🤖 Regression Dashboard")
st.write("Upload dataset, train a regression model, view metrics, SHAP explainability, and data drift.")

# -------------------------------
# Step 1: Upload CSV
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Dataset preview:")
    st.dataframe(df.head())

    df.columns = df.columns.str.strip()
    
    # -------------------------------
    # Step 2: Select target
    # -------------------------------
    target = st.selectbox("🎯 Select target column", df.columns)
    features = [c for c in df.columns if c != target]
    
    # -------------------------------
    # Step 3: Encode categorical features
    # -------------------------------
    X = pd.get_dummies(df[features], drop_first=True)
    y = df[target]
    st.info(f"Features auto-encoded. Shape: {X.shape}")
    
    # -------------------------------
    # Step 4: Train/Test split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    # -------------------------------
    # Step 5: Train RandomForest Regressor
    # -------------------------------
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # -------------------------------
    # Step 6: Regression Metrics
    # -------------------------------
    st.subheader("📊 Model Metrics")
    st.metric("R² Score", f"{r2_score(y_test, preds):.2f}")
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, preds)):.2f}")
    st.metric("MAE", f"{mean_absolute_error(y_test, preds):.2f}")
    
    # -------------------------------
    # Step 7: SHAP Explainability
    # -------------------------------
    st.subheader("🌈 SHAP Feature Importance")
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)
    
    # -------------------------------
    # Step 8: Data Drift (KS Test)
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
