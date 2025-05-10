import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model and feature names
model = joblib.load("E:/Credit_risk_scoring_xai/model2.pkl")
feature_names = joblib.load("E:/Credit_risk_scoring_xai/features3.pkl")  # Ensure this matches the model features

# Selected top features used in training
top_features = [
    'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'EXT_SOURCE_1',
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_EMPLOYED', 'REGION_POPULATION_RELATIVE'
]

# Page title
st.title("Credit Risk Scoring App")

# Sidebar inputs
st.sidebar.header("Enter Loan Application Details")
user_input = {}

# Default values to guide user input
default_values = {
    'AMT_CREDIT': 500000.0,
    'AMT_INCOME_TOTAL': 150000.0,
    'EXT_SOURCE_1': 0.5,
    'EXT_SOURCE_2': 0.5,
    'EXT_SOURCE_3': 0.5,
    'DAYS_EMPLOYED': -1000.0,  # negative in dataset means currently employed
    'REGION_POPULATION_RELATIVE': 0.01
}

# Sidebar number inputs for each top feature
for feature in top_features:
    user_input[feature] = st.sidebar.number_input(
        f"{feature}", value=default_values[feature]
    )

# Convert user input to DataFrame with correct feature order
input_df = pd.DataFrame([user_input])[top_features]

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # Probability of class 1

    st.subheader("Prediction Result")
    st.write(f"### {'✅ Loan Approved' if prediction == 1 else '❌ Loan Denied'}")
    st.write(f"**Probability of approval: {probability * 100:.2f}%**")

    # Feature importance display (if supported)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_imp_df = pd.DataFrame({
            "Feature": top_features,
            "Importance": importances
        })
        feature_imp_df["Importance (%)"] = 100 * feature_imp_df["Importance"] / feature_imp_df["Importance"].sum()
        feature_imp_df = feature_imp_df.sort_values(by="Importance (%)", ascending=True)

        st.subheader("Feature Impact on Loan Decision")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(feature_imp_df["Feature"], feature_imp_df["Importance (%)"], color='skyblue')
        ax.set_xlabel("Importance (%)")
        ax.set_title("Model Feature Importances")
        st.pyplot(fig)

        # Show as table too
        st.dataframe(feature_imp_df.sort_values(by="Importance (%)", ascending=False).reset_index(drop=True))
    else:
        st.info("This model does not support feature importance")
