import streamlit as st
import joblib
import numpy as np

model = joblib.load("sales_forecast_model.pkl")

st.title("Sales Forecast Dashboard")

year = st.selectbox("Select Year", [2026, 2027])
month = st.slider("Select Month", 1, 12)

quarter = (month - 1) // 3 + 1
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

prediction = model.predict([[year, month, quarter, month_sin, month_cos]])

st.metric("Predicted Revenue", f"{prediction[0]:,.2f}")
