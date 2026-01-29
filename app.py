import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(layout="wide")

# Load data
monthly = pd.read_csv("monthly_sales.csv")
product_df = pd.read_csv("monthly_product_sales.csv", header=None,
    names=['Product','Year','Month','Units','Revenue'])

model = joblib.load("sales_forecast_model.pkl")

st.title("📊 Sales Analytics & Forecast Dashboard")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"{monthly['Revenue'].sum():,.0f}")
col2.metric("Total Units", f"{monthly['Units'].sum():,.0f}")
col3.metric("Avg Monthly Sales", f"{monthly['Revenue'].mean():,.0f}")

# Charts
st.subheader("Monthly Revenue Trend")
st.line_chart(monthly['Revenue'])

st.subheader("Top Products")
top_products = product_df.groupby('Product')['Revenue'].sum().nlargest(10)
st.bar_chart(top_products)

st.subheader("Seasonality")
season = product_df.groupby('Month')['Revenue'].mean()
st.bar_chart(season)

st.subheader("Revenue Distribution")
st.area_chart(product_df['Revenue'])

# Forecast
st.subheader("🔮 Sales Forecast")
year = st.selectbox("Year", [2026, 2027])
month = st.slider("Month", 1, 12)

quarter = (month - 1)//3 + 1
month_sin = np.sin(2*np.pi*month/12)
month_cos = np.cos(2*np.pi*month/12)

prediction = model.predict([[year, month, quarter, month_sin, month_cos]])
st.metric("Predicted Revenue", f"{prediction[0]:,.0f}")
