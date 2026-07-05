import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load Model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load Dataset
df = pd.read_csv("house_data.csv")

# Title
st.title("🏠 House Price Prediction")
st.write("Predict the price of a house using Machine Learning.")

st.markdown("---")

# Input Section
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=500, max_value=5000, value=1000)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

with col2:
    age = st.number_input("Age (Years)", min_value=0, max_value=50, value=5)
    parking = st.number_input("Parking", min_value=0, max_value=5, value=2)

st.markdown("---")

# Prediction
if st.button("🔮 Predict Price", use_container_width=True):

    data = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Age": [age],
        "Parking": [parking]
    })

    prediction = model.predict(data)

    st.success(f"🏷️ Predicted House Price: ₹ {prediction[0]:,.0f}")

    st.balloons()

st.markdown("---")

# Dataset Preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")

# Charts
st.subheader("📈 House Price Distribution")
st.bar_chart(df["Price"])

st.subheader("🏠 Area Distribution")
st.line_chart(df["Area"])

st.markdown("---")

# Statistics
st.subheader("📋 Dataset Statistics")
st.write(df.describe())

st.markdown("---")

# Sidebar
st.sidebar.title("🏠 House Price Prediction")

st.sidebar.write("""
This project predicts house prices using a Machine Learning model.

### Features
- Predict House Price
- Dataset Preview
- Charts
- Statistics

### Developed By
Rajesh G R
ECE Student
""")