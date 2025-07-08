# 🚀 Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re  # Import regex for special character handling

# 🎯 Load pre-trained model and column structure
model = joblib.load("pollution_model.pkl")
expected_columns = joblib.load("model_columns.pkl")

# 🎨 Streamlit App UI
st.set_page_config(page_title="Water Quality Estimator", layout="centered")
st.title("💧 Water Pollution Forecasting Tool")
st.markdown("This app predicts the expected concentration of various pollutants based on the selected year and monitoring station.")

# 📅 User Input - Year
selected_year = st.number_input("Select Year", min_value=2000, max_value=2100, value=2023)

# 🏢 User Input - Station ID
station_input = st.text_input("Enter Monitoring Station ID", value="1")

# 🚀 Trigger Prediction
if st.button("🔍 Predict Pollutant Levels"):

    if not station_input.strip():
        st.warning("⚠️ Please enter a valid Station ID.")
    else:
        # Clean station ID: strip + replace special characters
        clean_station = re.sub(r'[^a-zA-Z0-9]', '_', station_input.strip())
        
        # Prepare input as DataFrame
        input_data = pd.DataFrame({'year': [selected_year], 'id': [clean_station]})

        # One-hot encode Station ID
        encoded_input = pd.get_dummies(input_data, columns=['id'])

        # Ensure all expected columns are present
        for col in expected_columns:
            if col not in encoded_input.columns:
                encoded_input[col] = 0

        # Reorder columns to match training data
        encoded_input = encoded_input[expected_columns]

        # 🔮 Predict pollutant values
        predictions = model.predict(encoded_input)[0]

        # List of pollutants being predicted
        pollutant_names = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

        # 🧪 Display results
        st.subheader(f"📊 Estimated Pollutant Levels for Station `{station_input}` in {selected_year}")
        for name, value in zip(pollutant_names, predictions):
            st.success(f"**{name}**: {value:.2f} mg/L")
