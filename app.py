import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("streetlight_energy_model.pkl")

# Load feature names
model_features = joblib.load("model_features.pkl")

st.title("💡 Streetlight Energy Optimizer")

st.write(
    "Predict streetlight energy consumption and get a lighting recommendation."
)

# User inputs
hour = st.number_input(
    "Hour of day",
    min_value=0,
    max_value=23,
    value=20
)

day = st.number_input(
    "Day of month",
    min_value=1,
    max_value=31,
    value=15
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=9
)

day_of_week = st.number_input(
    "Day of week (0=Monday, 6=Sunday)",
    min_value=0,
    max_value=6,
    value=1
)

is_weekend = 1 if day_of_week >= 5 else 0

device_id = st.text_input(
    "Device ID",
    value="a126ebe4-2cca-4cbb-953e-71eaca8ac837"
)

# Cyclic features
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

# Prediction button
if st.button("Predict Energy Consumption"):

    new_data = pd.DataFrame({
        "device_id": [device_id],
        "hour": [hour],
        "day": [day],
        "month": [month],
        "day_of_week": [day_of_week],
        "is_weekend": [is_weekend],
        "hour_sin": [hour_sin],
        "hour_cos": [hour_cos],
        "month_sin": [month_sin],
        "month_cos": [month_cos]
    })

    # One-hot encode device
    new_data = pd.get_dummies(
        new_data,
        columns=["device_id"],
        dtype=int
    )

    # Match training features
    new_data = new_data.reindex(
        columns=model_features,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(new_data)[0]

    st.subheader("Prediction")

    st.metric(
        "Predicted Energy Consumption",
        f"{prediction:.2f} kWh"
    )

    # Recommendation
    if prediction > 1.0:
        st.warning("⚠️ High energy consumption — Consider Reduce/DIM lighting.")
    else:
        st.success("✅ Normal energy consumption — Normal lighting.")