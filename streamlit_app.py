import streamlit as st
import requests
import numpy as np

st.set_page_config(page_title="Fraud Detection UI")

st.title("Credit Card Fraud Detection")

st.write("Enter transaction features (43 values)")

# Generate input
features = st.text_area(
    "Enter 43 comma-separated values",
    "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,100"
)

if st.button("Check Fraud"):

    try:
        feature_list = [float(x.strip()) for x in features.split(",")]

        if len(feature_list) != 43:
            st.error("You must enter exactly 43 values")
        else:
            url = "http://127.0.0.1:8000/predict"

            response = requests.post(url, json={"features": feature_list})

            if response.status_code == 200:
                result = response.json()

                st.success("Prediction Result")

                st.write("Prediction:", result["label"])
                st.write("Fraud Probability:", result["fraud_probability"])
                st.write("Risk Band:", result["risk_band"])
                st.write("Recommended Action:", result["recommended_action"])
                st.write("Explanation:", result["explanation"])

            else:
                st.error("API Error")

    except:
        st.error("Invalid input format")