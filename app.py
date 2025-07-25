import streamlit as st
import pandas as pd
import joblib
import gzip

# Load the model
with gzip.open('model_small.pkl.gz', 'rb') as f:
    model = joblib.load(f)

st.title("🩺 Medical Appointment No-show Predictor")

st.write("Fill in the patient details below:")

# Patient input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
neighbourhood = st.number_input("Neighbourhood (as numeric ID)", min_value=0)
scholarship = st.selectbox("Scholarship", ["No", "Yes"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
alcoholism = st.selectbox("Alcoholism", ["No", "Yes"])
handcap = st.selectbox("Handcap (0 = None, up to 4 = High)", [0, 1, 2, 3, 4])
sms = st.selectbox("SMS Received", ["No", "Yes"])
waiting_days = st.slider("Days Waiting for Appointment", 0, 100, 10)

# Format input
input_data = [[
    1 if gender == "Male" else 0,
    age,
    int(neighbourhood),
    1 if scholarship == "Yes" else 0,
    1 if hypertension == "Yes" else 0,
    1 if diabetes == "Yes" else 0,
    1 if alcoholism == "Yes" else 0,
    handcap,
    1 if sms == "Yes" else 0,
    waiting_days
]]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("🚫 The patient is likely to MISS the appointment.")
    else:
        st.success("✅ The patient is likely to SHOW UP.")
