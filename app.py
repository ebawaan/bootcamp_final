import streamlit as st
import numpy as np
import joblib
import shutil

# Load model and scaler
model = joblib.load('winner_classifier.joblib')
scaler = joblib.load('scaler.joblib')

st.title("Tour de France Winner Probability Predictor")

st.write("Adjust the values below to see the predicted probability that a rider with these measurements would win:")

height = st.number_input("Height (m)", min_value=1.5, max_value=2.0, value=1.75, step=0.01)
weight = st.number_input("Weight (Kg)", min_value=50.0, max_value=90.0, value=65.0, step=0.1)
age = st.number_input("Age", min_value=18, max_value=40, value=25, step=1)
bmi = st.number_input("BMI", min_value=16.0, max_value=25.0, value=20.0, step=0.1)

input_features = np.array([[height, weight, age, bmi]])
input_scaled = scaler.transform(input_features)
prob = model.predict_proba(input_scaled)[0, 1]

st.subheader("Predicted Probability of Winning")
st.write(f"{prob * 100:.1f}%")