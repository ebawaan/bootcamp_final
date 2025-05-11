import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
rf = joblib.load('rf_tour_winner_model.joblib')
scaler = joblib.load('scaler_tour_winner.joblib')

st.title('Tour de France Rider Score Predictor')
st.write('Adjust the sliders to see the predicted rider score (0-100) based on ability to win the Tour de France.')

# Define feature ranges (these should match the training data)
HEIGHT_MIN, HEIGHT_MAX = 1.60, 2.00  # meters
WEIGHT_MIN, WEIGHT_MAX = 55, 85      # kg
AGE_MIN, AGE_MAX = 20, 40            # years
BMI_MIN, BMI_MAX = 18.0, 28.0        # BMI as floats

height = st.slider('Height (m)', HEIGHT_MIN, HEIGHT_MAX, 1.75, 0.01)
weight = st.slider('Weight (kg)', WEIGHT_MIN, WEIGHT_MAX, 70, 1)
age = st.slider('Age', AGE_MIN, AGE_MAX, 28, 1)
bmi = st.slider('BMI', BMI_MIN, BMI_MAX, 22.0, 0.1)

# Prepare input for model
input_features = np.array([[height, weight, age, bmi]])
input_scaled = scaler.transform(input_features)

# Predict and scale to 0-100
score = rf.predict(input_scaled)[0]
score = np.clip(score, 0, 1) * 100

st.metric('Rider Score', f'{score:.1f} / 100')