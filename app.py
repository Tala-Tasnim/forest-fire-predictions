import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('regression_nn_model.keras')

# Define possible values for month and day (for one-hot encoding)
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

# Streamlit app
st.title("Forest Fire Burned Area Prediction")

st.write("This app predicts the surface area burned by forest fire using neural network model.")

# Input values from user
X = st.number_input("X coordinate")
Y = st.number_input("Y coordinate")
month = st.selectbox("Month", months)
day = st.selectbox("Day", days)
FFMC = st.number_input("FFMC")
DMC = st.number_input("DMC")
DC = st.number_input("DC")
ISI = st.number_input("ISI")
temp = st.number_input("Temperature")
RH = st.number_input("Relative Humidity")
wind = st.number_input("Wind Speed")
rain = st.number_input("Rainfall")

# Prepare the input data
# 1. Collect the base features
input_data = [X, Y, FFMC, DMC, DC, ISI, temp, RH, wind, rain]

# 2. Add one-hot encoding for 'month'
month_one_hot = [1 if m == month else 0 for m in months]
input_data.extend(month_one_hot)

# 3. Add one-hot encoding for 'day'
day_one_hot = [1 if d == day else 0 for d in days]
input_data.extend(day_one_hot)

# Convert to NumPy array
input_data = np.array(input_data).reshape(1, -1)

# Ensure the input shape matches the model's expected input shape
if input_data.shape[1] != 29:
    st.error(f"Expected 29 features, but got {input_data.shape[1]}. Please check your inputs.")
else:
    # Make prediction
    prediction = model.predict(input_data)
    st.write(f"Predicted Surface Area: {prediction[0][0]:.2f} ha")


