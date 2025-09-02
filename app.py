import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Flight Delay Prediction", layout="wide")
st.title('✈️ Flight Delay Prediction')
st.markdown('A machine learning model to predict the probability of a flight being delayed.')

# --- Load the pre-trained model and features ---
try:
    model = joblib.load('flight_delay_model.pkl')
    model_features = joblib.load('model_features.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please run 'prediction.py' first to train and save the model.")
    st.stop()

# --- Load the original data to get categorical lists for dropdowns ---
@st.cache_data
def load_data_for_ui():
    df = pd.read_csv('flight_data.csv')
    df.dropna(subset=['carrier', 'origin', 'dest'], inplace=True)
    return df

df_ui = load_data_for_ui()

# --- Sidebar for User Input ---
st.sidebar.header('Flight Details')
st.sidebar.markdown('Enter the flight information below to get a prediction.')

# Get unique values for dropdowns
carriers = sorted(df_ui['carrier'].unique())
origins = sorted(df_ui['origin'].unique())
destinations = sorted(df_ui['dest'].unique())

# Input widgets
dep_delay = st.sidebar.number_input('Departure Delay (minutes)', value=0, min_value=-100)
air_time = st.sidebar.number_input('Air Time (minutes)', value=100, min_value=0)
distance = st.sidebar.number_input('Distance (miles)', value=500, min_value=0)
month = st.sidebar.selectbox('Month', list(range(1, 13)))
day_of_week = st.sidebar.selectbox('Day of the Week', list(range(0, 7)), format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
carrier = st.sidebar.selectbox('Carrier', carriers)
origin = st.sidebar.selectbox('Origin Airport', origins)
dest = st.sidebar.selectbox('Destination Airport', destinations)

# Prepare input data for prediction
input_data = {
    'dep_delay': dep_delay,
    'air_time': air_time,
    'distance': distance,
    'month': month,
    'day_of_week': day_of_week,
    'carrier': carrier,
    'origin': origin,
    'dest': dest
}

input_df = pd.DataFrame([input_data])
input_encoded = pd.get_dummies(input_df, columns=['carrier', 'origin', 'dest'], drop_first=True)

# Align columns to match the training data
input_aligned = input_encoded.reindex(columns=model_features, fill_value=0)

# --- Prediction and Output ---
if st.sidebar.button('Predict Delay'):
    # Make prediction
    prediction = model.predict(input_aligned)[0]

    # Display result
    st.subheader('Prediction Result')
    if prediction == 1:
        st.error('This flight is **predicted to be delayed** (more than 15 minutes).')
    else:
        st.success('This flight is **predicted to be on time**.')