✈️ Flight Delay Prediction

A machine learning web application that predicts whether a flight will be delayed based on flight details such as carrier, origin, destination, departure delay, distance, and time-related features. The application is built using Python, Scikit-learn, and Streamlit and provides an interactive interface for real-time predictions.

Project Overview:

Flight delays cause major disruptions in air travel. This project uses machine learning to analyze flight attributes and predict whether a flight will be delayed by more than 15 minutes.

The trained model is integrated into a Streamlit web application where users can input flight details and instantly receive a prediction.

Features:

Interactive Streamlit web interface

Predicts flight delay status

Uses trained machine learning model

Handles categorical variables using one-hot encoding

Real-time prediction with user inputs

Clean and simple UI for easy interaction

Tech Stack:

Python

Pandas

NumPy

Scikit-learn

Joblib

Streamlit

Project Structure:

flight-delay-prediction

├── app.py                  # Streamlit web application

├── flight_delay_model.pkl  # Trained machine learning model

├── model_features.pkl      # Feature columns used during training

├── requirements.txt        # Project dependencies

└── README.md

How It Works:

User enters flight information in the sidebar.

Input data is converted into a dataframe.

Categorical variables are encoded using one-hot encoding.

Features are aligned with the training dataset structure.

The trained model predicts whether the flight will be delayed.

Installation:

Clone the repository:

git clone https://github.com/your-username/flight-delay-prediction.git

cd flight-delay-prediction

Install dependencies:

pip install -r requirements.txt

Run the Application

Start the Streamlit app:

streamlit run app.py

The application will open in your browser.

Example Inputs:

Users can provide the following information:

Carrier

Origin Airport

Destination Airport

Departure Delay

Air Time

Distance

Month

Day of the Week

The model will return whether the flight is predicted to be delayed or on time.

Future Improvements:

Display probability of delay

Add more airlines and airports

Use larger flight datasets

Deploy using Docker / cloud infrastructure

Add model performance metrics

Live Demo:

https://flight-delay-prediction-vpktev2xxsv78lkmghnv2z.streamlit.app/
