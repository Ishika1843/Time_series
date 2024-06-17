import streamlit as st
import pandas as pd
from nixtla import NixtlaClient
import os

# Initialize Streamlit app title
st.title("Time Series Forecasting with Nixtla")

# Initialize NixtlaClient
api_key = os.getenv('NIXTLA_API_KEY')  # Replace with your API key or set it as an environment variable
if not api_key:
    st.error("API key is missing. Please set the NIXTLA_API_KEY environment variable.")
    st.stop()

nixtla_client = NixtlaClient(api_key=api_key)

# Function to load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv"
    df = pd.read_csv(url)
    # Rename columns to match the expected format
    df.columns = ['timestamp', 'value']
    return df

# Load data
df = load_data()

# Display the loaded data
st.subheader("Dataset Preview")
st.write(df.head())

# Get the number of periods to forecast from user input
h_value = st.text_input(label='Enter number of periods to forecast')
if not h_value:
    st.warning("Please enter the number of periods to forecast.")
    st.stop()

try:
    h = int(h_value)  # Number of periods to forecast
    if h <= 0:
        st.error("Please enter a positive integer for the number of periods.")
        st.stop()
except ValueError:
    st.error("Invalid input. Please enter a valid integer.")
    st.stop()

# Perform forecasting
try:
    forecast_df = nixtla_client.forecast(
        df=df,
        h=h,
        time_col='timestamp',
        target_col='value'
    )

    # Display the forecast results
    st.subheader(f"Forecast for the next {h} month")
    st.write(forecast_df)

    # Plot predictions
    st.subheader("Forecast Plot")
    fig = nixtla_client.plot(
        df=df,
        forecasts_df=forecast_df,
        time_col='timestamp',
        target_col='value'
    )
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error during forecasting: {e}")
