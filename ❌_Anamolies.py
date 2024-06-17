import streamlit as st
import pandas as pd
from nixtla import NixtlaClient
import os


api_key = os.getenv('NIXTLA_API_KEY')
nixtla_client = NixtlaClient(api_key=api_key)


@st.cache_data
def load_data(url):
    return pd.read_csv(url)

def main():
    st.title('Anomaly Detection and Visualization with Nixtla and Streamlit')

    # Load data from URL
    data_url = 'https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/peyton_manning.csv'
    df = load_data(data_url)

    # Display some basic information about the dataset
    st.subheader('Dataset Information')
    st.write(df.head())  # Display first few rows of the dataset

    # Detect anomalies
    anomalies_df = nixtla_client.detect_anomalies(
        df,
        time_col='timestamp',
        target_col='value',
        freq='D',
    )

    # Display detected anomalies
    st.subheader('Detected Anomalies')
    st.write(anomalies_df)

    # Plot anomalies if anomalies_df is not empty
    if not anomalies_df.empty:
        st.subheader('Anomaly Visualization')
        # Check column names in anomalies_df before plotting
        st.write(anomalies_df.columns)
    nixtla_client.plot(
    df, 
    anomalies_df,
    time_col='timestamp', 
    target_col='value'
)

      

if __name__ == '__main__':
    main()
