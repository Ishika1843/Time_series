import nixtla
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from nixtla import NixtlaClient
import os
from utilsforecast.losses import mae
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Time Series Web Forecasting")
st.title("Web Traffic Forecasting with Nixtla API")
st.subheader("Uploaded file should not have time , only date will work")
st.subheader("you can view the needed dataset by NIXTLA in the sidebar")
# Create a file uploader widget
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
x=st.number_input("Enter the range of days you want data for:", min_value=1, max_value=365)
# Validate the input file or use a default URL
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=',', parse_dates=[0], date_format='%d/%m/%y')
    except pd.errors.ParserError:
        st.error("Error: The file is not a valid CSV file.")
        st.stop()
    else:
        st.write("DataFrame:")
        st.write(df.head())
else:
    st.info("Please upload a CSV file.")
    url = (uploaded_file)
    df = pd.read_csv(url, sep=',', parse_dates=[0], date_format='%d/%m/%y')

df['unique_id'] = 'daily_visits'
st.dataframe(df.head(10))
st.header('Summary Statistics')
st.write(df.describe())
st.header('User Visits Over Time')
st.line_chart(df.set_index('date')['users'])

# Initialize NixtlaClient with API key
nixtla_client = NixtlaClient(api_key=os.getenv('NIXTLA_API_KEY'))

# Perform cross-validation
timegpt_cv_df = nixtla_client.cross_validation(
    df, 
    h=x, 
    n_windows=8, 
    time_col='date', 
    target_col='users', 
    freq='D',
    level=[80, 90, 99.5]
)
st.subheader("Cross Validation")
st.dataframe(timegpt_cv_df.head())
st.write(timegpt_cv_df.describe())

# Plot cross-validation results
fig = nixtla_client.plot(
    df, 
    timegpt_cv_df.drop(columns=['cutoff', 'users']), 
    time_col='date',
    target_col='users',
    max_insample_length=90, 
    level=[80, 90, 99.5]
)
st.pyplot(fig)

# Calculate Mean Absolute Error (MAE)
mae_timegpt = mae(df=timegpt_cv_df.drop(columns=['cutoff']),
                  models=['TimeGPT'],
                  target_col='users')
st.dataframe(mae_timegpt)

# Adding weekday indicators
st.subheader("Adding weekday indicators")

for i in range(x):
    df[f'week_day_{i + 1}'] = 1 * (df['date'].dt.weekday == i)
st.dataframe(df.head(10))

# Cross-validation with exogenous variables
st.subheader("Cross-validation with exogenous variables")
timegpt_cv_df_with_ex = nixtla_client.cross_validation(
    df, 
    h=x, 
    n_windows=8, 
    time_col='date', 
    target_col='users', 
    freq='D',
    level=[80, 90, 99.5]
)
st.dataframe(timegpt_cv_df_with_ex.head())

# Plot cross-validation results with exogenous variables
fig = nixtla_client.plot(
    df, 
    timegpt_cv_df_with_ex.drop(columns=['cutoff', 'users']), 
    time_col='date',
    target_col='users',
    max_insample_length=90, 
    level=[80, 90, 99.5]
)
st.pyplot(fig)

# Calculate MAE with exogenous variables
st.subheader("MAE with exogenous variables")
mae_timegpt_with_exogenous = mae(df=timegpt_cv_df_with_ex.drop(columns=['cutoff']),
                                 models=['TimeGPT'],
                                 target_col='users')
st.dataframe(mae_timegpt_with_exogenous)

# Summary of results
st.subheader("Conclusions")
mae_timegpt['Exogenous features'] = False
mae_timegpt_with_exogenous['Exogenous features'] = True
df_results = pd.concat([mae_timegpt, mae_timegpt_with_exogenous])
df_results = df_results.rename(columns={'TimeGPT': 'MAE backtest'})
df_results = df_results.drop(columns={'unique_id'})
df_results['model'] = 'TimeGPT'

st.dataframe(df_results[['model', 'Exogenous features', 'MAE backtest']])
