import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from prophet import Prophet
import io
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set the title of the Streamlit app
st.title("Time Series Analysis with Prophet")
st.header("You can hearby give datset that is in datatime format only")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Validate the input file
if uploaded_file is not None:
    if not uploaded_file.name.endswith(".csv"):
        st.error("Error: The file must be a CSV file.")
    else:
        # Read the CSV file into a Pandas DataFrame
        try:
            df = pd.read_csv(uploaded_file)
        except pd.errors.ParserError:
            st.error("Error: The file is not a valid CSV file.")
        else:
            # Display the DataFrame
            st.write("DataFrame:")
            st.write(df.head())
            st.dataframe(df)
            
            # Display the count of null values in each column
            st.write("Null values in each column:")
            st.dataframe(df.isnull().sum())
            
            # Convert Datetime column to datetime object
            df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
            
            # Display DataFrame info
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text("DataFrame info:")
            st.text(info_str)
            
            # Plot the original data
            st.write("Original Data Plot:")
            plt.figure(figsize=(10, 7))
            plt.plot(df['Datetime'], df['Count'])
            plt.xlabel("Datetime")
            plt.ylabel("Count")
            plt.title("Original Data Plot")
            st.pyplot(plt)
            
            # Data preparation
            df.index = pd.to_datetime(df['Datetime'])
            df['y'] = df['Count']
            df.drop(columns=['ID', 'Datetime', 'Count'], axis=1, inplace=True)
            df = df.resample('D').sum()
            df['ds'] = df.index
            
            st.write("Prepared DataFrame:")
            st.dataframe(df.head())
            
            # Split data into train and test sets
            size = 60
            train, test = train_test_split(df, test_size=size/len(df), shuffle=False)
            
            st.write("Train DataFrame:")
            st.dataframe(train.tail())
            
            st.write("Test DataFrame:")
            st.dataframe(test.head())

            # Train the Prophet model
            model = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.9)
            model.fit(train)

            # Make future predictions
            future = model.make_future_dataframe(periods=60)
            forecast = model.predict(future)

            # Display forecast components
            st.write("Forecast Components:")
            fig1 = model.plot_components(forecast)
            st.pyplot(fig1)

            # Plot the forecast vs actuals
            st.write("Forecast vs Actuals:")
            pred = forecast.iloc[-60:, :]
            plt.figure(figsize=(10, 7))
            plt.plot(test['ds'], test['y'], label='Actual')
            plt.plot(pred['ds'], pred['yhat'], color='red', label='Predicted')
            plt.fill_between(pred['ds'], pred['yhat_lower'], pred['yhat_upper'], color='orange', alpha=0.3)
            plt.xlabel("Date")
            plt.ylabel("Count")
            plt.title("Forecast vs Actuals")
            plt.legend()
            st.pyplot(plt)

            # Train model on the entire dataset and forecast future values
            model = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.9)
            model.fit(df)
            future = model.make_future_dataframe(periods=200)
            forecast = model.predict(future)

            # Plot the final forecast
            st.write("Final Forecast:")
            plt.figure(figsize=(10, 7))
            plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
            plt.xlabel("Date")
            plt.ylabel("Count")
            plt.title("Future Forecast")
            plt.legend()
            st.pyplot(plt)

            # Format the output
            st.write("-" * 20)
            st.write("Processing complete.")
else:
    st.info("Please upload a CSV file.")
