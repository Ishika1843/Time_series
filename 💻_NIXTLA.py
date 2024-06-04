import nixtla
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from nixtla import NixtlaClient
import os
from utilsforecast.losses import mae

from dotenv import load_dotenv
load_dotenv()
st.set_page_config("Time series web forcasting")
st.title("WEB TRAFFIC FORCASTING BY NIXTLA")


nixtla_client = NixtlaClient(
    api_key = 'my_api_key_provided_by_nixtla'
)
nixtla_client = nixtla.NixtlaClient(api_key=os.getenv('NIXTLA_API_KEY'))
url= ('https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-machine-learning-python/' +
       'master/data/visitas_por_dia_web_cienciadedatos.csv')
df = pd.read_csv(url, sep=',', parse_dates=[0], date_format='%d/%m/%y')
df['unique_id'] = 'daily_visits'

df.head(10) 
st.dataframe(df)
st.header('Summary Statistics')
st.write(df.describe())
st.header('User Visits Over Time')
st.line_chart(df.set_index('date')['users'])
if __name__ == '__main__':
    st.write('DONE')

timegpt_cv_df = nixtla_client.cross_validation(
    df, 
    h=7, 
    n_windows=8, 
    time_col='date', 
    target_col='users', 
    freq='D',
    level=[80, 90, 99.5]
)
timegpt_cv_df.head()
print(timegpt_cv_df.head())
st.subheader("CROSS VALIDATION")
st.dataframe(timegpt_cv_df)
st.write(timegpt_cv_df.describe())
fig=nixtla_client.plot(
    df, 
    timegpt_cv_df.drop(columns=['cutoff', 'users']), 
    time_col='date',
    target_col='users',
    max_insample_length=90, 
    level=[80, 90, 99.5]
)
st.pyplot(fig)

mae_timegpt = mae(df = timegpt_cv_df.drop(columns=['cutoff']),
    models=['TimeGPT'],
    target_col='users')
st.dataframe(mae_timegpt)
st.subheader("We will add weekday indicators, which we will extract from the date column.")
for i in range(7):
    df[f'week_day_{i + 1}'] = 1 * (df['date'].dt.weekday == i)

df.head(10)
st.dataframe(df)
st.subheader("cross-validation procedure with the added exogenous variables.")

timegpt_cv_df_with_ex = nixtla_client.cross_validation(
    df, 
    h=7, 
    n_windows=8, 
    time_col='date', 
    target_col='users', 
    freq='D',
    level=[80, 90, 99.5]
)
timegpt_cv_df_with_ex.head()
st.dataframe(timegpt_cv_df_with_ex.head())

st.subheader("GRAPH")
fig=nixtla_client.plot(
    df, 
    timegpt_cv_df_with_ex.drop(columns=['cutoff', 'users']), 
    time_col='date',
    target_col='users',
    max_insample_length=90, 
    level=[80, 90, 99.5]
)
st.pyplot(fig)
st.subheader("MAE with exogenous variable")
mae_timegpt_with_exogenous = mae(df = timegpt_cv_df_with_ex.drop(columns=['cutoff']),
    models=['TimeGPT'],
    target_col='users')

st.dataframe(mae_timegpt_with_exogenous)
st.subheader("CONCLUSIONS")

mae_timegpt['Exogenous features'] = False
mae_timegpt_with_exogenous['Exogenous features'] = True
df_results = pd.concat([mae_timegpt, mae_timegpt_with_exogenous])
df_results = df_results.rename(columns={'TimeGPT':'MAE backtest'})
df_results = df_results.drop(columns={'unique_id'})
df_results['model'] = 'TimeGPT'

df_results[['model', 'Exogenous features', 'MAE backtest']]
st.dataframe(df_results)