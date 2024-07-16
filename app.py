import pandas as pd
import streamlit as st
from datetime import date, datetime

st.title('Dividend Harvesting')

with st.sidebar:
    start_date = st.date_input('Select start date', date(2024, 1, 1))

    end_date = st.date_input('Select end date', date(2024, 12, 31))

    target = st.text_input('Target', 100_000_000)

df = pd.read_csv('stockbit.csv', delimiter=';')

# st.dataframe(df)

col1, col2 = st.columns(2)

total_dividend = str(df['Total Dividend'].sum())
percentage = float(total_dividend) / float(target) * 100

with col1:
    st.metric(label='Total Dividend', value=total_dividend)

with col2:
    st.metric(label='Percent from Target', value=percentage)


st.write('Last 10 dividend')
st.dataframe(df[:10])


