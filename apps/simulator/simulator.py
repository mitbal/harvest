import pandas as pd
import streamlit as st


st.title('Historical Dividend Growth Simulator')

initial_value = st.number_input('Jumlah awal investasi', value=120_000_000)
num_year = st.number_input('Lama tahun', value=10)
avg_yield = st.number_input('Yield', value=0.10)

investments = [initial_value]
returns = []
for i in range(num_year):
    
    returns += [investments[i] * avg_yield]
    investments += [investments[i] + returns[i]]

returns += [investments[-1] * avg_yield]
return_df = pd.DataFrame({'investment': investments, 'returns': returns})
st.dataframe(return_df)
