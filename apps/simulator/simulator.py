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
return_df['year'] = [f'Year {i+1:02d}' for i in range(len(return_df))]
st.dataframe(return_df)

import altair as alt

investment_chart = alt.Chart(return_df).mark_bar().encode(
    x=alt.X('year:O', title='Year'),
    y=alt.Y('investment:Q', title='Investment')
)

return_chart = alt.Chart(return_df).mark_line(point=True).encode(
    x=alt.X('year:O', title='Year'),
    y=alt.Y('returns:Q', title='Returns'),
    color=alt.value('#FA8072')
)

compound_chart = alt.layer(investment_chart, return_chart).resolve_scale(y='independent')

st.altair_chart(compound_chart)
