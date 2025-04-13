import numpy as np
import pandas as pd
import streamlit as st


st.title('Historical Dividend Growth Simulator')

st.write('## Basic compounding simulator')

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


st.write('# Multi year compounding')

num_of_stocks = st.number_input('Jumlah saham', value=2)

investment_per_stock = [initial_value / num_of_stocks for _ in range(num_of_stocks)]
# st.write('investment per stock', investment_per_stock)

investment_per_stock = st.text_area('investment per stock', value='\n'.join([str(i) for i in investment_per_stock]))
investment_per_stock = [float(i) for i in investment_per_stock.split('\n')]

# simulate
investments = np.zeros((num_year, num_of_stocks))
returns = np.zeros((num_year, num_of_stocks))

investments[0, :] = investment_per_stock
for i in range(num_year):

    if i > 0:
        investments[i, 1:] = investments[i-1, 1:]
    
    for j in range(num_of_stocks):

        returns[i, j] = investments[i, j] * avg_yield
        if j+1 < num_of_stocks:
            investments[i, j+1] += returns[i, j]
        elif i+1 < num_year:
            investments[i+1, 0] = investments[i, j] + returns[i, j]

st.write('investments', investments)
st.write('returns', returns)

multi_return_df = pd.DataFrame({'investment': np.sum(investments, axis=1), 'returns': np.sum(returns, axis=1)})
multi_return_df['year'] = [f'Year {i+1:02d}' for i in range(len(multi_return_df))]
st.dataframe(multi_return_df)

multi_investment_chart = alt.Chart(multi_return_df).mark_bar().encode(
    x=alt.X('year:O', title='Year'),
    y=alt.Y('investment:Q', title='Investment')
)

multi_return_chart = alt.Chart(multi_return_df).mark_line(point=True).encode(
    x=alt.X('year:O', title='Year'),
    y=alt.Y('returns:Q', title='Returns'),
    color=alt.value('blue')
)

compound_chart = alt.layer(return_chart, multi_return_chart)

# .resolve_scale(y='independent')
st.altair_chart(compound_chart)

return_df['type'] = 'Basic'
multi_return_df['type'] = 'Multi'
combined_df = pd.concat([return_df, multi_return_df], axis=0)

combined_chart = alt.Chart(combined_df).mark_bar().encode(
    x=alt.X('year:O', title='Year'),
    y=alt.Y('returns:Q', title='Returns'),
    color=alt.Color('type:N', title='Type'),
    xOffset='type:N',
).properties(
    width=600,
    height=400
)

st.altair_chart(combined_chart)
