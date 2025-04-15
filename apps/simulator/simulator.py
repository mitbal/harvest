import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import harvest.data as hd


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
return_df = pd.DataFrame({'investment': investments, 'returns': returns})[:num_year]
return_df['year'] = [f'Year {i+1:02d}' for i in range(len(return_df))]
st.dataframe(return_df[['year', 'investment', 'returns']], hide_index=True)

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

return_df['type'] = 'Basic'
multi_return_df['type'] = 'Multi'
combined_df = pd.concat([return_df, multi_return_df], axis=0)

combined_chart = alt.Chart(combined_df).mark_bar().encode(
    x=alt.X('year:O', title='Year'),
    y=alt.Y('investment:Q', title='Investments'),
    color=alt.Color('type:N', title='Type'),
    xOffset='type:N',
).properties(
    width=600,
    height=400
)

st.altair_chart((combined_chart + compound_chart).resolve_scale(y='independent'))


st.write('## Single Stock Dividend Reinvestment')
stock_name = st.text_input(label='Stock Name', value='INDF.JK')
start_year = st.number_input(label='Start Year', value=2014, min_value=2000, max_value=2024)
end_year = st.number_input(label='End Year', value=2024, min_value=2000, max_value=2024)
div_df = hd.get_dividend_history_single_stock(stock_name)
# st.dataframe(div_df)

price_df = hd.get_daily_stock_price(stock_name, start_from=f'{start_year}-01-01')
# st.dataframe(price_df)

activities = []
cash = initial_value
porto = []
divs = []
investments = []
returns = []

for y in range(start_year, end_year+1):

    # if y == start_year:
    # close_price = ['close'].iloc[-1]
    buy_date = price_df[price_df['date'] >= f'{y}-01-01'].iloc[-1]
    # st.write(buy_date)
    close_price = buy_date['close']
    
    buy = cash / close_price / 100
    porto += [int(buy)]
    cash -= int(buy) * close_price * 100
    activities.append(f'buy {int(buy)} lots of {stock_name} @ {close_price} at {buy_date["date"]}')

    buy_date = price_df[price_df['date'] <= f'{y}-12-31'].iloc[0]
    investments += [np.sum(porto) * buy_date['close'] * 100]
    
    div = div_df[(div_df['date'] >= f'{y}-01-01') & (div_df['date'] <= f'{y}-12-31')]['adjDividend'].sum() * np.sum(porto) * 100
    cash += div
    activities.append(f'receive dividend {div}')

    returns += [div]

st.write(activities)

return_df = pd.DataFrame({'investment': investments, 'returns': returns})
return_df['year'] = [f'Year {i}' for i in range(start_year, end_year+1)]
st.dataframe(return_df[['year', 'investment', 'returns']], hide_index=True)

investment_chart = alt.Chart(return_df).mark_bar().encode(
    x=alt.X('year:O', title='Year'),
    y=alt.Y('investment:Q', title='Investment')
)

return_chart = alt.Chart(return_df).mark_line(point=True).encode(
    x=alt.X('year:O', title='Year'),
    y=alt.Y('returns:Q', title='Returns'),
    color=alt.value('#FA8072')
)

st.altair_chart((investment_chart + return_chart).resolve_scale(y='independent')
                , use_container_width=True)
