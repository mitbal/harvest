import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import harvest.data as hd


st.title('# Simulator')


################################################################################


st.write('## Basic single instrument compounding simulation')

cols = st.columns(3)
initial_value = cols[0].number_input('Jumlah awal investasi (in million rupiah)', value=120) * 1_000_000
num_year = cols[1].number_input('Lama tahun', value=10)
avg_yield = cols[2].number_input('Yield', value=0.10)

investments = [initial_value]
returns = []
for i in range(num_year):
    
    returns += [investments[i] * avg_yield]
    investments += [investments[i] + returns[i]]

returns += [investments[-1] * avg_yield]
return_df = pd.DataFrame({'investment': investments, 'returns': returns})[:num_year]
return_df['year'] = [f'Year {i+1:02d}' for i in range(len(return_df))]

st.dataframe(return_df[['year', 'investment', 'returns']], 
             column_config={'investment': st.column_config.NumberColumn('Investment', format='accounting'), 
                           'returns': st.column_config.NumberColumn('Returns', format='accounting'), }, 
             hide_index=True)

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


##################################################################################


st.write('## Multi stock compounding simulation')

cols = st.columns(3)
num_of_stocks = cols[0].number_input('Jumlah saham', value=2, min_value=1, max_value=12)

investment_per_stock = [initial_value / num_of_stocks for _ in range(num_of_stocks)]
investment_per_stock = cols[1].text_area('investment per stock', value='\n'.join([str(i) for i in investment_per_stock]))
investment_per_stock = [float(i) for i in investment_per_stock.split('\n')]

yield_per_stock = cols[2].text_area('yield per stock', value='\n'.join([str(avg_yield) for _ in range(num_of_stocks)]))
yield_per_stock = [float(i) for i in yield_per_stock.split('\n')]

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

# st.write('investments', investments)
# st.write('returns', returns)

multi_return_df = pd.DataFrame({'investment': np.sum(investments, axis=1), 'returns': np.sum(returns, axis=1)})
multi_return_df['year'] = [f'Year {i+1:02d}' for i in range(len(multi_return_df))]
st.dataframe(multi_return_df[['year', 'investment', 'returns']], 
             column_config={'investment': st.column_config.NumberColumn('Investment', format='accounting'), 
                           'returns': st.column_config.NumberColumn('Returns', format='accounting'), }, 
             hide_index=True)

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


#############################################################################################


st.write('## #3 Single stock dividend reinvestment historical compounding simulation')

cols = st.columns(3)
stock_name = cols[0].text_input(label='Stock Name', value='BBCA.JK')
start_year = cols[1].number_input(label='Start Year', value=2014, min_value=2010, max_value=2024)
end_year = cols[2].number_input(label='End Year', value=2024, min_value=2014, max_value=2024)

div_df = hd.get_dividend_history_single_stock(stock_name)
price_df = hd.get_daily_stock_price(stock_name, start_from=f'{start_year}-01-01')

activities = []
cash = initial_value
porto = []
divs = []
investments = []
returns = []

initial_investment = 0
without_drip = pd.DataFrame()

for y in range(start_year, end_year+1):

    buy_date = price_df[price_df['date'] >= f'{y}-01-01'].iloc[-1]
    close_price = buy_date['close']
    
    buy = cash / close_price / 100
    porto += [int(buy)]
    cash -= int(buy) * close_price * 100
    activities.append(f'buy {int(buy)} lots of {stock_name} @ {close_price} at {buy_date["date"]}')

    if y == start_year:
        initial_investment = buy

    buy_date = price_df[price_df['date'] <= f'{y}-12-31'].iloc[0]
    investments += [np.sum(porto) * buy_date['close'] * 100]
    
    div_payment = div_df[(div_df['date'] >= f'{y}-01-01') & (div_df['date'] <= f'{y}-12-31')]['adjDividend'].sum()
    div = div_payment * np.sum(porto) * 100
    cash += div
    activities.append(f'receive dividend {div}')

    returns += [div]

    without_drip = pd.concat([without_drip, 
                              pd.DataFrame({'year': [f'Year {y}'], 
                                            'returns': [div_payment * initial_investment * 100],
                                            'investment': [ initial_investment * buy_date['close'] * 100 + div]
                                            })
                            ])

st.write('activities', activities)

return_df = pd.DataFrame({'investment': investments, 'returns': returns})
return_df['year'] = [f'Year {i}' for i in range(start_year, end_year+1)]
return_df['type'] = 'reinvest'

st.dataframe(return_df[['year', 'investment', 'returns']], 
             column_config={'investment': st.column_config.NumberColumn('Investment', format='accounting'), 
                           'returns': st.column_config.NumberColumn('Returns', format='accounting'), }, 
             hide_index=True)

without_drip['type'] = 'no reinvest'
return_df = pd.concat([without_drip, return_df])

bar_color_scale = alt.Scale(scheme='tableau10') # Or 'category10', 'accent', etc.
investment_chart = alt.Chart(return_df).mark_bar().encode(
    x=alt.X('year:O', title='Year'),
    y=alt.Y('investment:Q', title='Investment'),
    xOffset=alt.XOffset('type:N', sort=['no reinvest', 'reinvest']),
    color=alt.Color('type:N',
                    scale=bar_color_scale,
                   ),
)

return_chart = alt.Chart(return_df).mark_line(point=True).encode(
    x=alt.X('year:O', title='Year'),
    y=alt.Y('returns:Q', title='Returns'),
    color=alt.Color('type:N').scale(domain=['reinvest', 'no reinvest'], range=['red', 'yellow'])
)

st.altair_chart((investment_chart + return_chart)\
                .resolve_scale(y='independent', color='independent'),
                use_container_width=True)


################################################################################


st.write('## #4 Multi Stock Dividend Reinvestment')

stock_list = st.text_input('Enter stock list (separated by comma):', 'SMSM.JK, SIDO.JK', max_chars=52)
stock_list = [stock.strip() for stock in stock_list.split(',')]

prices = {}
divs = {}
for stock in stock_list:
    prices[stock] = hd.get_daily_stock_price(stock, start_from=f'{start_year}-01-01')
    divs[stock] = hd.get_dividend_history_single_stock(stock)

porto = {s: 0 for s in stock_list}
returns = {}

porto_df = pd.DataFrame(columns=['year', 'stock', 'lot', 'price', 'value'])

cash = initial_value
activities = []
investments = []
returns = []
initial_purchase = {}
without_drip = pd.DataFrame()
for y in range(start_year, end_year+1):
    
    div_event = []
    for s in stock_list:

        if y == start_year:
            price_df = prices[s]
            buy_date = price_df[price_df['date'] >= f'{y}-01-01'].iloc[-1]
            close_price = buy_date['close']
            
            buy_lot = (initial_value/len(stock_list)) / close_price / 100
            buy_trx = int(buy_lot) * close_price * 100
            porto[s] += int(buy_lot)  
            cash -= buy_trx

            initial_purchase[s] = buy_lot
            activities.append(f'buy {int(buy_lot)} lots of {s} @ {close_price} at {buy_date['date']} for total {buy_trx}, cash remaining {cash}')

        div_df = pd.DataFrame(divs[s])
        div_df['stock'] = s
        div_event += [div_df[(div_df['date'] >= f'{y}-01-01') & (div_df['date'] <= f'{y}-12-31')]]

    div_event_df = pd.concat(div_event).sort_values('date', ascending=True).reset_index(drop=True)
    ret = 0
    for idx, last_d in div_event_df.iterrows():

        div = porto[last_d['stock']] * last_d['adjDividend'] * 100
        cash += div
        ret += div
        activities.append(f'receive dividend {div} from {last_d['stock']} @ {last_d['date']}')

        d = div_event_df.iloc[(idx+1) % len(div_event_df)]
        price_df = prices[d['stock']]
        buy_date = price_df[price_df['date'] >= last_d['date']].iloc[-1]
        close_price = buy_date['close']
        
        buy_lot = cash / close_price / 100
        buy_trx = int(buy_lot) * close_price * 100
        porto[d['stock']] += int(buy_lot)  
        cash -= buy_trx
        activities.append(f'buy {int(buy_lot)} lots of {d["stock"]} @ {close_price} at {buy_date['date']} for total {buy_trx}, cash remaining {cash}')

    returns += [ret]
    inv = 0
    for s in stock_list:
        price_df = prices[s]
        buy_date = price_df[price_df['date'] <= f'{y}-12-31'].iloc[0]
        val = porto[s] * buy_date['close'] * 100
        inv += val
        porto_df = pd.concat([porto_df,
                              pd.DataFrame({'stock': [s], 'lot': [porto[s]], 'price': [buy_date['close']], 'value': [val], 'year': [f'Year {y}']})
                            ])
        without_drip = pd.concat([without_drip,
                                  pd.DataFrame({
                                    'stock': [s],
                                    'year': [f'Year {y}'],
                                    'lot': [initial_purchase[s]],
                                    'price': [buy_date['close']],
                                    'value': [initial_purchase[s] * buy_date['close'] * 100]
                                    })
                                ])

    investments += [inv]

st.write('activity log', activities)
# st.dataframe(porto_df, hide_index=True)

return_df = pd.DataFrame({'investment': investments, 'returns': returns})
return_df['year'] = [f'Year {i}' for i in range(start_year, end_year+1)]
st.dataframe(return_df[['year', 'investment', 'returns']], 
             column_config={'investment': st.column_config.NumberColumn('Investment', format='accounting'), 
                           'returns': st.column_config.NumberColumn('Returns', format='accounting'), }, 
             hide_index=True)

porto_df['type'] = 'with drip'
without_drip['type'] = 'no drip'
porto_df = pd.concat([porto_df, without_drip])
investment_chart = alt.Chart(porto_df).mark_bar().encode(
    x=alt.X('year:O', title='Year'),
    y=alt.Y('value:Q', title='Investment'),
    color='stock',
    xOffset='type'
)

return_chart = alt.Chart(return_df).mark_line(point=True).encode(
    x=alt.X('year:O', title='Year'),
    y=alt.Y('returns:Q', title='Returns'),
    color=alt.value('#FA8072')
)

st.altair_chart((investment_chart + return_chart).resolve_scale(y='independent')
                , use_container_width=True)
