import logging
from datetime import datetime

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import harvest.data as hd
from harvest.utils import setup_logging

try:
    st.set_page_config(layout='wide')
except Exception as e:
    print('Set Page config has been called before')

st.title('# Simulator')

### Start of Function definition

@st.cache_resource
def get_logger(name, level=logging.INFO):

    logger = setup_logging(name, level)
    return logger

logger = get_logger('simulator')


@st.cache_data
def simulate_compounding(initial_value, num_year, avg_yield):

    logger.info(f'sim #1 simple compounding. {initial_value=}, {num_year=}, {avg_yield=}')
    
    return_df = hd.simulate_simple_compounding(initial_value, num_year, avg_yield)
    return return_df


@st.cache_data
def simulate_multi_stock_compounding(num_year, num_of_stocks, investment_per_stock, yield_per_stock):
    
    logger.info(f'sim #2 multi stock compounding. {num_year=}, {num_of_stocks=}, {investment_per_stock=}, {yield_per_stock=}')
    
    investments = np.zeros((num_year, num_of_stocks))
    returns = np.zeros((num_year, num_of_stocks))

    investments[0, :] = investment_per_stock
    for i in range(num_year):
        if i > 0:
            investments[i, 1:] = investments[i-1, 1:]
        
        for j in range(num_of_stocks):
            returns[i, j] = int(investments[i, j] * yield_per_stock[j])
            if j+1 < num_of_stocks:
                investments[i, j+1] += returns[i, j]
            elif i+1 < num_year:
                investments[i+1, 0] = investments[i, j] + returns[i, j]
    
    multi_return_df = pd.DataFrame({'investment': np.sum(investments, axis=1), 'returns': np.sum(returns, axis=1)})
    multi_return_df['year'] = [f'Year {i+1:02d}' for i in range(len(multi_return_df))]
    return multi_return_df


@st.cache_data
def simulate_real_multistock_compounding(initial_value, investment_per_stock, start_year, end_year, stock_list):
    
    divs = {}
    prices = {}
    for stock in stock_list:
        divs[stock] = hd.get_dividend_history_single_stock(stock, source='dag')
        prices[stock] = hd.get_daily_stock_price(stock, start_from=f'{start_year}-01-01')

    porto = {s: {'lot': 0, 'avg_price': 0} for s in stock_list}
    porto_df = pd.DataFrame()
    cash = initial_value
    investments = []
    returns = []
    initial_purchase = {}
    without_drip = pd.DataFrame()

    transactions = {}
    for y in range(start_year, end_year+1):
        div_event = []
        for i, s in enumerate(stock_list):
            if y == start_year:
                price_df = prices[s]
                buy_date = price_df[price_df['date'] >= f'{y}-01-01'].iloc[-1]
                close_price = buy_date['close']
                
                buy_lot = investment_per_stock[i] / close_price / 100
                buy_trx = int(buy_lot) * close_price * 100
                porto[s]['lot'] = int(buy_lot)
                porto[s]['avg_price'] = close_price
                cash -= buy_trx

                initial_purchase[s] = buy_lot
                if buy_date['date'] not in transactions:
                    transactions[buy_date['date']] = ''
                transactions[buy_date['date']] += \
                    f'buy {int(buy_lot)} lots of {s} @ {close_price} for total {int(buy_trx):,}\n'

            div_df = pd.DataFrame(divs[s])
            div_df['stock'] = s
            div_event += [div_df[(div_df['date'] >= f'{y}-01-01') & (div_df['date'] <= f'{y}-12-31')]]

        div_event_df = pd.concat(div_event).sort_values('date', ascending=True).reset_index(drop=True)
        ret = 0
        for idx, last_d in div_event_df.iterrows():
            div = porto[last_d['stock']]['lot'] * last_d['adjDividend'] * 100
            yield_on_cost = last_d['adjDividend'] / porto[last_d['stock']]['avg_price'] * 100
            cash += div
            ret += int(div)
            transactions[last_d['date']] = \
                f'receive dividend {last_d["adjDividend"]:,} of {last_d["stock"]} '\
                f'for {porto[last_d["stock"]]["lot"]} lots with total {int(div):,}\n'\
                f'yield-on-cost {yield_on_cost:.2f}% for average price {porto[last_d["stock"]]["avg_price"]:,.2f}'
               
            d = div_event_df.iloc[(idx+1) % len(div_event_df)]
            price_df = prices[d['stock']]
            buy_date = price_df[price_df['date'] >= last_d['date']].iloc[-1]
            close_price = buy_date['close']
            
            buy_lot = cash / close_price / 100
            buy_trx = int(buy_lot) * close_price * 100
            porto[d['stock']]['avg_price'] = (porto[d['stock']]['lot'] * porto[d['stock']]['avg_price']) \
               + buy_trx/100
            porto[d['stock']]['avg_price'] /= (porto[d['stock']]['lot'] + int(buy_lot))
            porto[d['stock']]['lot'] += int(buy_lot)
            cash -= buy_trx
            stock_name = d['stock']
            current_lot = porto[d['stock']]['lot']
            avg_price = porto[d['stock']]['avg_price']
            transactions[buy_date['date']] += '\n'\
                    f'buy {int(buy_lot)} lots of {stock_name} @ {close_price} for total {int(buy_trx):,}\n'\
                    f'current number of lots: {current_lot}, with average price: {avg_price:,.2f}'

        returns += [ret]
        inv = 0
        for s in stock_list:
            price_df = prices[s]
            buy_date = price_df[price_df['date'] <= f'{y}-12-31'].iloc[0]
            val = porto[s]['lot'] * buy_date['close'] * 100
            inv += val
            porto_df = pd.concat([porto_df,
                                pd.DataFrame({'stock': [s], 'lot': [porto[s]['lot']], 'price': [buy_date['close']], 'value': [val], 'year': [f'Year {y}']})
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
    return investments, returns, without_drip, porto_df, transactions

### End of Function definition

################################################################################


with st.container(border=True):
    st.write('## #1 Basic single instrument compounding simulation')

    cols = st.columns(3)
    initial_value = cols[0].number_input('Initial investment (in million rupiah)', value=120, min_value=1, max_value=1000) * 1_000_000
    num_year = cols[1].number_input('Number of years', value=10, min_value=1, max_value=50)
    avg_yield = cols[2].number_input('Yield (in percent)', value=6.35, min_value=0.1, max_value=99.9) / 100

    return_df = simulate_compounding(initial_value, num_year, avg_yield)

    cols = st.columns([0.33, 0.67])
    cols[0].dataframe(
        return_df[['year', 'investment', 'returns']],
        column_config={
            'year': st.column_config.TextColumn('Year'),
            'investment': st.column_config.NumberColumn('Investment', format='localized'), 
            'returns': st.column_config.NumberColumn('Returns', format='localized'), }, 
        hide_index=True
    )

    base_chart = alt.Chart(return_df)

    investment_chart = base_chart.mark_bar().encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('investment:Q', title='Investment')
    )

    return_chart = base_chart.mark_line(point=alt.OverlayMarkDef(size=100), size=5).encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('returns:Q', title='Returns'),
        color=alt.value('#FA8072')
    )

    compound_chart = alt.layer(investment_chart, return_chart)\
        .resolve_scale(y='independent')\
        .properties(title='Compound Interest Simulation',
                    height=420)

    cols[1].altair_chart(compound_chart)


##################################################################################


with st.container(border=True):
    st.write('## #2 Multi stock compounding simulation')

    cols = st.columns(3)
    num_of_stocks = cols[0].number_input('Number of stocks', value=2, min_value=1, max_value=12)

    investment_per_stock = [initial_value/1_000_000 / num_of_stocks for _ in range(num_of_stocks)]
    investment_per_stock = cols[1].text_area('investment per stock (in million rupiah)', value='\n'.join([f'{int(i):d}' for i in investment_per_stock]))
    investment_per_stock = [float(i)*1_000_000 for i in investment_per_stock.split('\n')]

    yield_per_stock = cols[2].text_area('yield per stock', value='\n'.join([f'{(avg_yield*100):.2f}' for _ in range(num_of_stocks)]))
    yield_per_stock = [float(i)/100 for i in yield_per_stock.split('\n')]

    multi_return_df = simulate_multi_stock_compounding(num_year, num_of_stocks, investment_per_stock, yield_per_stock)

    cols = st.columns([0.33, 0.67])

    cols[0].dataframe(
        multi_return_df[['year', 'investment', 'returns']], 
        column_config={
            'year': st.column_config.TextColumn('Year'),
            'investment': st.column_config.NumberColumn('Investment', format='localized'),
            'returns': st.column_config.NumberColumn('Returns', format='localized'),
        },
        hide_index=True
    )

    multi_investment_chart = alt.Chart(multi_return_df).mark_bar().encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('investment:Q', title='Investment')
    )

    multi_return_chart = alt.Chart(multi_return_df).mark_line(point=alt.OverlayMarkDef(size=100), size=5).encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('returns:Q', title='Returns'),
        color=alt.value('#8FBC8F')
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
        title='Investment Comparison',
        width=600,
        height=420
    )

    cols[1].altair_chart((combined_chart + compound_chart).resolve_scale(y='independent'))


#############################################################################################

with st.container(border=True):
    st.write('## #3 Single stock dividend reinvestment historical compounding simulation')

    this_year = datetime.now().year

    cols = st.columns(3)
    stock_name = cols[0].text_input(label='Stock Name', value='BBCA.JK').upper()
    start_year = cols[1].number_input(label='Start Year', value=2014, min_value=2010, max_value=this_year-2)
    end_year = cols[2].number_input(label='End Year', value=this_year-1, min_value=start_year+1, max_value=this_year-1)

    logger.info(f'Stock Name: {stock_name}, Start Year: {start_year}, End Year: {end_year}')
    div_df = hd.get_dividend_history_single_stock(stock_name, source='dag')

    try:
        price_df = hd.get_daily_stock_price(stock_name, start_from=f'{start_year}-01-01')
    except Exception as e:
        st.error(f'Cannot find the stock {stock_name}. Please check the stock name again and dont forget to add .JK for Indonesian stocks')
        st.stop()

    activities = []
    cash = initial_value
    porto = []
    divs = []
    investments = []
    returns = []

    initial_investment = 0
    without_drip = pd.DataFrame()

    if f'{start_year}-12-31' < price_df['date'].min():
        st.error(f'Data for {stock_name} is not available before {price_df["date"].min()}, Please change the year or select different stocks')
        st.stop()

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
        div = int(div_payment * np.sum(porto) * 100)
        cash += div
        activities.append(f'receive dividend payment {div_payment}')

        returns += [div]

        without_drip = pd.concat([without_drip, 
                                pd.DataFrame({'year': [f'Year {y}'], 
                                                'returns': [div_payment * initial_investment * 100],
                                                'investment': [ initial_investment * buy_date['close'] * 100 + div]
                                                })
                                ])

    with st.expander('Activity Log'):
        st.write(activities)

    return_df = pd.DataFrame({'investment': investments, 'returns': returns})
    return_df['year'] = [f'Year {i}' for i in range(start_year, end_year+1)]
    return_df['type'] = 'reinvest'

    cols = st.columns([0.33, 0.67])

    cols[0].dataframe(
        return_df[['year', 'investment', 'returns']], 
        column_config={
            'year': st.column_config.TextColumn('Year'),
            'investment': st.column_config.NumberColumn('Investment', format='localized'),
            'returns': st.column_config.NumberColumn('Returns', format='localized'),
        }, 
        hide_index=True,
        height=430)

    without_drip['type'] = 'no reinvest'
    return_df = pd.concat([without_drip, return_df])

    bar_color_scale = alt.Scale(scheme='tableau10')
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
    ).properties(
        title=f'{stock_name} Dividend Reinvestment Compounding',
        height=430
    )

    cols[1].altair_chart((investment_chart + return_chart)\
                    .resolve_scale(y='independent', color='independent'),
                    use_container_width=True)


################################################################################

with st.container(border=True):
    st.write('## #4 Multi stock dividend reinvestment simulation')

    # process input form
    cols = st.columns(2)
    with cols[0]:
        stocks_input_str = st.text_area(
            label='Enter stock list (one per line):',
            value='BJTM.JK\nSMSM.JK',
            height=70
        )
        stock_list_raw = stocks_input_str.split('\n')
        stock_list = [stock.strip().upper() for stock in stock_list_raw if stock.strip()]

    with cols[1]:
        num_of_stocks = len(stock_list)
        investment_per_stock = [initial_value/1_000_000 / num_of_stocks for _ in range(num_of_stocks)]
        investment_per_stock = cols[1].text_area('investment per stock (in million rupiah)', 
                                                 value='\n'.join([f'{int(i):d}' for i in investment_per_stock]), 
                                                 key='investment_per_stock_sim_4',
                                                 height=70)
        investment_per_stock = [float(i)*1_000_000 for i in investment_per_stock.split('\n')]

    # run the simulation
    investments, returns, without_drip, porto_df, transactions = simulate_real_multistock_compounding(initial_value, investment_per_stock, start_year, end_year, stock_list)

    # show log, display result table, and plot the graph
    with st.expander('Activity Log'):
        st.write(transactions)

    return_df = pd.DataFrame({'investment': investments, 'returns': returns})
    return_df['year'] = [f'Year {i}' for i in range(start_year, end_year+1)]

    cols = st.columns([0.33, 0.67])

    cols[0].dataframe(
        return_df[['year', 'investment', 'returns']], 
        column_config={
            'year': st.column_config.TextColumn('Year'),
            'investment': st.column_config.NumberColumn('Investment', format='localized'),
            'returns': st.column_config.NumberColumn('Returns', format='localized'),
        }, 
        hide_index=True,
        height=430
    )

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
    ).properties(
        height=430
    )

    cols[1].altair_chart((investment_chart + return_chart).resolve_scale(y='independent'),
                         use_container_width=True)
