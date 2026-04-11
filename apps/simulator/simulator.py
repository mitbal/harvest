import logging
from datetime import datetime

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import harvest.data as hd
from harvest.utils import setup_logging


st.set_page_config(page_title='Harvest | Simulator', page_icon='📈', layout='wide')

# Custom CSS for modern look
st.markdown("""
<style>
    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 10px;
        border-radius: 10px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(250, 250, 250, 0.2);
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
    }
    .main-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 18px;
        color: #808495;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📈 Investor Simulator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Compound interest and historical dividend reinvestment modeling</p>', unsafe_allow_html=True)

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
def simulate_single_stock_compounding(initial_value, stock_name, start_year, end_year):
    
    logger.info(f'sim #3 single stock. {stock_name=}, {initial_value=}, {start_year=}, {end_year=}')
    div_df = hd.get_dividend_history_single_stock(stock_name, source='dag')

    if div_df is None:
        logger.error(f'No dividend data found for {stock_name}')

    price_df = hd.get_daily_stock_price(stock_name, start_from=f'{start_year}-01-01')

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
        activities.append(f'buy {int(buy):,} lots of {stock_name} @ {close_price:,.2f} at {buy_date["date"]}')

        if y == start_year:
            initial_investment = buy

        buy_date = price_df[price_df['date'] <= f'{y}-12-31'].iloc[0]
        investments += [np.sum(porto) * buy_date['close'] * 100]
        
        try:
            div_payment = div_df[(div_df['date'] >= f'{y}-01-01') & (div_df['date'] <= f'{y}-12-31')]['adjDividend'].sum()
            div = int(div_payment * np.sum(porto) * 100)
            cash += div
            activities.append(f'receive dividend payment {div_payment:,.2f}')

            returns += [div]
        except Exception as e:
            logger.error(f"Error calculating dividend for {stock_name} in year {y}: {e}")
            div_payment = 0
            div = 0
            returns += [0]

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
    
    return return_df,without_drip


@st.cache_data
def simulate_real_multistock_compounding(initial_value, investment_per_stock, start_year, end_year, stock_list):
    
    logger.info(f'sim #4 historical multistock. {stock_list=}, {initial_value=}, {investment_per_stock=}, {start_year=}, {end_year=}')

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
                if f'{start_year}-12-31' < price_df['date'].min():
                    st.error(f'Data for {s} is not available before {price_df["date"].min()}, Please change the year or select different stocks')
                    st.stop()
                    break

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

    cols = st.columns([1, 1, 1])
    final_investment = return_df['investment'].iloc[-1]
    final_returns = return_df['returns'].iloc[-1]
    total_returns = return_df['returns'].sum()
    
    cols[0].metric('Final Asset Value', f'IDR {final_investment:,.0f}')
    cols[1].metric('Total Passive Income', f'IDR {total_returns:,.0f}')
    cols[2].metric('Final Annual Yield', f'{(final_returns/final_investment*100):.2f}%')

    st.divider()

    cols = st.columns([0.33, 0.67])
    cols[0].dataframe(
        return_df[['year', 'investment', 'returns']],
        column_config={
            'year': st.column_config.TextColumn('Year'),
            'investment': st.column_config.NumberColumn('Investment', format='IDR %,d'), 
            'returns': st.column_config.NumberColumn('Returns (p.a.)', format='IDR %,d'), }, 
        hide_index=True,
        width='stretch'
    )

    base_chart = alt.Chart(return_df)

    investment_chart = base_chart.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('investment:Q', title='Investment (Asset Value)'),
        tooltip=[alt.Tooltip('year:O', title='Year'), 
                 alt.Tooltip('investment:Q', title='Investment', format=',.0f'),
                 alt.Tooltip('returns:Q', title='Returns', format=',.0f')]
    )

    return_chart = base_chart.mark_line(point=alt.OverlayMarkDef(size=60, filled=True), size=3, color='#FA8072').encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('returns:Q', title='Returns (Passive Income)'),
    )

    compound_chart = alt.layer(investment_chart, return_chart)\
        .resolve_scale(y='independent')\
        .properties(title='Single Instrument Compounding Projection',
                    height=450)

    cols[1].altair_chart(compound_chart, width="stretch")
    
    st.download_button(
        label="Download Projection Data (CSV)",
        data=return_df.to_csv(index=False),
        file_name='basic_compounding_sim.csv',
        mime='text/csv',
    )


##################################################################################


with st.container(border=True):
    st.write('## #2 Multi stock compounding simulation')

    cols = st.columns([1, 2])
    num_of_stocks = cols[0].number_input('Number of stocks', value=2, min_value=1, max_value=12)

    # Initial data for data_editor
    initial_allocations = pd.DataFrame({
        'Stock': [f'Stock {i+1}' for i in range(num_of_stocks)],
        'Investment (Mio IDR)': [initial_value/1_000_000 / num_of_stocks for _ in range(num_of_stocks)],
        'Expected Yield (%)': [avg_yield*100 for _ in range(num_of_stocks)]
    })

    edited_allocations = cols[1].data_editor(
        initial_allocations,
        num_rows='dynamic',
        column_config={
            'Stock': st.column_config.TextColumn('Stock Name'),
            'Investment (Mio IDR)': st.column_config.NumberColumn('Investment (Mio IDR)', format='%,d', min_value=0),
            'Expected Yield (%)': st.column_config.NumberColumn('Yield (%)', format='%.2f', min_value=0, max_value=100)
        },
        width='stretch',
        hide_index=True,
        key='multi_stock_editor'
    )

    investment_per_stock = [row['Investment (Mio IDR)'] * 1_000_000 for _, row in edited_allocations.iterrows()]
    yield_per_stock = [row['Expected Yield (%)'] / 100 for _, row in edited_allocations.iterrows()]
    num_of_stocks = len(edited_allocations)

    multi_return_df = simulate_multi_stock_compounding(num_year, num_of_stocks, investment_per_stock, yield_per_stock)

    cols = st.columns([1, 1, 1])
    final_investment = multi_return_df['investment'].iloc[-1]
    final_returns = multi_return_df['returns'].iloc[-1]
    total_returns = multi_return_df['returns'].sum()
    
    cols[0].metric('Final Aggregated Value', f'IDR {final_investment:,.0f}')
    cols[1].metric('Total Passive Income', f'IDR {total_returns:,.0f}')
    cols[2].metric('Weighted Annual Yield', f'{(final_returns/final_investment*100):.2f}%')

    st.divider()

    cols = st.columns([0.33, 0.67])

    cols[0].dataframe(
        multi_return_df[['year', 'investment', 'returns']], 
        column_config={
            'year': st.column_config.TextColumn('Year'),
            'investment': st.column_config.NumberColumn('Investment', format='IDR %,d'),
            'returns': st.column_config.NumberColumn('Returns', format='IDR %,d'),
        },
        hide_index=True,
        width='stretch'
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

    cols[1].altair_chart((combined_chart + compound_chart).resolve_scale(y='independent'), width="stretch")


#############################################################################################

with st.container(border=True):
    st.write('## #3 Single stock dividend reinvestment historical compounding simulation')

    this_year = datetime.now().year

    cols = st.columns([2, 1, 1, 1])
    stock_name = cols[0].text_input(label='Stock Name', value='BBCA.JK', help='Add .JK for Indonesian stocks').upper()
    start_year = cols[1].number_input(label='Start Year', value=2014, min_value=2010, max_value=this_year-2)
    end_year = cols[2].number_input(label='End Year', value=this_year-1, min_value=start_year+1, max_value=this_year-1)
    drip_enabled = cols[3].toggle('Enable DRIP', value=True, help='Automatically reinvest dividends to buy more shares')

    try:
        return_df, without_drip = simulate_single_stock_compounding(initial_value, stock_name, start_year, end_year)
    except Exception as e:
        logger.error(f'Error running sim3 for {stock_name}: {e}')
        st.error(f'Cannot find the stock {stock_name}. Please check the stock name again and dont forget to add .JK for Indonesian stocks')
        st.stop()

    cols = st.columns([0.33, 0.67])

    without_drip['type'] = 'No DRIP'
    return_df['type'] = 'With DRIP'
    display_df = return_df if drip_enabled else without_drip
    
    # Summary Metrics for Sim #3
    metric_cols = st.columns([1, 1, 1])
    final_val = display_df['investment'].iloc[-1]
    final_div = display_df['returns'].iloc[-1]
    total_div = display_df['returns'].sum()
    
    metric_cols[0].metric('Final Portfolio Value', f'IDR {final_val:,.0f}')
    metric_cols[1].metric('Total Dividend Income', f'IDR {total_div:,.0f}')
    metric_cols[2].metric('Yield on Cost (Final)', f'{(final_div/initial_value*100):.2f}%')

    st.divider()

    cols = st.columns([0.33, 0.67])
    cols[0].dataframe(
        display_df[['year', 'investment', 'returns']], 
        column_config={
            'year': st.column_config.TextColumn('Year'),
            'investment': st.column_config.NumberColumn('Asset Value', format='IDR %,d'),
            'returns': st.column_config.NumberColumn('Div. Received', format='IDR %,d'),
        }, 
        hide_index=True,
        width='stretch',
        height=430)

    # Combine for visual comparison in the main chart
    plot_df = pd.concat([without_drip, return_df])

    bar_color_scale = alt.Scale(domain=['No DRIP', 'With DRIP'], range=['#87CEFA', '#4682B4'])
    
    investment_chart = alt.Chart(plot_df).mark_bar(opacity=0.8, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('investment:Q', title='Investment Value'),
        xOffset=alt.XOffset('type:N', sort=['No DRIP', 'With DRIP']),
        color=alt.Color('type:N', scale=bar_color_scale, title='Strategy'),
        tooltip=[alt.Tooltip('year:O'), alt.Tooltip('type:N'), alt.Tooltip('investment:Q', format=',.0f')]
    )

    return_chart = alt.Chart(plot_df).mark_line(point=True).encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('returns:Q', title='Dividends'),
        color=alt.Color('type:N', scale=alt.Scale(range=['#FFD700', '#FF4500']), title='Dividend strategy')
    ).properties(
        title=f'{stock_name} Historical Performance: DRIP Comparison',
        height=430
    )

    cols[1].altair_chart((investment_chart + return_chart)\
                    .resolve_scale(y='independent', color='independent'),
                    width="stretch")
    
    st.download_button(
        label=f"Download {stock_name} Historical Data (CSV)",
        data=return_df.to_csv(index=False),
        file_name=f'{stock_name}_historical_sim.csv',
        mime='text/csv',
    )


################################################################################

with st.container(border=True):
    st.write('## #4 Multi stock dividend reinvestment simulation')

    # process input form
    cols = st.columns([1, 2])
    
    # Default stocks for Sim #4
    default_sim4_stocks = pd.DataFrame({
        'Ticker': ['BJTM.JK', 'SMSM.JK'],
        'Investment (Mio IDR)': [initial_value/1_000_000 / 2 for _ in range(2)]
    })

    edited_sim4 = cols[1].data_editor(
        default_sim4_stocks,
        num_rows='dynamic',
        column_config={
            'Ticker': st.column_config.TextColumn('Stock Ticker (e.g. BBCA.JK)'),
            'Investment (Mio IDR)': st.column_config.NumberColumn('Initial Investment', format='%,d', min_value=0)
        },
        width='stretch',
        hide_index=True,
        key='sim4_stock_editor'
    )

    stock_list = [row['Ticker'].strip().upper() for _, row in edited_sim4.iterrows() if row['Ticker'].strip()]
    investment_per_stock = [row['Investment (Mio IDR)'] * 1_000_000 for _, row in edited_sim4.iterrows()]

    # run the simulation
    try:
        if not stock_list:
            st.warning('Please add at least one stock ticker.')
            st.stop()
            
        investments, returns, without_drip, porto_df, transactions = simulate_real_multistock_compounding(initial_value, investment_per_stock, start_year, end_year, stock_list)
    except Exception as e:
        st.error('Error running simulation. Please check your stock tickers (add .JK for Indonesian stocks).')
        logger.error(f'Error on sim4 for stocks {stock_list}: {e}')
        st.stop()

    # show log, display result table, and plot the graph
    with st.expander('Historical Transaction Log'):
        log_items = []
        for date, desc in transactions.items():
            log_items.append({'Date': date, 'Activity': desc})
        st.table(pd.DataFrame(log_items).sort_values('Date', ascending=False))

    return_df = pd.DataFrame({'investment': investments, 'returns': returns})
    return_df['year'] = [f'Year {i}' for i in range(start_year, end_year+1)]

    # Summary Metrics for Sim #4
    metric_cols = st.columns([1, 1, 1])
    final_val = return_df['investment'].iloc[-1]
    final_div = return_df['returns'].iloc[-1]
    total_div = return_df['returns'].sum()
    
    metric_cols[0].metric('Final Aggregate Value', f'IDR {final_val:,.0f}')
    metric_cols[1].metric('Total Dividend Collected', f'IDR {total_div:,.0f}')
    metric_cols[2].metric('Portfolio Yield (Final)', f'{(final_div/initial_value*100):.2f}%')

    st.divider()

    cols = st.columns([0.33, 0.67])
    cols[0].dataframe(
        return_df[['year', 'investment', 'returns']], 
        column_config={
            'year': st.column_config.TextColumn('Year'),
            'investment': st.column_config.NumberColumn('Total Value', format='IDR %,d'),
            'returns': st.column_config.NumberColumn('Total Div.', format='IDR %,d'),
        }, 
        hide_index=True,
        width='stretch',
        height=430
    )

    porto_df['Strategy'] = 'With DRIP'
    without_drip['Strategy'] = 'No DRIP'
    
    # Value column rename for consistency
    without_drip.rename(columns={'value': 'Value'}, inplace=True)
    porto_df.rename(columns={'value': 'Value'}, inplace=True)
    
    combined_plot_df = pd.concat([porto_df, without_drip])
    
    investment_chart = alt.Chart(combined_plot_df).mark_bar(opacity=0.8).encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('Value:Q', title='Portfolio Value'),
        color=alt.Color('stock:N', title='Stock'),
        xOffset='Strategy:N',
        tooltip=[alt.Tooltip('year:O'), alt.Tooltip('stock:N'), alt.Tooltip('Strategy:N'), alt.Tooltip('Value:Q', format=',.0f')]
    )

    return_chart = alt.Chart(return_df).mark_line(point=True, color='#FA8072', size=3).encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('returns:Q', title='Total Dividends'),
    ).properties(
        title='Multi-Stock Historical DRIP Comparison',
        height=430
    )

    cols[1].altair_chart((investment_chart + return_chart).resolve_scale(y='independent'),
                         width="stretch")
    
    st.download_button(
        label="Download Multi-Stock Sim Data (CSV)",
        data=return_df.to_csv(index=False),
        file_name='multi_stock_historical_sim.csv',
        mime='text/csv',
    )
