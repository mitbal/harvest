import logging
from datetime import datetime

import pandas as pd
import altair as alt
import streamlit as st

import harvest.data as hd
import harvest.simulator as hs
from harvest.utils import setup_logging


st.set_page_config(page_title='Panen Dividen | Compounding Simulator')
st.title('Compounding Simulator')

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

# st.markdown('<p class="main-title">📈 Investor Simulator</p>', unsafe_allow_html=True)
# st.markdown('<p class="sub-title">Compound interest and historical dividend reinvestment modeling</p>', unsafe_allow_html=True)

### Start of Function definition

@st.cache_resource
def get_logger(name, level=logging.INFO):

    logger = setup_logging(name, level)
    return logger

logger = get_logger('simulator')


@st.cache_data(max_entries=512)
def simulate_compounding(initial_value, num_year, avg_yield):

    logger.info(f'sim #1 simple compounding. {initial_value=}, {num_year=}, {avg_yield=}')
    
    return_df = hd.simulate_simple_compounding(initial_value, num_year, avg_yield)
    return return_df


@st.cache_data(max_entries=256)
def simulate_single_stock_compounding(initial_value, stock_name, start_year, end_year):
    logger.info(f'sim #2 single stock. {stock_name=}, {initial_value=}, {start_year=}, {end_year=}')
    dividends = hd.get_dividend_history_single_stock(stock_name, source='dag')
    prices = hd.get_daily_stock_price(stock_name, start_from=f'{start_year}-01-01')
    aggregate, _, transactions = hs.simulate_historical_portfolio(
        {stock_name: initial_value},
        {stock_name: prices},
        {stock_name: dividends},
        start_year,
        end_year,
    )
    with_drip = aggregate[aggregate['strategy'] == hs.STRATEGY_DRIP].copy()
    without_drip = aggregate[aggregate['strategy'] == hs.STRATEGY_NO_DRIP].copy()
    return with_drip, without_drip, transactions


@st.cache_data(max_entries=256)
def simulate_real_multistock_compounding(allocations, start_year, end_year):
    logger.info(
        f'sim #3 historical multistock. {allocations=}, '
        f'{start_year=}, {end_year=}'
    )
    prices = {}
    dividends = {}
    for stock in allocations:
        dividends[stock] = hd.get_dividend_history_single_stock(stock, source='dag')
        prices[stock] = hd.get_daily_stock_price(
            stock, start_from=f'{start_year}-01-01'
        )
    return hs.simulate_historical_portfolio(
        allocations, prices, dividends, start_year, end_year
    )

### End of Function definition

@st.cache_data(max_entries=64)
def df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

################################################################################


with st.container(border=True):
    st.write('## #1 Basic single instrument compounding simulation')
    st.caption(
        'Estimate how an investment grows when it earns the same yield every '
        'year and all income is reinvested. This projection does not model '
        'market-price changes, taxes, fees, or additional deposits.'
    )

    cols = st.columns(3)
    initial_value = cols[0].number_input('Initial investment (in million rupiah)', value=120, min_value=1, max_value=10000) * 1_000_000
    num_year = cols[1].number_input('Number of years', value=10, min_value=1, max_value=50)
    avg_yield = cols[2].number_input('Yield (in percent)', value=6.35, min_value=0.1, max_value=99.9) / 100

    return_df = simulate_compounding(initial_value, num_year, avg_yield)

    cols = st.columns([1, 1, 1])
    final_investment = return_df['investment'].iloc[-1]
    final_returns = return_df['returns'].iloc[-1]
    total_returns = return_df['returns'].sum()
    
    cols[0].metric('Final Asset Value', f'IDR {final_investment:,.0f}')
    cols[1].metric('Total Passive Income', f'IDR {total_returns:,.0f}')
    beginning_balance = final_investment - final_returns
    cols[2].metric('Final Annual Yield', f'{(final_returns/beginning_balance*100):.2f}%')

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
        data=df_to_csv(return_df),
        file_name='basic_compounding_sim.csv',
        mime='text/csv',
    )


#############################################################################################

with st.container(border=True):
    st.write('## #2 Single stock dividend reinvestment historical compounding simulation')
    st.caption(
        'Replay one stock using its historical prices and dividends. The '
        'simulator buys whole 100-share lots near the start of the selected '
        'period, then compares reinvesting dividends in the same stock with '
        'keeping them as cash. Portfolio value includes shares and residual '
        'cash; past results do not predict future returns.'
    )

    this_year = datetime.now().year

    cols = st.columns([2, 1, 1, 1])
    stock_name = cols[0].text_input(label='Stock Name', value='BBCA.JK', help='Add .JK for Indonesian stocks').upper()
    start_year = cols[1].number_input(label='Start Year', value=2014, min_value=2010, max_value=this_year-2)
    end_year = cols[2].number_input(label='End Year', value=this_year-1, min_value=start_year+1, max_value=this_year-1)
    drip_enabled = cols[3].toggle('Enable DRIP', value=True, help='Automatically reinvest dividends to buy more shares')

    try:
        return_df, without_drip, transactions = simulate_single_stock_compounding(
            initial_value, stock_name, start_year, end_year
        )
    except hs.SimulatorValidationError as e:
        st.error(str(e))
        st.stop()
    except Exception:
        logger.exception(f'Unexpected error running sim2 for {stock_name}')
        st.error('The market data service could not complete this simulation.')
        st.stop()

    with st.expander('Activity Log'):
        st.dataframe(transactions, hide_index=True, width='stretch')

    without_drip['type'] = 'No DRIP'
    return_df['type'] = 'With DRIP'
    display_df = return_df if drip_enabled else without_drip
    
    # Summary Metrics for Sim #3
    metric_cols = st.columns(4)
    final_val = display_df['investment'].iloc[-1]
    final_div = display_df['returns'].iloc[-1]
    total_div = display_df['returns'].sum()
    
    metric_cols[0].metric('Final Portfolio Value', f'IDR {final_val:,.0f}')
    metric_cols[1].metric('Total Dividend Income', f'IDR {total_div:,.0f}')
    metric_cols[2].metric('Yield on Cost (Final)', f'{(final_div/initial_value*100):.2f}%')
    metric_cols[3].metric('Residual Cash', f'IDR {display_df["cash"].iloc[-1]:,.0f}')

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
        data=df_to_csv(display_df),
        file_name=f'{stock_name}_historical_sim.csv',
        mime='text/csv',
    )


################################################################################

with st.container(border=True):
    st.write('## #3 Multi stock dividend reinvestment simulation')
    st.caption(
        'Replay a portfolio using the allocations below and the date range '
        'selected in Simulation #2. Each position starts with whole 100-share '
        'lots; dividends are either reinvested in the stock that paid them or '
        'kept as cash. The comparison includes historical price changes and '
        'uninvested cash.'
    )

    # process input form
    cols = st.columns([1, 2])
    
    # Default stocks for simulation #3
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

    allocation_rows = [
        (row['Ticker'], row['Investment (Mio IDR)'] * 1_000_000)
        for _, row in edited_sim4.iterrows()
    ]
    stock_list = [
        str(stock).strip().upper()
        for stock, _ in allocation_rows
        if pd.notna(stock) and str(stock).strip()
    ]

    # run the simulation
    try:
        allocations = hs.build_allocations(allocation_rows)
        aggregate_df, combined_plot_df, transactions = \
            simulate_real_multistock_compounding(allocations, start_year, end_year)
    except hs.SimulatorValidationError as e:
        st.error(str(e))
        st.stop()
    except Exception:
        st.error('The market data service could not complete this simulation.')
        logger.exception(f'Unexpected error on sim4 for stocks {stock_list}')
        st.stop()

    # show log, display result table, and plot the graph
    with st.expander('Historical Transaction Log'):
        st.dataframe(transactions, hide_index=True, width='stretch')

    return_df = aggregate_df[
        aggregate_df['strategy'] == hs.STRATEGY_DRIP
    ].copy()
    starting_capital = sum(allocations.values())

    # Summary metrics for simulation #3
    metric_cols = st.columns(4)
    final_val = return_df['investment'].iloc[-1]
    final_div = return_df['returns'].iloc[-1]
    total_div = return_df['returns'].sum()
    
    metric_cols[0].metric('Final Aggregate Value', f'IDR {final_val:,.0f}')
    metric_cols[1].metric('Total Dividend Collected', f'IDR {total_div:,.0f}')
    metric_cols[2].metric('Portfolio Yield (Final)', f'{(final_div/starting_capital*100):.2f}%')
    metric_cols[3].metric('Residual Cash', f'IDR {return_df["cash"].iloc[-1]:,.0f}')

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

    combined_plot_df = combined_plot_df.rename(columns={'value': 'Value'})
    
    investment_chart = alt.Chart(combined_plot_df).mark_bar(opacity=0.8).encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('Value:Q', title='Portfolio Value'),
        color=alt.Color('stock:N', title='Stock'),
        xOffset='Strategy:N',
        tooltip=[alt.Tooltip('year:O'), alt.Tooltip('stock:N'), alt.Tooltip('Strategy:N'), alt.Tooltip('Value:Q', format=',.0f')]
    )

    return_chart = alt.Chart(aggregate_df).mark_line(point=True, size=3).encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('returns:Q', title='Total Dividends'),
        color=alt.Color('strategy:N', title='Dividend strategy'),
    ).properties(
        title='Multi-Stock Historical DRIP Comparison',
        height=430
    )

    cols[1].altair_chart(
        (investment_chart + return_chart).resolve_scale(
            y='independent', color='independent'
        ),
        width="stretch"
    )
    
    st.download_button(
        label="Download Multi-Stock Sim Data (CSV)",
        data=df_to_csv(return_df),
        file_name='multi_stock_historical_sim.csv',
        mime='text/csv',
    )
