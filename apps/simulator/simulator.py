import logging
from datetime import datetime

import pandas as pd
import altair as alt
import streamlit as st

import harvest.data as hd
import harvest.plot as hp
import harvest.simulator as hs
from harvest import chart_style as cs
from harvest.utils import setup_logging


st.set_page_config(page_title='Panen Dividen | Compounding Simulator')
hp.enable_chart_theme()
st.title('Compounding Simulator')

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: rgba(15, 118, 110, 0.08);
        padding: 10px;
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

### Start of Function definition

@st.cache_resource
def get_logger(name, level=logging.INFO):

    logger = setup_logging(name, level)
    return logger

logger = get_logger('simulator')


@st.cache_data(max_entries=32, ttl=6 * 60 * 60)
def simulate_compounding(initial_value, num_year, avg_yield):

    logger.info(f'sim #1 simple compounding. {initial_value=}, {num_year=}, {avg_yield=}')
    
    return_df = hd.simulate_simple_compounding(initial_value, num_year, avg_yield)
    return return_df


@st.cache_data(max_entries=32, ttl=6 * 60 * 60)
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


@st.cache_data(max_entries=16, ttl=6 * 60 * 60)
def simulate_real_multistock_compounding(allocations_tuple, start_year, end_year):
    allocations = dict(allocations_tuple)
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

@st.cache_data(max_entries=16, ttl=6 * 60 * 60)
def df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


EXPORT_COLUMNS = [
    'year', 'strategy', 'holdings_value', 'cash', 'portfolio_value',
    'dividend_income',
]

### End of Function definition

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
    cols[2].metric(
        'Yield on Cost (Final)',
        f'{(final_returns/initial_value*100):.2f}%',
        help='Final year income divided by the initial investment.',
    )

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

    return_chart = base_chart.mark_line(
        point=alt.OverlayMarkDef(size=60, filled=True),
        size=2.5,
        color=cs.CHART_AMBER,
    ).encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y(
            'returns:Q',
            title='Returns (Passive Income)',
            axis=alt.Axis(titleColor=cs.CHART_AMBER),
        ),
        tooltip=[
            alt.Tooltip('year:O', title='Year'),
            alt.Tooltip('returns:Q', title='Passive Income', format=',.0f'),
        ],
    )

    compound_chart = alt.layer(investment_chart, return_chart)\
        .resolve_scale(y='independent')\
        .properties(title='Single Instrument Compounding Projection',
                    height=450)

    cols[1].altair_chart(compound_chart, width="stretch", theme=None)
    
    st.download_button(
        label="Download Projection Data (CSV)",
        data=df_to_csv(return_df),
        file_name='basic_compounding_sim.csv',
        mime='text/csv',
    )


################################################################################

this_year = datetime.now().year
YEAR_MIN = 2010
YEAR_MAX = this_year - 1

sim2_box = st.container(border=True)
with sim2_box:
    st.write('## #2 Single stock dividend reinvestment historical compounding simulation')
    st.caption(
        'Replay one Indonesian (IDX) stock using its historical prices and '
        'dividends. The simulator buys whole 100-share lots near the start of '
        'the selected period, then compares reinvesting dividends in the same '
        'stock (bought on the payment date) with keeping them as cash. '
        'Portfolio value includes shares and residual cash; past results do '
        'not predict future returns.'
    )

    cols = st.columns([2, 2, 1, 1])
    raw_stock_name = cols[0].text_input(
        label='Stock Name', value='BBCA.JK', key='sim2_stock',
        help='IDX tickers only, e.g. BBCA or BBCA.JK',
    )
    sim2_value = cols[1].number_input(
        'Initial investment (in million rupiah)', value=120, min_value=1,
        max_value=10000, key='sim2_value',
    ) * 1_000_000
    sim2_start = cols[2].number_input(
        label='Start Year', value=2014, min_value=YEAR_MIN, max_value=YEAR_MAX,
        key='sim2_start',
    )
    sim2_end = cols[3].number_input(
        label='End Year', value=YEAR_MAX, min_value=YEAR_MIN, max_value=YEAR_MAX,
        key='sim2_end',
    )
    sim2_drip = st.toggle(
        'Enable DRIP', value=True, key='sim2_drip',
        help='Show results for reinvesting dividends into more shares',
    )

    sim2_result = None
    stock_name = raw_stock_name
    try:
        stock_name = hs.normalize_idx_ticker(raw_stock_name)
        if not stock_name:
            raise hs.SimulatorValidationError('Please enter a stock ticker.')
        sim2_result = simulate_single_stock_compounding(
            sim2_value, stock_name, sim2_start, sim2_end
        )
    except hs.SimulatorValidationError as e:
        st.error(str(e))
    except Exception:
        logger.exception(f'Unexpected error running sim2 for {stock_name}')
        st.error('The market data service could not complete this simulation.')

with sim2_box:
    if sim2_result is not None:
        with_drip, without_drip, transactions = sim2_result

        with st.expander('Activity Log'):
            st.dataframe(transactions, hide_index=True, width='stretch')

        without_drip['type'] = hs.STRATEGY_NO_DRIP
        with_drip['type'] = hs.STRATEGY_DRIP
        display_df = with_drip if sim2_drip else without_drip

        # Summary metrics for simulation #2
        metric_cols = st.columns(4)
        final_val = display_df['investment'].iloc[-1]
        final_div = display_df['returns'].iloc[-1]
        total_div = display_df['returns'].sum()

        metric_cols[0].metric('Final Portfolio Value', f'IDR {final_val:,.0f}')
        metric_cols[1].metric('Total Dividend Income', f'IDR {total_div:,.0f}')
        metric_cols[2].metric('Yield on Cost (Final)', f'{(final_div/sim2_value*100):.2f}%')
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
        plot_df = pd.concat([without_drip, with_drip])

        bar_color_scale = alt.Scale(
            domain=[hs.STRATEGY_NO_DRIP, hs.STRATEGY_DRIP],
            range=[cs.CHART_NEUTRAL, cs.CHART_PRIMARY],
        )

        investment_chart = alt.Chart(plot_df).mark_bar(opacity=0.86, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('year:O', title='Year'),
            y=alt.Y('investment:Q', title='Investment Value'),
            xOffset=alt.XOffset('type:N', sort=[hs.STRATEGY_NO_DRIP, hs.STRATEGY_DRIP]),
            color=alt.Color('type:N', scale=bar_color_scale, title='Strategy'),
            tooltip=[alt.Tooltip('year:O'), alt.Tooltip('type:N'), alt.Tooltip('investment:Q', format=',.0f')]
        )

        return_chart = alt.Chart(plot_df).mark_line(point=True, strokeWidth=2.25).encode(
            x=alt.X('year:O', title='Year'),
            y=alt.Y('returns:Q', title='Dividends'),
            color=alt.Color(
                'type:N',
                scale=bar_color_scale,
                title='Strategy',
            ),
            tooltip=[
                alt.Tooltip('year:O', title='Year'),
                alt.Tooltip('type:N', title='Strategy'),
                alt.Tooltip('returns:Q', title='Dividends', format=',.0f'),
            ],
        ).properties(
            title=f'{stock_name} Historical Performance: DRIP Comparison',
            height=430
        )

        cols[1].altair_chart((investment_chart + return_chart)\
                        .resolve_scale(y='independent'),
                        width="stretch", theme=None)

        st.download_button(
            label=f"Download {stock_name} Historical Data (CSV)",
            data=df_to_csv(plot_df[EXPORT_COLUMNS]),
            file_name=f'{stock_name}_historical_sim.csv',
            mime='text/csv',
        )


################################################################################

sim3_box = st.container(border=True)
with sim3_box:
    st.write('## #3 Multi stock dividend reinvestment simulation')
    st.caption(
        'Replay a portfolio of Indonesian (IDX) stocks using the allocations '
        'below. Each position starts with whole 100-share lots; dividends are '
        'either reinvested in the stock that paid them (bought on the payment '
        'date) or kept as cash. The comparison includes historical price '
        'changes and uninvested cash.'
    )

    cols = st.columns([2, 1, 1])
    sim3_allocations = cols[0].data_editor(
        pd.DataFrame({
            'Ticker': ['BJTM.JK', 'SMSM.JK'],
            'Investment (Mio IDR)': [60.0, 60.0],
        }),
        num_rows='dynamic',
        column_config={
            'Ticker': st.column_config.TextColumn('Stock Ticker (e.g. BBCA.JK)'),
            'Investment (Mio IDR)': st.column_config.NumberColumn(
                'Initial Investment (Mio IDR)', format='%,.1f', min_value=0,
                step=0.1,
            ),
        },
        width='stretch',
        hide_index=True,
        key='sim3_stock_editor'
    )
    sim3_start = cols[1].number_input(
        label='Start Year', value=2014, min_value=YEAR_MIN, max_value=YEAR_MAX,
        key='sim3_start',
    )
    sim3_end = cols[1].number_input(
        label='End Year', value=YEAR_MAX, min_value=YEAR_MIN, max_value=YEAR_MAX,
        key='sim3_end',
    )
    sim3_drip = cols[2].toggle(
        'Enable DRIP', value=True, key='sim3_drip',
        help='Show results for reinvesting dividends into more shares',
    )

    allocation_rows = [
        (row['Ticker'], row['Investment (Mio IDR)'] * 1_000_000)
        for _, row in sim3_allocations.iterrows()
    ]

    sim3_result = None
    allocations = {}
    try:
        allocations = hs.build_allocations(allocation_rows)
        sim3_result = simulate_real_multistock_compounding(
            tuple(sorted(allocations.items())), sim3_start, sim3_end
        )
    except hs.SimulatorValidationError as e:
        st.error(str(e))
    except Exception:
        logger.exception(f'Unexpected error on sim3 for stocks {list(allocations)}')
        st.error('The market data service could not complete this simulation.')

with sim3_box:
    if sim3_result is not None:
        aggregate_df, combined_plot_df, transactions = sim3_result

        with st.expander('Historical Transaction Log'):
            st.dataframe(transactions, hide_index=True, width='stretch')

        sim3_strategy = hs.STRATEGY_DRIP if sim3_drip else hs.STRATEGY_NO_DRIP
        display_df = aggregate_df[aggregate_df['strategy'] == sim3_strategy]
        starting_capital = sum(allocations.values())

        # Summary metrics for simulation #3
        metric_cols = st.columns(4)
        final_val = display_df['investment'].iloc[-1]
        final_div = display_df['returns'].iloc[-1]
        total_div = display_df['returns'].sum()

        metric_cols[0].metric('Final Aggregate Value', f'IDR {final_val:,.0f}')
        metric_cols[1].metric('Total Dividend Collected', f'IDR {total_div:,.0f}')
        metric_cols[2].metric('Yield on Cost (Final)', f'{(final_div/starting_capital*100):.2f}%')
        metric_cols[3].metric('Residual Cash', f'IDR {display_df["cash"].iloc[-1]:,.0f}')

        st.divider()

        cols = st.columns([0.33, 0.67])
        cols[0].dataframe(
            display_df[['year', 'investment', 'returns']],
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

        investment_chart = alt.Chart(combined_plot_df).mark_bar(opacity=0.86).encode(
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
            tooltip=[
                alt.Tooltip('year:O', title='Year'),
                alt.Tooltip('strategy:N', title='Strategy'),
                alt.Tooltip('returns:Q', title='Total Dividends', format=',.0f'),
            ],
        ).properties(
            title='Multi-Stock Historical DRIP Comparison',
            height=430
        )

        cols[1].altair_chart(
            (investment_chart + return_chart).resolve_scale(
                y='independent', color='independent'
            ),
            width="stretch",
            theme=None,
        )

        st.download_button(
            label="Download Multi-Stock Sim Data (CSV)",
            data=df_to_csv(aggregate_df[EXPORT_COLUMNS]),
            file_name='multi_stock_historical_sim.csv',
            mime='text/csv',
        )
