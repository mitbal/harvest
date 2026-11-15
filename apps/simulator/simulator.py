import logging
from datetime import datetime

import pandas as pd
import altair as alt
import streamlit as st

import harvest.data as hd
import harvest.plot as hp
import harvest.simulator as hs
from harvest import chart_style as cs
from harvest import page_ui as ui
from harvest.utils import setup_logging


st.set_page_config(page_title='Panen Dividen | Compounding Simulator')
hp.enable_chart_theme()

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

SIM_MODES = {
    'fixed': 'Fixed yield',
    'stock': 'Single stock',
    'portfolio': 'Portfolio',
}
DEFAULT_SIM_MODE = 'fixed'

CAPITAL_COLOR = '#073f2e'
REINVESTED_COLOR = '#7cc4a4'


def parse_sim_param(qp):
    """Simulation mode from the URL, falling back to the default on a missing or bad value.

    Supported param: ?sim=fixed|stock|portfolio (default: fixed)
    """
    mode = str(qp.get('sim', DEFAULT_SIM_MODE)).strip().lower()
    return mode if mode in SIM_MODES else DEFAULT_SIM_MODE


def short_idr(value):
    """Compact amount for chart labels: 222M, 1.4B, 2.1T."""
    for size, suffix in ((1e12, 'T'), (1e9, 'B'), (1e6, 'M')):
        if abs(value) >= size:
            scaled = value / size
            text = f'{scaled:,.0f}' if abs(scaled) >= 100 else f'{scaled:,.1f}'.removesuffix('.0')
            return text + suffix
    return f'{value:,.0f}'


def fixed_yield_summary(return_df, initial_value):
    """The three Fixed yield tiles' numbers, with the same formulas as the old metrics."""
    final_investment = return_df['investment'].iloc[-1]
    final_returns = return_df['returns'].iloc[-1]
    return {
        'final_value': final_investment,
        'capital_multiple': final_investment / initial_value,
        'total_income': return_df['returns'].sum(),
        'final_income': final_returns,
        'yield_on_cost': final_returns / initial_value * 100,
    }


def year_by_year_table(return_df, initial_value):
    """Year, start value, income received, end value and yield on cost (percent) per year."""
    end_value = return_df['investment'].astype(float)
    start_value = end_value.shift(1).fillna(float(initial_value))
    return pd.DataFrame({
        'year': return_df['year'].values,
        'start_value': start_value.values,
        'income': return_df['returns'].values,
        'end_value': end_value.values,
        'yield_on_cost': (return_df['returns'] / initial_value * 100).values,
    })


def asset_chart_data(return_df, initial_value):
    """One row per year: capital, accumulated reinvested income, total and that year's income."""
    labels = [f'Y{i}' for i in range(1, len(return_df) + 1)]
    total = return_df['investment'].astype(float)
    return pd.DataFrame({
        'label': labels,
        'year': return_df['year'].values,
        'capital': float(initial_value),
        'reinvested': (total - initial_value).values,
        'total': total.values,
        'income': return_df['returns'].values,
        'total_label': [short_idr(v) for v in total],
    })


def asset_value_chart(chart_df):
    labels = chart_df['label'].tolist()
    x = alt.X('label:O', sort=labels, title=None, axis=alt.Axis(labelAngle=0, labelOverlap=True))
    tooltip = [
        alt.Tooltip('year:N', title='Year'),
        alt.Tooltip('capital:Q', title='Capital', format=',.0f'),
        alt.Tooltip('reinvested:Q', title='Reinvested income', format=',.0f'),
        alt.Tooltip('total:Q', title='Total', format=',.0f'),
        alt.Tooltip('income:Q', title="That year's income", format=',.0f'),
    ]
    y_axis = alt.Axis(
        title=None,
        labelExpr=(
            "datum.value >= 1e12 ? format(datum.value / 1e12, '~r') + 'T' : "
            "datum.value >= 1e9 ? format(datum.value / 1e9, '~r') + 'B' : "
            "datum.value >= 1e6 ? format(datum.value / 1e6, '~r') + 'M' : format(datum.value, ',.0f')"
        ),
    )
    bars = alt.Chart(chart_df).transform_fold(
        ['capital', 'reinvested'], as_=['part', 'amount']
    ).transform_calculate(
        part_order="datum.part === 'capital' ? 0 : 1"
    ).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=x,
        y=alt.Y('amount:Q', stack='zero', axis=y_axis),
        color=alt.Color(
            'part:N',
            scale=alt.Scale(domain=['capital', 'reinvested'], range=[CAPITAL_COLOR, REINVESTED_COLOR]),
            legend=None,
        ),
        order=alt.Order('part_order:Q'),
        tooltip=tooltip,
    )
    # Centred over the last bar; the right padding below covers up to about 6 characters
    # ("1,234B"). A longer label (only for extreme inputs) is right-aligned to the bar's
    # right edge instead, so it grows leftwards and can never run off the chart.
    label_text = chart_df['total_label'].iloc[-1]
    centred = len(label_text) <= 6
    last_label = alt.Chart(chart_df.tail(1)).mark_text(
        baseline='bottom', dy=-5, fontSize=12, fontWeight=800, color=ui.TOKENS['ink'],
        align='center' if centred else 'right',
    ).encode(
        x=x if centred else alt.X('label:O', sort=labels, bandPosition=1),
        y=alt.Y('total:Q'), text='total_label:N',
    )
    # Room on the right and top so the last bar's label is not clipped.
    return alt.layer(bars, last_label).properties(
        height=320, padding={'left': 5, 'top': 22, 'right': 30, 'bottom': 5},
    )


def legend_html(items):
    return '<div class="pdx-legend">' + ''.join(
        f'<span><span class="pdx-swatch" style="background:{color}"></span>{ui.esc(name)}</span>'
        for name, color in items
    ) + '</div>'


def card_heading(title, note='', extra=''):
    note_html = f'<p class="pdx-note" style="margin-top:3px">{ui.esc(note)}</p>' if note else ''
    ui.render(f'<div class="pdx-hd"><div><h2 class="pdx-h2">{ui.esc(title)}</h2>{note_html}</div>{extra}</div>')


def assumptions_heading(note):
    card_heading('Assumptions', note)


def note(text):
    ui.render(f'<p class="pdx-note">{ui.esc(text)}</p>')


def mode_layout(key):
    """Left assumptions pane and right results pane; they wrap, assumptions on top, when narrow."""
    return ui.split(key)

### End of Function definition

################################################################################

this_year = datetime.now().year
YEAR_MIN = 2010
YEAR_MAX = this_year - 1


def render_fixed():
    left, right = mode_layout('sim1')
    with left:
        with ui.card('sim1-assumptions'):
            assumptions_heading('A steady yield, with every dividend reinvested.')
            initial_value = st.number_input(
                'Initial investment (million rupiah)', value=120, min_value=1, max_value=10000,
                key='sim1_value',
            ) * 1_000_000
            num_year = st.number_input(
                'Number of years', value=10, min_value=1, max_value=50, key='sim1_years',
            )
            avg_yield = st.number_input(
                'Yield (percent)', value=6.35, min_value=0.1, max_value=99.9, key='sim1_yield',
            ) / 100
            note(
                'This projection does not model market-price changes, taxes, fees, or '
                'additional deposits. Single stock and Portfolio replay real dividend history '
                'instead of a fixed yield.'
            )

    return_df = simulate_compounding(initial_value, num_year, avg_yield)
    summary = fixed_yield_summary(return_df, initial_value)

    with right:
        ui.kpi_row([
            ui.kpi('Final asset value', f'IDR {summary["final_value"]:,.0f}',
                   f'{summary["capital_multiple"]:.2f}x your capital', tone='up'),
            ui.kpi('Total passive income', f'IDR {summary["total_income"]:,.0f}',
                   f'Over {num_year} years', tone='muted'),
            ui.kpi(f'Income in year {num_year}', f'IDR {summary["final_income"]:,.0f}',
                   f'{summary["yield_on_cost"]:.2f}% yield on cost', tone='up'),
        ])

        with ui.card('sim1-chart'):
            card_heading('Asset value by year', extra=legend_html([
                ('Capital', CAPITAL_COLOR), ('Reinvested income', REINVESTED_COLOR),
            ]))
            st.altair_chart(
                asset_value_chart(asset_chart_data(return_df, initial_value)),
                width='stretch', theme=None,
            )

        with ui.card('sim1-table'):
            with st.container(horizontal=True, vertical_alignment='center'):
                card_heading('Year by year')
                st.download_button(
                    label='Download CSV',
                    data=df_to_csv(return_df),
                    file_name='basic_compounding_sim.csv',
                    mime='text/csv',
                    icon=':material/download:',
                    width='content',
                )
            st.dataframe(
                year_by_year_table(return_df, initial_value),
                column_config={
                    'year': st.column_config.TextColumn('Year'),
                    'start_value': st.column_config.NumberColumn('Start value', format='IDR %,d'),
                    'income': st.column_config.NumberColumn('Income received', format='IDR %,d'),
                    'end_value': st.column_config.NumberColumn('End value', format='IDR %,d'),
                    'yield_on_cost': st.column_config.NumberColumn('Yield on cost', format='%.2f%%'),
                },
                hide_index=True,
                width='stretch',
            )


def render_stock():
    left, right = mode_layout('sim2')
    with left:
        with ui.card('sim2-assumptions'):
            assumptions_heading('Replay one IDX stock using its historical prices and dividends.')
            raw_stock_name = st.text_input(
                label='Stock name', value='BBCA.JK', key='sim2_stock',
                help='IDX tickers only, e.g. BBCA or BBCA.JK',
            )
            sim2_value = st.number_input(
                'Initial investment (million rupiah)', value=120, min_value=1,
                max_value=10000, key='sim2_value',
            ) * 1_000_000
            sim2_start = st.number_input(
                label='Start year', value=2014, min_value=YEAR_MIN, max_value=YEAR_MAX,
                key='sim2_start',
            )
            sim2_end = st.number_input(
                label='End year', value=YEAR_MAX, min_value=YEAR_MIN, max_value=YEAR_MAX,
                key='sim2_end',
            )
            sim2_drip = st.toggle(
                'Enable DRIP', value=True, key='sim2_drip',
                help='Show results for reinvesting dividends into more shares',
            )
            note(
                'The simulator buys whole 100-share lots near the start of the selected '
                'period, then compares reinvesting dividends in the same stock (bought on '
                'the payment date) with keeping them as cash. Portfolio value includes '
                'shares and residual cash; past results do not predict future returns.'
            )

    with right:
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

        if sim2_result is None:
            return

        with_drip, without_drip, transactions = sim2_result

        without_drip['type'] = hs.STRATEGY_NO_DRIP
        with_drip['type'] = hs.STRATEGY_DRIP
        display_df = with_drip if sim2_drip else without_drip

        final_val = display_df['investment'].iloc[-1]
        final_div = display_df['returns'].iloc[-1]
        total_div = display_df['returns'].sum()

        ui.kpi_row([
            ui.kpi('Final portfolio value', f'IDR {final_val:,.0f}'),
            ui.kpi('Total dividend income', f'IDR {total_div:,.0f}'),
            ui.kpi('Yield on cost (final)', f'{(final_div/sim2_value*100):.2f}%'),
            ui.kpi('Residual cash', f'IDR {display_df["cash"].iloc[-1]:,.0f}'),
        ])

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

        with ui.card('sim2-chart'):
            st.altair_chart((investment_chart + return_chart)\
                            .resolve_scale(y='independent'),
                            width="stretch", theme=None)

        with ui.card('sim2-table'):
            st.dataframe(
                display_df[['year', 'investment', 'returns']],
                column_config={
                    'year': st.column_config.TextColumn('Year'),
                    'investment': st.column_config.NumberColumn('Asset Value', format='IDR %,d'),
                    'returns': st.column_config.NumberColumn('Div. Received', format='IDR %,d'),
                },
                hide_index=True,
                width='stretch',
            )

        with st.expander('Activity Log'):
            st.dataframe(transactions, hide_index=True, width='stretch')

        st.download_button(
            label=f"Download {stock_name} Historical Data (CSV)",
            data=df_to_csv(plot_df[EXPORT_COLUMNS]),
            file_name=f'{stock_name}_historical_sim.csv',
            mime='text/csv',
        )


def render_portfolio():
    left, right = mode_layout('sim3')
    with left:
        with ui.card('sim3-assumptions'):
            assumptions_heading('Replay a portfolio of IDX stocks using the allocations below.')
            sim3_allocations = st.data_editor(
                pd.DataFrame({
                    'Ticker': ['BJTM.JK', 'SMSM.JK'],
                    'Investment (Mio IDR)': [60.0, 60.0],
                }),
                num_rows='dynamic',
                column_config={
                    'Ticker': st.column_config.TextColumn(
                        'Ticker', help='Stock ticker, e.g. BBCA.JK',
                    ),
                    'Investment (Mio IDR)': st.column_config.NumberColumn(
                        'Million IDR', help='Initial investment in million rupiah',
                        format='%,.1f', min_value=0, step=0.1,
                    ),
                },
                width='stretch',
                hide_index=True,
                key='sim3_stock_editor'
            )
            sim3_start = st.number_input(
                label='Start year', value=2014, min_value=YEAR_MIN, max_value=YEAR_MAX,
                key='sim3_start',
            )
            sim3_end = st.number_input(
                label='End year', value=YEAR_MAX, min_value=YEAR_MIN, max_value=YEAR_MAX,
                key='sim3_end',
            )
            sim3_drip = st.toggle(
                'Enable DRIP', value=True, key='sim3_drip',
                help='Show results for reinvesting dividends into more shares',
            )
            note(
                'Each position starts with whole 100-share lots; dividends are either '
                'reinvested in the stock that paid them (bought on the payment date) or '
                'kept as cash. The comparison includes historical price changes and '
                'uninvested cash.'
            )

    allocation_rows = [
        (row['Ticker'], row['Investment (Mio IDR)'] * 1_000_000)
        for _, row in sim3_allocations.iterrows()
    ]

    with right:
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

        if sim3_result is None:
            return

        aggregate_df, combined_plot_df, transactions = sim3_result

        sim3_strategy = hs.STRATEGY_DRIP if sim3_drip else hs.STRATEGY_NO_DRIP
        display_df = aggregate_df[aggregate_df['strategy'] == sim3_strategy]
        starting_capital = sum(allocations.values())

        final_val = display_df['investment'].iloc[-1]
        final_div = display_df['returns'].iloc[-1]
        total_div = display_df['returns'].sum()

        ui.kpi_row([
            ui.kpi('Final aggregate value', f'IDR {final_val:,.0f}'),
            ui.kpi('Total dividend collected', f'IDR {total_div:,.0f}'),
            ui.kpi('Yield on cost (final)', f'{(final_div/starting_capital*100):.2f}%'),
            ui.kpi('Residual cash', f'IDR {display_df["cash"].iloc[-1]:,.0f}'),
        ])

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

        with ui.card('sim3-chart'):
            st.altair_chart(
                (investment_chart + return_chart).resolve_scale(
                    y='independent', color='independent'
                ),
                width="stretch",
                theme=None,
            )

        with ui.card('sim3-table'):
            st.dataframe(
                display_df[['year', 'investment', 'returns']],
                column_config={
                    'year': st.column_config.TextColumn('Year'),
                    'investment': st.column_config.NumberColumn('Total Value', format='IDR %,d'),
                    'returns': st.column_config.NumberColumn('Total Div.', format='IDR %,d'),
                },
                hide_index=True,
                width='stretch',
            )

        with st.expander('Historical Transaction Log'):
            st.dataframe(transactions, hide_index=True, width='stretch')

        st.download_button(
            label="Download Multi-Stock Sim Data (CSV)",
            data=df_to_csv(aggregate_df[EXPORT_COLUMNS]),
            file_name='multi_stock_historical_sim.csv',
            mime='text/csv',
        )


RENDERERS = {
    'fixed': render_fixed,
    'stock': render_stock,
    'portfolio': render_portfolio,
}

################################################################################

ui.inject_styles()

st.title('Compounding Simulator')
ui.render('<p class="pdx-sub">See what reinvested dividends grow into over time</p>')

_default_mode = parse_sim_param(st.query_params)
sim_mode = st.segmented_control(
    'Simulation',
    list(SIM_MODES),
    format_func=SIM_MODES.get,
    default=_default_mode,
    required=True,
    key='sim_mode',
    label_visibility='collapsed',
)
if sim_mode not in SIM_MODES:
    sim_mode = _default_mode

# Keep the URL param in sync with the current widget state
st.query_params['sim'] = sim_mode

RENDERERS[sim_mode]()
