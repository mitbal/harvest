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
from harvest.universe import read_redis_frame
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

# Chart heights chosen so the tiles and the chart card fit the first screen at 1440x800
FIXED_CHART_HEIGHT = 360
DRIP_CHART_HEIGHT = 370
ALLOCATION_EDITOR_HEIGHT = 143  # header, two rows and the add-row line

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
        height=FIXED_CHART_HEIGHT, padding={'left': 5, 'top': 22, 'right': 30, 'bottom': 5},
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


STOCK_STATE_KEY = 'sim2_submitted'
PORTFOLIO_STATE_KEY = 'sim3_submitted'
ALLOCATION_COLUMNS = ['Ticker', 'Investment (Mio IDR)']


def stock_submission(raw_ticker, value_mio, start, end, drip):
    """Single stock inputs as submitted. The ticker is kept as typed, so the form can show it back."""
    return {
        'stock': '' if raw_ticker is None else str(raw_ticker),
        'value_mio': value_mio,
        'start': int(start),
        'end': int(end),
        'drip': bool(drip),
    }


def default_stock_submission():
    return stock_submission('BBCA.JK', 120, 2014, YEAR_MAX, True)


IDX_UNIVERSE_KEY = 'div_score_jkse'  # same Redis table the screener reads
STOCK_HELP_SELECT = 'Type to search IDX stocks by ticker'
STOCK_HELP_TEXT = 'IDX tickers only, e.g. BBCA or BBCA.JK'


def _safe_normalize(raw_ticker):
    """hs.normalize_idx_ticker, or '' for anything it rejects."""
    try:
        return hs.normalize_idx_ticker(raw_ticker)
    except hs.SimulatorValidationError:
        return ''


def idx_ticker_options(frame):
    """Sorted, unique XXXX.JK tickers from a dividend score table (symbol or stock column)."""
    if frame is None or frame.empty:
        return ()
    column = 'symbol' if 'symbol' in frame.columns else 'stock' if 'stock' in frame.columns else None
    values = frame[column] if column else pd.Series(frame.index)
    tickers = {_safe_normalize(value) for value in values.dropna()}
    tickers.discard('')
    return tuple(sorted(tickers))


def stock_select_options(options, submitted_stock):
    """Selectbox options and index for the last submitted ticker.

    The submitted value is normalised first, so 'bbca' selects BBCA.JK. A ticker that is not
    in the list is added (sorted in), so the selectbox never fails on it. Blank gives no
    selection (index None).
    """
    options = list(options)
    raw = '' if submitted_stock is None else str(submitted_stock).strip()
    if not raw:
        return options, None
    current = _safe_normalize(raw) or raw
    if current not in options:
        options = sorted(options + [current])
    return options, options.index(current)


@st.cache_data(ttl=10 * 60, show_spinner=False)
def load_idx_tickers():
    """Every JKSE ticker in the screener's dividend score table, or () if it cannot be read."""
    try:
        tickers = idx_ticker_options(read_redis_frame(IDX_UNIVERSE_KEY))
    except Exception:
        logger.exception('Could not load the IDX stock list for the simulator')
        return ()
    if not tickers:
        logger.warning(f'IDX stock list ({IDX_UNIVERSE_KEY}) is empty; using free-text input')
    return tickers


def _clean_cell(value):
    return None if value is None or pd.isna(value) else value


def portfolio_submission(editor_df, start, end, drip):
    """Portfolio inputs as submitted: the allocation rows exactly as edited (blanks become None)."""
    rows = []
    for ticker, investment in editor_df[ALLOCATION_COLUMNS].itertuples(index=False, name=None):
        ticker = _clean_cell(ticker)
        investment = _clean_cell(investment)
        rows.append((
            None if ticker is None else str(ticker),
            None if investment is None else float(investment),
        ))
    return {'rows': rows, 'start': int(start), 'end': int(end), 'drip': bool(drip)}


def default_portfolio_submission():
    return {
        'rows': [('BJTM.JK', 60.0), ('SMSM.JK', 60.0)],
        'start': 2014, 'end': YEAR_MAX, 'drip': True,
    }


def allocations_frame(rows):
    """The allocation editor's data for stored rows, with the editor's column names and dtypes."""
    frame = pd.DataFrame(list(rows), columns=ALLOCATION_COLUMNS)
    frame['Ticker'] = frame['Ticker'].astype(object)
    frame['Investment (Mio IDR)'] = pd.to_numeric(frame['Investment (Mio IDR)'], errors='coerce').astype(float)
    return frame


def allocation_rows(rows):
    """(ticker, investment in rupiah) pairs for hs.build_allocations, as the page always built them."""
    return [
        (row['Ticker'], row['Investment (Mio IDR)'] * 1_000_000)
        for _, row in allocations_frame(rows).iterrows()
    ]


def submitted_inputs(state_key, default_factory):
    """The last submitted inputs for a mode, seeded with the defaults on first load."""
    if state_key not in st.session_state:
        st.session_state[state_key] = default_factory()
    return st.session_state[state_key]


def drip_advantage(final_with_drip, final_without_drip):
    """DRIP advantage tile: final value with DRIP minus without, and that as a share of without."""
    diff = float(final_with_drip) - float(final_without_drip)
    if abs(diff) < 0.5:
        diff = 0.0
    sign = '-' if diff < 0 else ''
    value = f'{sign}IDR {abs(diff):,.0f}'
    if final_without_drip:
        pct = diff / float(final_without_drip) * 100
        note_text = f'{pct:+.1f}% vs no DRIP' if diff else '0.0% vs no DRIP'
    else:
        pct = None
        note_text = ''
    tone = 'up' if diff > 0 else 'down' if diff < 0 else 'muted'
    return {'diff': diff, 'pct': pct, 'value': value, 'note': note_text, 'tone': tone}


def drip_advantage_kpi(final_with_drip, final_without_drip):
    adv = drip_advantage(final_with_drip, final_without_drip)
    return ui.kpi('DRIP advantage', adv['value'], adv['note'], tone=adv['tone'])


def run_button():
    return st.form_submit_button('Run simulation', type='primary', width='stretch')


def table_card(key, df, column_config, csv_bytes, file_name):
    with ui.card(key):
        with st.container(horizontal=True, vertical_alignment='center'):
            card_heading('Year by year')
            st.download_button(
                label='Download CSV',
                data=csv_bytes,
                file_name=file_name,
                mime='text/csv',
                icon=':material/download:',
                width='content',
            )
        st.dataframe(df, column_config=column_config, hide_index=True, width='stretch')


def log_card(key, title, transactions):
    with ui.card(key):
        with st.expander(title, expanded=False):
            st.dataframe(transactions, hide_index=True, width='stretch')


def render_stock():
    submitted = submitted_inputs(STOCK_STATE_KEY, default_stock_submission)
    left, right = mode_layout('sim2')
    with left:
        with ui.card('sim2-assumptions'):
            assumptions_heading('Replay one IDX stock using its historical prices and dividends.')
            # Widget defaults come from the last submission, so after switching modes and back
            # (Streamlit drops the state of widgets that were not rendered) the form still
            # matches the results. Keyed widgets ignore `value` while they have state.
            tickers = load_idx_tickers()
            with st.form('sim2_form', border=False), st.container(key='sim-fields-sim2'):
                if tickers:
                    options, index = stock_select_options(tickers, submitted['stock'])
                    raw_stock_name = st.selectbox(
                        label='Stock name', options=options, index=index, key='sim2_stock',
                        placeholder='Search IDX stocks', accept_new_options=True,
                        help=STOCK_HELP_SELECT,
                    )
                else:
                    # Stock list unavailable (e.g. Redis down): plain ticker input.
                    raw_stock_name = st.text_input(
                        label='Stock name', value=submitted['stock'], key='sim2_stock',
                        help=STOCK_HELP_TEXT,
                    )
                value_mio = st.number_input(
                    'Initial investment (million rupiah)', value=submitted['value_mio'],
                    min_value=1, max_value=10000, key='sim2_value',
                )
                start_col, end_col = st.columns(2, gap='small')
                start = start_col.number_input(
                    label='Start year', value=submitted['start'], min_value=YEAR_MIN,
                    max_value=YEAR_MAX, key='sim2_start',
                )
                end = end_col.number_input(
                    label='End year', value=submitted['end'], min_value=YEAR_MIN,
                    max_value=YEAR_MAX, key='sim2_end',
                )
                drip = st.toggle(
                    'Enable DRIP', value=submitted['drip'], key='sim2_drip',
                    help='Show results for reinvesting dividends into more shares',
                )
                if run_button():
                    submitted = stock_submission(raw_stock_name, value_mio, start, end, drip)
                    st.session_state[STOCK_STATE_KEY] = submitted
            note(
                'The simulator buys whole 100-share lots near the start of the selected '
                'period, then compares reinvesting dividends in the same stock (bought on '
                'the payment date) with keeping them as cash. Portfolio value includes '
                'shares and residual cash; past results do not predict future returns.'
            )

    sim2_value = submitted['value_mio'] * 1_000_000
    with right:
        sim2_result = None
        stock_name = submitted['stock']
        try:
            stock_name = hs.normalize_idx_ticker(submitted['stock'])
            if not stock_name:
                raise hs.SimulatorValidationError('Please enter a stock ticker.')
            sim2_result = simulate_single_stock_compounding(
                sim2_value, stock_name, submitted['start'], submitted['end']
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
        display_df = with_drip if submitted['drip'] else without_drip

        final_val = display_df['investment'].iloc[-1]
        final_div = display_df['returns'].iloc[-1]
        total_div = display_df['returns'].sum()

        ui.kpi_row([
            ui.kpi('Final portfolio value', f'IDR {final_val:,.0f}'),
            ui.kpi('Total dividend income', f'IDR {total_div:,.0f}'),
            ui.kpi('Yield on cost (final)', f'{(final_div/sim2_value*100):.2f}%'),
            drip_advantage_kpi(with_drip['investment'].iloc[-1], without_drip['investment'].iloc[-1]),
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
        ).properties(height=DRIP_CHART_HEIGHT)

        with ui.card('sim2-chart'):
            card_heading(f'{stock_name}: DRIP comparison')
            st.altair_chart((investment_chart + return_chart)\
                            .resolve_scale(y='independent'),
                            width="stretch", theme=None)

        table_card(
            'sim2-table',
            display_df[['year', 'investment', 'returns']],
            {
                'year': st.column_config.TextColumn('Year'),
                'investment': st.column_config.NumberColumn('Asset Value', format='IDR %,d'),
                'returns': st.column_config.NumberColumn('Div. Received', format='IDR %,d'),
            },
            df_to_csv(plot_df[EXPORT_COLUMNS]),
            f'{stock_name}_historical_sim.csv',
        )

        log_card('sim2-log', 'Activity Log', transactions)


def render_portfolio():
    submitted = submitted_inputs(PORTFOLIO_STATE_KEY, default_portfolio_submission)
    left, right = mode_layout('sim3')
    with left:
        with ui.card('sim3-assumptions'):
            assumptions_heading('Replay a portfolio of IDX stocks using the allocations below.')
            with st.form('sim3_form', border=False), st.container(key='sim-fields-sim3'):
                # The editor's data is the last submission. With num_rows='dynamic' its widget
                # identity includes that data, so storing a submission with different rows
                # changes the editor's identity on the next run and drops its old edit state
                # (edits are never applied twice). The browser, though, still shows the editor
                # from the run where Run was pressed, so the submit handler below reruns at
                # once when the rows changed; otherwise the next edits would go into that stale
                # editor and be discarded on the following Run.
                sim3_allocations = st.data_editor(
                    allocations_frame(submitted['rows']),
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
                    height=ALLOCATION_EDITOR_HEIGHT,
                    hide_index=True,
                    key='sim3_stock_editor'
                )
                start_col, end_col = st.columns(2, gap='small')
                start = start_col.number_input(
                    label='Start year', value=submitted['start'], min_value=YEAR_MIN,
                    max_value=YEAR_MAX, key='sim3_start',
                )
                end = end_col.number_input(
                    label='End year', value=submitted['end'], min_value=YEAR_MIN,
                    max_value=YEAR_MAX, key='sim3_end',
                )
                drip = st.toggle(
                    'Enable DRIP', value=submitted['drip'], key='sim3_drip',
                    help='Show results for reinvesting dividends into more shares',
                )
                if run_button():
                    previous_rows = submitted['rows']
                    submitted = portfolio_submission(sim3_allocations, start, end, drip)
                    st.session_state[PORTFOLIO_STATE_KEY] = submitted
                    if submitted['rows'] != previous_rows:
                        st.rerun()
            note(
                'Each position starts with whole 100-share lots; dividends are either '
                'reinvested in the stock that paid them (bought on the payment date) or '
                'kept as cash. The comparison includes historical price changes and '
                'uninvested cash.'
            )

    with right:
        sim3_result = None
        allocations = {}
        try:
            allocations = hs.build_allocations(allocation_rows(submitted['rows']))
            sim3_result = simulate_real_multistock_compounding(
                tuple(sorted(allocations.items())), submitted['start'], submitted['end']
            )
        except hs.SimulatorValidationError as e:
            st.error(str(e))
        except Exception:
            logger.exception(f'Unexpected error on sim3 for stocks {list(allocations)}')
            st.error('The market data service could not complete this simulation.')

        if sim3_result is None:
            return

        aggregate_df, combined_plot_df, transactions = sim3_result

        sim3_strategy = hs.STRATEGY_DRIP if submitted['drip'] else hs.STRATEGY_NO_DRIP
        display_df = aggregate_df[aggregate_df['strategy'] == sim3_strategy]
        starting_capital = sum(allocations.values())

        final_val = display_df['investment'].iloc[-1]
        final_div = display_df['returns'].iloc[-1]
        total_div = display_df['returns'].sum()
        final_by_strategy = {
            strategy: aggregate_df[aggregate_df['strategy'] == strategy]['investment'].iloc[-1]
            for strategy in (hs.STRATEGY_DRIP, hs.STRATEGY_NO_DRIP)
        }

        ui.kpi_row([
            ui.kpi('Final aggregate value', f'IDR {final_val:,.0f}'),
            ui.kpi('Total dividend collected', f'IDR {total_div:,.0f}'),
            ui.kpi('Yield on cost (final)', f'{(final_div/starting_capital*100):.2f}%'),
            drip_advantage_kpi(
                final_by_strategy[hs.STRATEGY_DRIP], final_by_strategy[hs.STRATEGY_NO_DRIP],
            ),
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
        ).properties(height=DRIP_CHART_HEIGHT)

        with ui.card('sim3-chart'):
            card_heading('Portfolio: DRIP comparison')
            st.altair_chart(
                (investment_chart + return_chart).resolve_scale(
                    y='independent', color='independent'
                ),
                width="stretch",
                theme=None,
            )

        table_card(
            'sim3-table',
            display_df[['year', 'investment', 'returns']],
            {
                'year': st.column_config.TextColumn('Year'),
                'investment': st.column_config.NumberColumn('Total Value', format='IDR %,d'),
                'returns': st.column_config.NumberColumn('Total Div.', format='IDR %,d'),
            },
            df_to_csv(aggregate_df[EXPORT_COLUMNS]),
            'multi_stock_historical_sim.csv',
        )

        log_card('sim3-log', 'Historical Transaction Log', transactions)


RENDERERS = {
    'fixed': render_fixed,
    'stock': render_stock,
    'portfolio': render_portfolio,
}

################################################################################

# Page-only sizing so the assumptions, tiles and chart fit the first screen (1440x800). Selectors
# use this page's own keys (sim cards, panes, sim_header), so the calendar page is untouched. The
# main-block prefix outranks the shared card and tile rules in page_ui, injected after this block.
st.html("""
<style>
    [data-testid="stMainBlockContainer"] { padding-top: 3.5rem; }
    .st-key-sim_header h1 { font-size: 1.75rem; padding: 0; }
    /* The shared -.6rem pull-up assumes a full-size h1 and overlaps this one */
    .st-key-sim_header .pdx .pdx-sub { margin-top: 4px; }
    [data-testid="stMainBlockContainer"] [class*="st-key-pdx-card-sim"] { padding: 14px 16px; gap: .6rem; }
    [data-testid="stMainBlockContainer"] [class*="st-key-sim-fields-"] { gap: .6rem; }
    [data-testid="stMainBlockContainer"] [class*="st-key-pdx-pane-right-sim"] { gap: .75rem; }
    /* One row of four tiles in the results pane; they still wrap when the pane is narrow */
    [data-testid="stMainBlockContainer"] [class*="st-key-pdx-pane-right-sim"] .pdx-kpis {
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px;
    }
    [data-testid="stMainBlockContainer"] [class*="st-key-pdx-pane-right-sim"] .pdx-kpi { padding: 10px 11px; }
    /* Amounts may wrap after "IDR" but never inside the number */
    [data-testid="stMainBlockContainer"] [class*="st-key-pdx-pane-right-sim"] .pdx-kpi .pdx-kpi-value {
      font-size: 15px; overflow-wrap: normal;
    }
</style>
""")
ui.inject_styles()

with st.container(key='sim_header'):
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
