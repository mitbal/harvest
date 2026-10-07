"""Track record: what following each published trading card would have returned.

Reads ``trading_outcomes``, written nightly by ``pipeline/trading_track.py`` (rules in
``harvest/track_record.py``). Settled results are open to everyone; open trades are
today's positions, so their names are for supporters.
"""
import logging

import altair as alt
import pandas as pd
import streamlit as st

import harvest.plot as hp
import harvest.track_record as tr
import harvest.trading_store as store
import harvest.trading_ui as ui
from harvest import chart_style as cs
from harvest.premium import get_connection, has_access


# Options are the labels, because the bound URL value is the displayed one (?tool=Swing+Trading).
TOOLS = {'Dividend Play': 'dividend_play', 'Swing Trading': 'swing', 'Long Term': 'long_term'}
METHOD = {
    'dividend_play': (
        'Each card is a limit order at its "entry at or below" price, rounded down to an IDX tick, for the '
        'session after it was published. If that day\'s low reaches the limit, the trade fills at the open or '
        'the limit, whichever is lower. Otherwise the next day\'s card is tried while the stock stays in its buy '
        'zone; a streak that ends unfilled counts as "not filled". A filled trade is sold at the close on its '
        'sell-by date, or earlier on the setup\'s take profit or stop loss. Results are after a 0.15% buy fee, '
        '0.25% sell fee and tax, and 10% tax on a dividend received while holding.'
    ),
    'swing': (
        'Each card is bought at the next session\'s open. The stop and target are the card\'s percentages of '
        'that fill: a stop or target inside a day fills at its level, a gap through it fills at the open, and '
        'the stop wins when both are touched. Otherwise the trade is sold at the close after the maximum hold. '
        'Results are after a 0.15% buy fee, 0.25% sell fee and tax, and half an IDX tick of slippage each way, '
        'as in the backtest.'
    ),
    'long_term': (
        'Each company is bought at the close of the first day it appears on the screen and closed when the '
        'price reaches the target on that first card (its mean valuation). Picks that have not reached it are '
        'still open and marked to the latest close. Price returns after buy and sell fees, without dividends, '
        'against the JKSE price index over the same days. The growth-based exit is not replayed.'
    ),
}
SIMULATED = {
    'dividend_play': 'the published shortlist-wide setup on every shortlist stock, over the held-out ex-dates '
                     'the search never saw (stock-specific setups are left out because they were picked on '
                     'those same trades)',
    'swing': 'the walk-forward record: in each half year, the setup chosen on the three years before it',
}
logger = logging.getLogger(__name__)

st.set_page_config(page_title='Track Record - Panen Dividen')
hp.enable_chart_theme()
ui.inject_styles()
ui.hero('Trading · Track record', 'Track record',
        'What following every published card would have returned, after costs. Losing trades and orders that '
        'never filled count too.', 'fact_check')

if st.session_state.get('tool') not in TOOLS:
    st.session_state['tool'] = next(iter(TOOLS))
label = st.segmented_control('Tool', list(TOOLS), required=True, key='tool', bind='query-params',
                             label_visibility='collapsed')
tool = TOOLS[label]


@st.cache_data(ttl=10 * 60, show_spinner=False)
def _load(tool: str) -> dict:
    client = get_connection()
    return {
        'outcomes': store.read_outcomes(client, tool),
        'run': store.latest_run(client, tool, 'track'),
        # The weekly search stores its portfolio simulation in the run stats.
        'optimize': store.latest_run(client, tool, 'optimize') if tool in SIMULATED else None,
    }


def load(tool: str) -> dict | None:
    try:
        return _load(tool)
    except Exception:
        logger.exception('Could not load the %s track record', tool)
        return None


def _num(values) -> pd.Series:
    """Numbers from stored JSON values, missing ones as NaN (an all-missing column stays numeric)."""
    return pd.to_numeric(pd.Series(values), errors='coerce').astype(float)


def _pct(value, signed: bool = True) -> str:
    if value is None or pd.isna(value):
        return '–'
    return f'{value:+.1f}%' if signed else f'{value:.0f}%'


def status_pill(run: dict | None, outcomes: pd.DataFrame) -> None:
    parts = []
    if run is not None:
        parts.append(f"Updated {ui.day(run['as_of'])} after close")
    first = ((run or {}).get('stats') or {}).get('first_card')
    if first or not outcomes.empty:
        start = first or outcomes['signal_date'].min()
        parts.append(f'Cards since {ui.day(start)} {pd.Timestamp(start).year}')
    if parts:
        ui.render(f'<span class="hv-status"><span class="dot"></span>{ui.esc(" · ".join(parts))}</span>')


def results_chart(closed: pd.DataFrame) -> None:
    """One bar per closed trade, oldest exit first; position and colour both carry win or loss."""
    data = closed.sort_values(['exit_date', 'symbol']).reset_index(drop=True)
    data = pd.DataFrame({
        'Trade': range(1, len(data) + 1),
        'Stock': data['symbol'].map(ui.short),
        'Sold': pd.to_datetime(data['exit_date']),
        'Result': _num(data['return_pct']),
        'Outcome': (_num(data['return_pct']) > 0).map({True: 'Won', False: 'Lost'}),
        'Reason': data['exit_reason'],
    })
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X('Trade:O', title='Closed trades, oldest first', axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y('Result:Q', title='Result after costs (%)'),
        color=alt.Color('Outcome:N', scale=alt.Scale(domain=['Won', 'Lost'],
                                                     range=[cs.CHART_POSITIVE, cs.CHART_NEGATIVE]),
                        legend=alt.Legend(title=None, orient='top')),
        tooltip=['Stock:N', alt.Tooltip('Sold:T', format='%a %-d %b %Y'),
                 alt.Tooltip('Result:Q', format='+.2f', title='Result (%)'), 'Reason:N'],
    )
    zero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color=cs.CHART_NEUTRAL).encode(y='y:Q')
    st.altair_chart((zero + bars).properties(height=260), width='stretch', theme=None)


def trades_table(rows: pd.DataFrame, long_term: bool = False) -> None:
    open_ = rows['status'] == 'open'
    last = rows['detail'].map(lambda d: (d or {}).get('last_close'))
    table = pd.DataFrame({
        'Stock': rows['symbol'].map(ui.short),
        'First card': pd.to_datetime(rows['signal_date']),
        'Bought': pd.to_datetime(rows['entry_date']),
        'Buy price': _num(rows['entry_price']),
        'Sold': pd.to_datetime(rows['exit_date']),
        'Sell price': _num(rows['exit_price']).where(~open_, _num(last)),
        'Result': _num(rows['return_pct']),
        'JKSE': _num(rows['benchmark_pct']),
        'Status': rows['exit_reason'].where(~open_, 'Open, at the latest close'),
    })
    if table['JKSE'].isna().all():
        table = table.drop(columns='JKSE')
    if long_term:
        table = table.drop(columns='First card').rename(columns={'Bought': 'Listed', 'Buy price': 'Price then',
                                                                'Sell price': 'Exit or now'})
    st.dataframe(
        table.iloc[::-1], hide_index=True, width='stretch',
        column_config={
            **{c: st.column_config.DateColumn(format='D MMM YYYY')
               for c in ('First card', 'Bought', 'Sold', 'Listed') if c in table},
            **{c: st.column_config.NumberColumn(format='%,.0f')
               for c in ('Buy price', 'Sell price', 'Price then', 'Exit or now') if c in table},
            'Result': st.column_config.NumberColumn(format='%+.2f%%', help='After costs'),
            'JKSE': st.column_config.NumberColumn(format='%+.2f%%', help='The JKSE price index over the same days'),
        },
    )


def render_trades(outcomes: pd.DataFrame, access: bool) -> None:
    """Dividend Play and Swing: closed trades, then open ones and unfilled cards."""
    summary = tr.summarize(outcomes)
    metrics = st.container(horizontal=True)
    metrics.metric('Closed trades', summary['closed'])
    metrics.metric('Won', f"{summary['wins']} of {summary['closed']}" if summary['closed'] else '–',
                   delta=_pct(summary['win_rate'], signed=False) if summary['closed'] else None,
                   delta_color='off', delta_arrow='off')
    metrics.metric('Average per trade', _pct(summary['avg_return']), help='Closed trades, after costs')
    metrics.metric('Backtest said', _pct(summary['expected']),
                   help='Average holdout result the same cards showed when they were published')
    metrics.metric('JKSE, same days', _pct(summary['avg_benchmark']),
                   help='Average JKSE price change over each closed trade\'s holding days')

    closed = outcomes[outcomes['status'] == 'closed']
    ui.section('Closed trades', len(closed))
    if closed.empty:
        ui.empty_state('No closed trades yet', 'Results appear here as published trades reach their sell-by date.',
                       'hourglass_empty')
    else:
        results_chart(closed)
        trades_table(closed)

    held = outcomes[outcomes['status'] == 'open']
    ui.section('Open trades', len(held), 'Marked to the latest close')
    if held.empty:
        st.caption('No open trades.')
    elif access:
        trades_table(held)
    else:
        many = len(held) != 1
        st.caption(f"{len(held)} {'trades' if many else 'trade'} from recent cards {'are' if many else 'is'} still "
                   f"open. Supporters see which {'ones' if many else 'one'}; everyone sees the result once it closes.")

    missed = outcomes[outcomes['status'] == 'not_filled']
    if not missed.empty:
        ui.section('Not filled', len(missed), 'The price never reached the limit')
        st.dataframe(
            pd.DataFrame({
                'Stock': missed['symbol'].map(ui.short),
                'First card': pd.to_datetime(missed['signal_date']),
                'Cards': missed['detail'].map(lambda d: (d or {}).get('cards')),
                'Last limit': _num(missed['detail'].map(lambda d: (d or {}).get('limit'))),
            }).iloc[::-1],
            hide_index=True, width='stretch',
            column_config={'First card': st.column_config.DateColumn(format='D MMM YYYY'),
                           'Last limit': st.column_config.NumberColumn(format='%,.0f')},
        )


def render_long_term(outcomes: pd.DataFrame, access: bool) -> None:
    """Long Term: every pick since it was listed, against the JKSE over the same days."""
    returns = _num(outcomes['return_pct'])
    benchmark = _num(outcomes['benchmark_pct'])
    compared = returns[benchmark.notna()] - benchmark[benchmark.notna()]
    closed = outcomes[outcomes['status'] == 'closed']
    metrics = st.container(horizontal=True)
    metrics.metric('Companies picked', len(outcomes))
    metrics.metric('Average since listed', _pct(returns.mean() if len(returns) else None),
                   help='Open picks at the latest close, closed picks at their exit, after fees')
    metrics.metric('JKSE, same days', _pct(benchmark.mean() if benchmark.notna().any() else None))
    metrics.metric('Ahead of the JKSE', f'{int((compared > 0).sum())} of {len(compared)}' if len(compared) else '–')
    metrics.metric('Reached target', len(closed))

    ui.section('Picks', len(outcomes), 'Newest first')
    shown = outcomes if access else closed
    if shown.empty:
        if access:
            ui.empty_state('No picks yet', 'Companies appear here from the first day the screen lists them.',
                           'hourglass_empty')
        else:
            st.caption('Every pick is still open. Supporters see which companies; everyone sees a pick '
                       'once it reaches its target.')
        return
    trades_table(shown, long_term=True)
    if not access and len(outcomes) > len(closed):
        hidden = len(outcomes) - len(closed)
        st.caption(f"{hidden} open {'picks are' if hidden != 1 else 'pick is'} for supporters.")


def growth_chart(curve: list[dict]) -> None:
    """The account and the JKSE as growth of 100, one line each, with a shared hover rule."""
    frame = pd.DataFrame(curve)
    frame['date'] = pd.to_datetime(frame['date'])
    series = {'account': 'Account', 'jkse': 'JKSE'}
    long = frame.melt('date', [c for c in series if c in frame], var_name='Series', value_name='Value').dropna()
    long['Series'] = long['Series'].map(series)
    domain = [series[c] for c in series if c in frame]
    hover = alt.selection_point(fields=['date'], nearest=True, on='pointerover', empty=False)
    base = alt.Chart(long).encode(
        x=alt.X('date:T', title=None),
        y=alt.Y('Value:Q', title='Growth of 100', scale=alt.Scale(zero=False)),
        color=alt.Color('Series:N', scale=alt.Scale(domain=domain, range=[cs.CHART_PRIMARY, cs.CHART_AMBER][:len(domain)]),
                        # One series needs no legend: the axis title names it.
                        legend=alt.Legend(title=None, orient='top') if len(domain) > 1 else None),
    )
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_point(size=60, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip('date:T', format='%-d %b %Y', title='Week of'), 'Series:N',
                 alt.Tooltip('Value:Q', format='.1f')],
    ).add_params(hover)
    rule = alt.Chart(long).mark_rule(color=cs.CHART_GRID).encode(x='date:T').transform_filter(hover)
    st.altair_chart((lines + rule + points).properties(height=280), width='stretch', theme=None)


def render_simulation(tool: str, optimize_run: dict | None) -> None:
    """The backtest's trades taken as one account with fixed capital, slots, lots and a liquidity cap."""
    portfolio = ((optimize_run or {}).get('stats') or {}).get('portfolio')
    ui.section('Backtest as one account', note='Out-of-sample trades, not live results')
    if not portfolio or not portfolio.get('stats'):
        st.caption('The portfolio simulation appears after the next weekly search.'
                   if not (portfolio or {}).get('error') else 'The portfolio simulation failed in the last weekly search.')
        return
    settings, stats = portfolio['settings'], portfolio['stats']
    ui.render(ui.strip(
        f"Rp {settings['capital'] / 1e6:,.0f} juta, up to <b>{settings['max_positions']} positions</b> of equal size, "
        f"{settings['lot_size']}-share lots, each at most {settings['liquidity_pct']:g}% of the stock's average "
        f"daily trading. {ui.esc(ui.day(stats['start']))} {pd.Timestamp(stats['start']).year} to "
        f"{ui.esc(ui.day(stats['end']))} {pd.Timestamp(stats['end']).year}.",
        icon_name='account_balance_wallet',
    ))
    bench = stats.get('benchmark') or {}
    metrics = st.container(horizontal=True)
    metrics.metric('Account a year', _pct(stats.get('cagr')), help='Compound annual growth after costs')
    metrics.metric('JKSE a year', _pct(bench.get('cagr')), help='The JKSE price index over the same period')
    metrics.metric('Worst fall', _pct(stats.get('max_drawdown')), help='Largest drop from a previous high')
    metrics.metric('Trades taken', f"{stats['trades_taken']} of {stats['trades_offered']}",
                   help='The rest were skipped: every slot taken, the stock already held, or too small for one lot')
    if portfolio.get('curve'):
        growth_chart(portfolio['curve'])
    skipped = stats['skipped']
    st.caption(
        f"Skipped: {skipped['full']} with every slot taken, {skipped['held']} with the stock already held, "
        f"{skipped['too_small']} too small or illiquid for one lot. Cut to the liquidity limit: "
        f"{stats['liquidity_capped']}. On average {stats['avg_positions']:.1f} positions were open and "
        f"{stats['avg_invested_pct']:.0f}% of the account invested. The trades are {SIMULATED[tool]}."
    )


published = load(tool)
if published is None:
    st.warning('The track record is unavailable right now. Please try again in a few minutes.')
    st.stop()
outcomes, run = published['outcomes'], published['run']
access = has_access()
if run is None and outcomes.empty:
    ui.empty_state('The track record starts after the first nightly update',
                   'Every card published from then on is counted here, winners and losers alike.', 'hourglass_empty')
else:
    status_pill(run, outcomes)
    if tool == 'long_term':
        render_long_term(outcomes, access)
    else:
        render_trades(outcomes, access)
if tool in SIMULATED:
    render_simulation(tool, published['optimize'])

with st.expander('How results are counted', icon=':material/info:'):
    st.markdown(METHOD[tool])
    st.markdown('One position per stock at a time, as in the backtests: cards for a stock already held belong to '
                'that trade. Settled results are stored once and never revised.')
st.caption('Past results do not guarantee future returns. Not financial advice.')
