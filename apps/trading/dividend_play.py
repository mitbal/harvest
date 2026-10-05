import concurrent.futures
import datetime
import hashlib
import logging
import io
import json
import os
import pickle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from st_supabase_connection import SupabaseConnection
from zoneinfo import ZoneInfo

import harvest.data as hd
import harvest.dividend_play as dp
import harvest.plot as hp
import harvest.prices as hprices
import harvest.swing as sw
import harvest.trading_store as store
from harvest import chart_style as cs
from harvest.dividend_play import (
    BASELINE_SETUP,
    DEFAULT_BUY_FEE,
    DEFAULT_DIVIDEND_TAX,
    DEFAULT_SELL_FEE,
    DEFAULT_SELL_TAX,
    MIN_PROFIT_FACTOR,
    MIN_STOCK_VALIDATION_TRADES,
    MIN_STOCK_VALIDATION_WIN_RATE,
    MIN_T_STAT,
    PRICE_WARMUP_DAYS,
    SEARCH_PRESETS,
)
from harvest.premium import current_email, get_connection, has_access, require_premium


FETCH_WORKERS = 8


VIEW_TODAY, VIEW_EVIDENCE, VIEW_LAB = 'Today', 'Evidence', 'Lab'
# The one stock whose evidence is open to everyone.
FREE_EVIDENCE_TICKER = 'BBRI.JK'
SOON_METRIC_DAYS = 14
WIB = ZoneInfo('Asia/Jakarta')
logger = logging.getLogger(__name__)

st.set_page_config(page_title='Dividend Play - Panen Dividen')
hp.enable_chart_theme()
st.title('Dividend Play')
# Initial values go through session state rather than `default`, because buttons below
# also set them; a ?view= or ?stock= in the URL still wins.
if 'view' not in st.session_state:
    # ?stock=SYMBOL opens the evidence for that stock.
    st.session_state['view'] = VIEW_EVIDENCE if 'stock' in st.query_params else VIEW_TODAY
view = st.segmented_control(
    'View', [VIEW_TODAY, VIEW_EVIDENCE, VIEW_LAB],
    required=True, key='view', bind='query-params', label_visibility='collapsed',
)


@st.cache_data(max_entries=64)
def df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


st.sidebar.markdown(
    """
    **Dividend play** buys a dividend stock while it is cheap a few months before
    its expected ex-date, then sells into the run-up as investors buy to qualify
    for the payout.

    * **Entry:** inside a window of days before the ex-date, either at the window
      start or on the first dip below a moving average.
    * **Exit:** before the ex-date (price run-up only) or a few days after it
      (collect the dividend, absorb the ex-date drop), with optional take-profit
      and stop-loss.
    * **Validation:** every historical ex-date is one trade; the most recent
      ex-dates are held out to check the setup still works.

    Shortlist dividend payers, search for the most reliable setup, inspect a
    single ticker, and scan for upcoming entry windows.
    """
)


@st.cache_resource(show_spinner=False)
def get_db_connection() -> SupabaseConnection:
    conn = st.connection("supabase", type=SupabaseConnection)
    try:
        conn.auth.sign_in_with_password({
            "email": st.secrets["connections"]["supabase"]["EMAIL_ADDRESS"],
            "password": st.secrets["connections"]["supabase"]["PASSWORD"],
        })
    except Exception:
        pass
    return conn


conn = get_db_connection()


def _read_redis_frame(key: str) -> pd.DataFrame | None:
    url = os.getenv('REDIS_URL')
    if not url:
        return None
    try:
        import redis
        value = redis.from_url(url, socket_connect_timeout=10, socket_timeout=30).get(key)
        if value is None:
            return None
        if isinstance(value, bytes) and value.startswith(b'PAR1'):
            return pd.read_parquet(io.BytesIO(value))
        decoded = json.loads(value)
        if isinstance(decoded, dict) and 'content' in decoded:
            decoded = json.loads(decoded['content'])
        return pd.DataFrame(decoded)
    except Exception:
        return None


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_dividend_universe() -> pd.DataFrame:
    """Dividend statistics per IDX stock: yield, paying years, size and liquidity."""
    frame = _read_redis_frame('div_score_jkse')
    if frame is None or frame.empty:
        try:
            frame = pd.read_csv('data/jkse/div_score.csv')
        except Exception:
            return pd.DataFrame()
    frame = frame.rename(columns={'symbol': 'stock', 'index': 'stock'})
    if 'stock' not in frame.columns:
        frame = frame.reset_index().rename(columns={frame.index.name or 'index': 'stock'})
    if not {'companyName', 'volAvg'}.issubset(frame.columns):
        try:
            profiles = pd.read_csv('data/jkse/company_profiles.csv')
            missing = [column for column in ('companyName', 'volAvg') if column not in frame.columns]
            frame = frame.merge(
                profiles[['symbol', *missing]].rename(columns={'symbol': 'stock'}), on='stock', how='left',
            )
        except Exception:
            pass
    for column in ('price', 'yield', 'mktCap', 'numDividendYear', 'volAvg', 'DScore'):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
    return frame.drop_duplicates('stock')


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_dividend_events() -> dict[str, pd.DataFrame]:
    """Ex-dividend dates and amounts per symbol.

    The pipeline publishes this pickle to Supabase storage; the local copy is a
    fallback for development.
    """
    path = 'data/jkse/dividends.pkl'
    raw = None
    try:
        raw = pickle.loads(conn.client.storage.from_('harvest_dividend').download(path))
    except Exception:
        try:
            raw = pd.read_pickle(path)
        except Exception:
            return {}

    events = {}
    for symbol, records in (raw or {}).items():
        frame = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records
        frame = dp.prepare_dividends(frame)
        if not frame.empty:
            events[symbol] = frame
    return events


@st.cache_data(ttl=6 * 60 * 60, max_entries=32, show_spinner=False)
def load_stored_prices(symbols: tuple[str, ...], start: str) -> pd.DataFrame:
    """Daily closes for ``symbols`` since ``start`` from the Parquet price history."""
    return hprices.load_prices('jkse', symbols=symbols, start=start)


def fetch_live_prices(symbol: str, start: str) -> pd.DataFrame:
    """Daily closes from the live API, for stocks missing from the stored history."""
    try:
        return dp.prepare_prices(hd.get_daily_stock_price(symbol, start_from=start))
    except Exception:
        return pd.DataFrame(columns=['date', 'close'])


def fetch_price_history(symbol: str, start: str) -> pd.DataFrame:
    """Daily closes from the stored price history, then the live API."""
    prices = dp.prepare_prices(load_stored_prices((symbol,), start))
    if not prices.empty:
        return prices
    return fetch_live_prices(symbol, start)


def fetch_price_histories(symbols: list[str], start: str, label: str) -> dict[str, pd.DataFrame]:
    stored = load_stored_prices(tuple(sorted(set(symbols))), start)
    stored_by_symbol = {symbol: dp.prepare_prices(group) for symbol, group in stored.groupby('symbol')}
    histories = {
        symbol: stored_by_symbol[symbol]
        for symbol in symbols
        if symbol in stored_by_symbol and not stored_by_symbol[symbol].empty
    }
    missing = [symbol for symbol in symbols if symbol not in histories]
    if not missing:
        return histories

    progress = st.progress(0.0, text=label)
    with concurrent.futures.ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        futures = {executor.submit(fetch_live_prices, symbol, start): symbol for symbol in missing}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            symbol = futures[future]
            try:
                prices = future.result()
            except Exception:
                prices = pd.DataFrame()
            if not prices.empty:
                histories[symbol] = prices
            progress.progress(completed / len(futures), text=f"{label} {completed}/{len(futures)}")
    progress.empty()
    return histories


def _pct(value, digits=1, signed=True):
    if value is None or not np.isfinite(value):
        return 'N/A'
    return f'{value:+.{digits}f}%' if signed else f'{value:.{digits}f}%'



def render_backtest(stock: str, prices: pd.DataFrame, divs: pd.DataFrame, params: dict,
                    window_start: pd.Timestamp, window_end: pd.Timestamp) -> None:
    """Performance, price chart, typical run-up and trade log for one stock and setup."""
    if prices.empty:
        st.error("No historical price data found for the given stock and timeframe.")
    elif divs.empty:
        st.error("No dividend history found for this stock.")
    else:
        trades = dp.backtest_dividend_play(dp.play_data(prices, divs, window_start, window_end), **params)
        stats = dp.summarize_trades(trades)
        st.subheader('Performance Outline')
        if trades.empty:
            st.warning("No complete trades in this period. Try an earlier start date or a narrower entry window.")
        else:
            metrics = st.columns(6)
            metrics[0].metric("Win Rate", _pct(stats['win_rate'], 0, signed=False),
                              help=f"{stats['trades']} trades")
            metrics[1].metric("Average Return", _pct(stats['avg_return']), help="Per trade, net of costs")
            metrics[2].metric("Best Trade", _pct(stats['best']))
            metrics[3].metric("Worst Trade", _pct(stats['worst']))
            metrics[4].metric("Avg Drawdown", _pct(stats['avg_drawdown']),
                              help="Average worst mark-to-market loss while holding")
            metrics[5].metric("Compounded", _pct(stats['compounded']),
                              help="Growth of capital when every trade is taken in sequence")
            st.caption(
                f"Average holding {stats['avg_days']:.0f} days | "
                f"{int(trades['dip_triggered'].sum())} of {len(trades)} entries on a dip | "
                f"return per month held {stats['return_per_month']:+.2f}% | "
                f"compounded CAGR {stats['cagr']:+.1f}%"
            )

        chart_prices = prices[(prices['date'] >= window_start) & (prices['date'] <= window_end)]
        layers = [
            alt.Chart(chart_prices).mark_line(color=cs.CHART_PRIMARY, strokeWidth=1.6).encode(
                x=alt.X('date:T', title=None),
                y=alt.Y('close:Q', title='Close', scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('close:Q', title='Close', format=',.0f')],
            )
        ]
        if not trades.empty:
            windows = trades[['ex_date']].assign(
                start=trades['ex_date'] - pd.Timedelta(days=params['entry_start_days']),
                end=trades['ex_date'] - pd.Timedelta(days=params['entry_end_days']),
            )
            layers.insert(0, alt.Chart(windows).mark_rect(color=cs.CHART_AMBER, opacity=0.12).encode(
                x='start:T', x2='end:T',
            ))
            layers.append(alt.Chart(trades).mark_rule(color=cs.CHART_NEUTRAL, strokeDash=[4, 4]).encode(
                x='ex_date:T',
                tooltip=[alt.Tooltip('ex_date:T', title='Ex-date'),
                         alt.Tooltip('dividend:Q', title='Dividend', format=',.2f')],
            ))
            markers = pd.concat([
                trades[['entry_date', 'entry_price']].set_axis(['date', 'price'], axis=1).assign(Action='Buy'),
                trades[['exit_date', 'exit_price', 'total_return']]
                .set_axis(['date', 'price', 'total_return'], axis=1).assign(Action='Sell'),
            ], ignore_index=True)
            layers.append(alt.Chart(markers).mark_point(filled=True, size=70, opacity=1).encode(
                x='date:T', y='price:Q',
                shape=alt.Shape('Action:N', scale=alt.Scale(domain=['Buy', 'Sell'], range=['triangle-up', 'triangle-down'])),
                color=alt.Color('Action:N', scale=alt.Scale(domain=['Buy', 'Sell'], range=[cs.CHART_POSITIVE, cs.CHART_NEGATIVE]),
                                legend=alt.Legend(title=None, orient='top')),
                tooltip=[alt.Tooltip('Action:N'), alt.Tooltip('date:T', title='Date'),
                         alt.Tooltip('price:Q', title='Price', format=',.0f'),
                         alt.Tooltip('total_return:Q', title='Trade return %', format='+.1f')],
            ))

        upcoming = dp.estimate_next_ex_dates(divs, as_of=pd.Timestamp(datetime.date.today()))
        if not upcoming.empty:
            nxt = upcoming.iloc[0]
            next_window = pd.DataFrame({
                'start': [nxt['expected_date'] - pd.Timedelta(days=params['entry_start_days'])],
                'end': [nxt['expected_date'] - pd.Timedelta(days=params['entry_end_days'])],
                'ex': [nxt['expected_date']],
            })
            layers.insert(0, alt.Chart(next_window).mark_rect(color=cs.CHART_BLUE, opacity=0.12).encode(
                x='start:T', x2='end:T',
                tooltip=[alt.Tooltip('start:T', title='Next window opens'),
                         alt.Tooltip('end:T', title='Window closes'),
                         alt.Tooltip('ex:T', title='Expected ex-date')],
            ))

        price_tab, path_tab, trade_tab = st.tabs(['Price and Trades', 'Typical Run-up', 'Trade Log'])
        with price_tab:
            st.altair_chart(alt.layer(*layers).properties(height=380), width='stretch', theme=None)
            st.caption(
                'Amber bands are historical entry windows, the blue band is the next expected window, dashed '
                'lines mark ex-dates, and markers show the simulated buys and sells.'
            )

        with path_tab:
            path_divs = divs[(divs['date'] >= window_start) & (divs['date'] <= window_end)]
            _, path_df, drop_stats = hd.calc_best_buy_timing(
                prices[prices['date'] <= window_end],
                path_divs.assign(date=path_divs['date'].dt.strftime('%Y-%m-%d')),
                pre_ex_days=max(180, params['entry_start_days'] + 10),
                post_ex_days=30,
            )
            if path_df is None or path_df.empty:
                st.info('Not enough dividend events to draw the typical price path.')
            else:
                band = alt.Chart(pd.DataFrame({
                    'start': [-params['entry_start_days']], 'end': [-params['entry_end_days']],
                })).mark_rect(color=cs.CHART_AMBER, opacity=0.14).encode(x='start:Q', x2='end:Q')
                iqr = alt.Chart(path_df).mark_area(color=cs.CHART_PRIMARY, opacity=0.12).encode(
                    x=alt.X('days_to_ex:Q', title='Days relative to ex-date'),
                    y=alt.Y('q25:Q', title='Price (ex-date = 100)', scale=alt.Scale(zero=False)),
                    y2='q75:Q',
                )
                median = alt.Chart(path_df).mark_line(color=cs.CHART_PRIMARY, strokeWidth=2.25).encode(
                    x='days_to_ex:Q', y='median:Q',
                    tooltip=[alt.Tooltip('days_to_ex:Q', title='Days to ex-date'),
                             alt.Tooltip('median:Q', title='Median', format='.1f'),
                             alt.Tooltip('q25:Q', title='25th pct', format='.1f'),
                             alt.Tooltip('q75:Q', title='75th pct', format='.1f')],
                )
                ex_rule = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(
                    color=cs.CHART_NEUTRAL, strokeDash=[4, 4]).encode(x='x:Q')
                st.altair_chart((band + iqr + median + ex_rule).properties(height=340), width='stretch', theme=None)

                pre = path_df[path_df['days_to_ex'] < 0]
                path_cols = st.columns(3)
                if not pre.empty:
                    low_row = pre.loc[pre['median'].idxmin()]
                    path_cols[0].metric('Typical Low', f"{abs(int(low_row['days_to_ex']))} days before ex",
                                        help='Day with the lowest median price before the ex-date')
                    path_cols[1].metric('Typical Run-up From Low',
                                        _pct((pre['median'].iloc[-1] / low_row['median'] - 1) * 100),
                                        help='Median path from its low to the day before the ex-date')
                if drop_stats:
                    path_cols[2].metric('Median Ex-date Drop', _pct(drop_stats['median']),
                                        help='Price change on the ex-date versus the prior close')
                st.caption(
                    'Every past ex-date is aligned at day 0 with its price set to 100. The line is the median '
                    'path, the shaded band the middle 50% of outcomes, and the amber band the entry window.'
                )

        with trade_tab:
            if trades.empty:
                st.info('No trades to show.')
            else:
                bars = trades.melt(
                    id_vars=['ex_date'], value_vars=['price_return', 'dividend_return'],
                    var_name='Component', value_name='Return',
                ).replace({'Component': {'price_return': 'Price', 'dividend_return': 'Dividend'}})
                st.altair_chart(
                    alt.Chart(bars).mark_bar().encode(
                        x=alt.X('yearmonthdate(ex_date):O', title='Ex-date', axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y('Return:Q', title='Net return (%)'),
                        color=alt.Color('Component:N', scale=alt.Scale(domain=['Price', 'Dividend'],
                                                                       range=[cs.CHART_PRIMARY, cs.CHART_AMBER]),
                                        legend=alt.Legend(title=None, orient='top')),
                        tooltip=[alt.Tooltip('ex_date:T', title='Ex-date'), 'Component:N',
                                 alt.Tooltip('Return:Q', format='+.2f')],
                    ).properties(height=280),
                    width='stretch', theme=None,
                )
                log = trades.sort_values('ex_date', ascending=False).copy()
                st.dataframe(
                    log,
                    column_config={
                        'ex_date': st.column_config.DateColumn('Ex-date'),
                        'dividend': st.column_config.NumberColumn('Dividend', format='%.2f'),
                        'entry_date': st.column_config.DateColumn('Bought'),
                        'entry_price': st.column_config.NumberColumn('Buy Price', format='%.0f'),
                        'dip_triggered': st.column_config.CheckboxColumn('Dip Hit'),
                        'exit_date': st.column_config.DateColumn('Sold'),
                        'exit_price': st.column_config.NumberColumn('Sell Price', format='%.0f'),
                        'exit_reason': st.column_config.TextColumn('Exit'),
                        'days_held': st.column_config.NumberColumn('Days'),
                        'collected_dividend': st.column_config.CheckboxColumn('Got Dividend'),
                        'price_return': st.column_config.NumberColumn('Price', format='%+.1f%%'),
                        'dividend_return': st.column_config.NumberColumn('Dividend', format='%+.1f%%'),
                        'total_return': st.column_config.NumberColumn('Total', format='%+.1f%%'),
                        'max_drawdown': st.column_config.NumberColumn('Max Drawdown', format='%+.1f%%'),
                    },
                    hide_index=True, width='stretch',
                )
                st.download_button(
                    'Download Trade Log CSV',
                    data=df_to_csv(log),
                    file_name=f"dividend_play_{stock.removesuffix('.JK')}.csv",
                    mime='text/csv',
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TODAY AND EVIDENCE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=10 * 60, show_spinner=False)
def _load_published() -> dict:
    client = get_connection()
    return {
        'signals': store.read_signals(client, 'dividend_play'),
        'run': store.latest_run(client, 'dividend_play', 'scan'),
        'setups': store.read_setups(client, 'dividend_play'),
        'stale': store.is_stale(client, 'dividend_play'),
    }


def load_published() -> dict | None:
    """Latest published signals, scan run and setups; None when the store is unreachable."""
    try:
        return _load_published()
    except Exception:
        logger.exception('Could not load published Dividend Play signals')
        return None


@st.cache_data(ttl=60, show_spinner=False)
def load_open_positions(email: str) -> pd.DataFrame:
    positions = store.list_positions(get_connection(), email, open_only=True)
    return positions[positions['tool'] == 'dividend_play']


def _short(symbol: str) -> str:
    return symbol.removesuffix('.JK')


def _day(value, estimated: bool = False) -> str:
    """'Mon 3 Nov', with '~' in front of estimated dates."""
    date = pd.Timestamp(value)
    return f"{'~' if estimated else ''}{date:%a} {date.day} {date:%b}"


def _idr(value) -> str:
    return f'Rp {value:,.0f}'


def _entry_level(price: float) -> float:
    """Round an 'at or below' price down to a valid IDX tick."""
    tick = float(sw.idx_tick_size([price])[0])
    return np.floor(price / tick) * tick


def _first_session(value) -> datetime.date:
    """The first IDX trading day on or after ``value``; windows are counted in calendar days."""
    day = pd.Timestamp(value)
    return store.idx_trading_days(day, day + pd.Timedelta(days=14))[0]


def _open_view(target: str, stock: str | None = None) -> None:
    st.session_state['view'] = target
    if stock:
        st.session_state['stock'] = stock


def _status_line(run: dict) -> None:
    as_of = pd.Timestamp(run['as_of'])
    finished = pd.Timestamp(run.get('finished_at') or run['started_at']).tz_convert(WIB)
    next_session = next(day for day in store.idx_trading_days(as_of + pd.Timedelta(days=1), as_of + pd.Timedelta(days=14)))
    st.caption(
        f"Updated {_day(as_of)}, {finished:%H:%M} WIB after close. "
        f"Next update {_day(next_session)} after close."
    )


def _holdout_text(holdout: dict) -> str:
    trades = holdout.get('trades') or 0
    if not trades:
        return 'no holdout trades'
    scope = 'this stock' if holdout.get('level') == 'stock' else 'all shortlist stocks'
    avg = holdout.get('avg_return')
    return (
        f"holdout {holdout.get('wins', 0)} of {trades} won on {scope}"
        + (f", {avg:+.1f}% average after costs" if avg is not None else '')
    )


def render_signal_card(signal: pd.Series) -> None:
    detail = signal['detail']
    estimated = bool(signal['key_date_estimated'])
    with st.container(border=True):
        st.markdown(f"**{_short(signal['symbol'])}** · {detail.get('company') or signal['symbol']}")
        facts = []
        if detail.get('yield') is not None:
            facts.append(f"yield {detail['yield']:.1f}%")
        if detail.get('next_dividend') is not None:
            facts.append(f"next dividend {_idr(detail['next_dividend'])}")
        if facts:
            line = ' · '.join(facts)
            st.caption(line[0].upper() + line[1:])
        if estimated:
            st.badge('Cum date estimated', icon=':material/event:', color='orange')
        else:
            st.badge('Cum date announced', icon=':material/event_available:', color='green')
        st.markdown(
            f"Entry at or below **{_idr(_entry_level(signal['entry_price']))}**  \n"
            f"Sell by **{_day(signal['exit_by'], estimated)}**  \n"
            f"Cum date **{_day(signal['key_date'], estimated)}**"
        )
        window = detail.get('window') or {}
        if window.get('of'):
            st.progress(
                min(window['day'] / window['of'], 1.0),
                text=f"Day {window['day']} of {window['of']} in window · {window['sessions_left']} sessions left",
            )
        dip = detail.get('dip') or {}
        if dip.get('rule') and dip.get('active') and dip.get('vs_ma_pct') is not None:
            st.caption(f"Dip rule active, {abs(dip['vs_ma_pct']):.1f}% below its {dip['ma_days']}-day average")
        st.caption(f"{detail.get('setup', '')}; {_holdout_text(detail.get('holdout') or {})}")
        st.button(
            'See evidence', key=f"evidence_{signal['symbol']}", icon=':material/query_stats:',
            on_click=_open_view, args=(VIEW_EVIDENCE, signal['symbol']),
        )


def render_unlock_prompt(hidden: int) -> None:
    with st.container(border=True):
        st.markdown(
            f"**{hidden} more {'setup' if hidden == 1 else 'setups'} today.** Supporters see every card, "
            "the evidence for any stock and the research lab."
        )
        st.button(
            'Unlock the trading lab', type='primary', icon=':material/lock_open:',
            on_click=_open_view, args=(VIEW_LAB,), key='today_unlock',
        )


def render_today(access: bool) -> None:
    published = load_published()
    if published is None:
        st.warning('Signals are unavailable right now. Please try again in a few minutes.')
        return
    run, signals = published['run'], published['signals']
    if run is None:
        st.info('Signals have not been published yet. The first update arrives after the next market close.')
        return
    _status_line(run)
    if published['stale']:
        st.warning("These signals missed at least one nightly update. They may be out of date.")

    actionable = signals[signals['status'] == 'actionable'].copy()
    actionable['expected'] = actionable['detail'].map(lambda d: (d.get('holdout') or {}).get('avg_return') or 0)
    actionable = actionable.sort_values('expected', ascending=False)
    soon = signals[signals['status'] == 'opening_soon'].copy()
    soon['days_to_window'] = soon['detail'].map(lambda d: d.get('days_to_window', 0))
    soon = soon.sort_values('days_to_window')

    metrics = st.container(horizontal=True)
    metrics.metric('Entry windows open', len(actionable))
    metrics.metric(f'Opening in {SOON_METRIC_DAYS} days', int((soon['days_to_window'] <= SOON_METRIC_DAYS).sum()))
    email = current_email()
    if email:
        try:
            positions = load_open_positions(email)
            next_exit = positions['exit_by'].dropna().min() if not positions.empty else None
            exiting = int((positions['exit_by'] == next_exit).sum()) if next_exit is not None else 0
            metrics.metric(
                'Your positions', len(positions),
                delta=f'{exiting} exits {_day(next_exit)}' if next_exit is not None else None,
                delta_color='off', delta_arrow='off',
            )
        except Exception:
            logger.exception('Could not load positions')
            metrics.metric('Your positions', '–')
    else:
        metrics.metric('Your positions', '–', help='Log in to track your trades')

    st.subheader('Entry window open now')
    if actionable.empty:
        st.info('No entry window is open today.')
    else:
        shown = actionable if access else actionable.head(1)
        for start in range(0, len(shown), 2):
            for column, (_, signal) in zip(st.columns(2), shown.iloc[start:start + 2].iterrows()):
                with column:
                    render_signal_card(signal)
        if not access and len(actionable) > 1:
            st.markdown('Also open: ' + ', '.join(_short(s) for s in actionable['symbol'].iloc[1:]))
            render_unlock_prompt(len(actionable) - 1)

    st.subheader('Opening soon')
    if soon.empty:
        st.info('No entry window opens in the next 45 days.')
    elif not access:
        st.markdown(', '.join(_short(s) for s in soon['symbol']))
    else:
        rows = pd.DataFrame({
            'Stock': soon['symbol'].map(_short),
            'Window opens': [
                _day(_first_session(d['window_start']), est)
                for d, est in zip(soon['detail'], soon['key_date_estimated'])
            ],
            'Historical run-up': soon['detail'].map(lambda d: (d.get('history') or {}).get('avg_return')),
            'Won': soon['detail'].map(lambda d: (d.get('history') or {}).get('win_rate')),
        })
        st.dataframe(
            rows, hide_index=True, width='stretch',
            column_config={
                'Window opens': st.column_config.TextColumn(help='~ means the ex-date is estimated from last year'),
                'Historical run-up': st.column_config.NumberColumn(
                    format='%+.1f%%', help='Average return of this setup on the stock since 2015, after costs'),
                'Won': st.column_config.NumberColumn(format='%.0f%%', help='Share of those trades that made money'),
            },
        )
    st.caption(
        'Historical backtest of past dividend events. Check the announced cum date with IDX before trading. '
        'Not financial advice.'
    )


def render_evidence(access: bool) -> None:
    events = load_dividend_events()
    if not events:
        st.warning('Dividend history is unavailable right now. Please try again in a few minutes.')
        return
    options = sorted(events)
    default = FREE_EVIDENCE_TICKER if FREE_EVIDENCE_TICKER in options else options[0]
    if st.session_state.get('stock') not in options:
        st.session_state['stock'] = default
    # No format_func: the bound URL parameter is the displayed value, and links use the full symbol.
    stock = st.selectbox('Stock', options, key='stock', bind='query-params')
    if not access and stock != FREE_EVIDENCE_TICKER:
        require_premium(teaser=lambda: st.info(
            f'The evidence for {_short(FREE_EVIDENCE_TICKER)} is free to explore. '
            'Supporters can open any stock.', icon=':material/info:',
        ))

    published = load_published()
    setups = published['setups'] if published else {}
    scope = stock if stock in setups else ('global' if 'global' in setups else None)
    if scope is None:
        setup = BASELINE_SETUP
        st.markdown(f'**{dp.format_setup(setup)}**')
        st.caption('No setup has been published yet, so this shows the baseline setup.')
    else:
        setup = setups[scope]['params']
        holdout = setups[scope]['holdout']
        st.markdown(f'**{dp.format_setup(setup)}**')
        source = 'Setup chosen for this stock' if scope == stock else 'Shortlist-wide setup'
        trades = holdout.get('trades') or 0
        wins = round((holdout.get('win_rate') or 0) * trades / 100)
        st.caption(
            f"{source}. Holdout {wins} of {trades} trades won, "
            f"{(holdout.get('avg_trade') or 0):+.1f}% average after costs."
        )

    start = pd.Timestamp(dp.DEFAULT_START)
    with st.spinner('Loading price history...'):
        prices = fetch_price_history(stock, (start - pd.Timedelta(days=PRICE_WARMUP_DAYS)).strftime('%Y-%m-%d'))
    render_backtest(
        stock, prices, events.get(stock, pd.DataFrame(columns=['date', 'dividend'])),
        {**setup, **dp.trade_costs()}, start, pd.Timestamp(datetime.date.today()),
    )


access = has_access()
if view == VIEW_TODAY:
    render_today(access)
elif view == VIEW_EVIDENCE:
    render_evidence(access)
else:
    require_premium()
    st.caption(
        "Research tools. Results here do not change the published signals on the Today view."
    )
    dividend_events = load_dividend_events()

    # ═══════════════════════════════════════════════════════════════════════════════
    # DIVIDEND STOCK SHORTLIST
    # ═══════════════════════════════════════════════════════════════════════════════

    with st.container():
        st.subheader("Dividend Stock Shortlist", divider='blue')
        filter_cols = st.columns(4)
        min_mcap_trill = filter_cols[0].number_input(
            "Min Market Cap (Trillion IDR)", min_value=0.0, value=0.5, step=0.5, key="dp_mcap",
        )
        min_yield = filter_cols[1].number_input(
            "Min Dividend Yield %", min_value=0.0, max_value=30.0, value=1.0, step=0.5, key="dp_min_yield",
        )
        min_div_years = filter_cols[2].number_input(
            "Min Years Paying Dividends", min_value=1, max_value=30, value=3, step=1, key="dp_min_years",
            help="Needs several ex-dates to learn the run-up pattern and hold some out for validation.",
        )
        max_stocks = filter_cols[3].number_input(
            "Max Stocks Output", min_value=10, max_value=500, value=200, step=10, key="dp_max_stocks",
        )
        rank_cols = st.columns(3)
        yield_weight = rank_cols[0].slider(
            "Yield Weight", min_value=0, max_value=100, value=50, step=5, key="dp_yield_weight",
            help="Higher values favor high yield; lower values favor a long, unbroken dividend record.",
        )
        min_value_traded_bn = rank_cols[1].number_input(
            "Min Avg Daily Value Traded (Billion IDR)", min_value=0.0, max_value=500.0,
            value=0.5, step=0.5, key="dp_min_value_traded",
            help="Average volume x price. Thin stocks cannot be exited at the modelled close.",
        )
        include_blue_chips = rank_cols[2].checkbox(
            "Pin Major Blue-chips", value=True, key="dp_blue",
            help="Always show major dividend-paying blue chips at the top of the list",
        )

        final_df = pd.DataFrame()
        universe_df = load_dividend_universe()
        if universe_df.empty:
            st.warning("Dividend universe data is unavailable.")
        else:
            final_df, eligible_count = dp.build_shortlist(
                universe_df, dividend_events,
                min_mcap_trill=min_mcap_trill, min_yield=min_yield, min_div_years=min_div_years,
                max_stocks=max_stocks, yield_weight=yield_weight,
                min_value_traded_bn=min_value_traded_bn, pin_blue_chips=include_blue_chips,
            )

            display_columns = [
                column for column in [
                    'stock', 'companyName', 'sector', 'price', 'yield', 'numDividendYear',
                    'mktCap', 'Avg Value Traded', 'Dividend Rank',
                ] if column in final_df.columns
            ]
            st.dataframe(
                final_df[display_columns],
                column_config={
                    "stock": st.column_config.TextColumn("Symbol"),
                    "companyName": st.column_config.TextColumn("Company"),
                    "sector": st.column_config.TextColumn("Sector"),
                    "price": st.column_config.NumberColumn("Price", format="%.0f"),
                    "yield": st.column_config.NumberColumn("Dividend Yield", format="%.2f%%"),
                    "numDividendYear": st.column_config.NumberColumn("Years Paid", format="%d"),
                    "mktCap": st.column_config.NumberColumn("Market Cap", format="compact"),
                    "Avg Value Traded": st.column_config.NumberColumn("Avg Value Traded", format="compact"),
                    "Dividend Rank": st.column_config.ProgressColumn(
                        "Dividend Rank", format="%.1f", min_value=0, max_value=100,
                    ),
                },
                hide_index=True,
                width='stretch',
            )
            st.caption(f"{len(final_df)} of {eligible_count} eligible dividend payers shown.")

    shortlist_tickers = final_df['stock'].tolist() if not final_df.empty else []


    # ═══════════════════════════════════════════════════════════════════════════════
    # SETUP OPTIMIZER
    # ═══════════════════════════════════════════════════════════════════════════════

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Find the Best Dividend Play Setup", divider='violet')
    st.caption(
        "Sweep entry windows, dip rules, and exits across the whole shortlist. Every historical ex-date is "
        "one trade. The most recent ex-dates are held out, and each setup is ranked by how consistently it "
        "made money on those unseen dividends."
    )

    search_cols = st.columns(2)
    search_depth = search_cols[0].selectbox(
        "Search Depth", options=list(SEARCH_PRESETS), index=1, key="dp_search_depth",
        help="Quick for iteration, Balanced for normal use, Deep for final research",
    )
    opt_validation_pct = search_cols[1].slider(
        "Holdout Data %", min_value=20, max_value=50, value=30, step=5, key="dp_validation_pct",
    )

    date_cols = st.columns(2)
    opt_start = date_cols[0].date_input("Start Date", value=datetime.date(2015, 1, 1), key="dp_start")
    opt_end = date_cols[1].date_input("End Date", value=datetime.date.today(), key="dp_end")

    with st.expander("Advanced: trading costs and dividend tax"):
        cost_cols = st.columns(4)
        opt_buy_fee = cost_cols[0].number_input(
            "Buy Fee %", min_value=0.0, max_value=5.0, value=DEFAULT_BUY_FEE, step=0.01, key="dp_buy_fee",
        )
        opt_sell_fee = cost_cols[1].number_input(
            "Sell Fee %", min_value=0.0, max_value=5.0, value=DEFAULT_SELL_FEE, step=0.01, key="dp_sell_fee",
        )
        opt_sell_tax = cost_cols[2].number_input(
            "Sell Tax %", min_value=0.0, max_value=5.0, value=DEFAULT_SELL_TAX, step=0.01, key="dp_sell_tax",
        )
        opt_dividend_tax = cost_cols[3].number_input(
            "Dividend Tax %", min_value=0.0, max_value=30.0, value=DEFAULT_DIVIDEND_TAX, step=1.0, key="dp_div_tax",
            help="Withheld from the dividend when a trade is held through the ex-date.",
        )

    search_space = SEARCH_PRESETS[search_depth]
    setups = dp.build_setup_grid(**search_space)
    total_runs = len(setups) * len(shortlist_tickers)
    split_date = dp.holdout_split_date(opt_start, opt_end, opt_validation_pct)
    opt_costs = dp.trade_costs(opt_buy_fee, opt_sell_fee, opt_sell_tax, opt_dividend_tax)

    info_cols = st.columns(5)
    info_cols[0].metric("Setups", f"{len(setups):,}")
    info_cols[1].metric("Stocks", f"{len(shortlist_tickers):,}")
    info_cols[2].metric("Backtest Runs", f"{total_runs:,}")
    info_cols[3].metric("Holdout", f"Latest {opt_validation_pct}%")
    info_cols[4].metric("Holdout Starts", split_date.strftime('%b %Y'))

    if total_runs > 150_000:
        st.warning(f"{total_runs:,} backtests may take a minute. Use Quick search while iterating.")

    run_optimizer = st.button("Find Best Setup", type="primary", disabled=total_runs == 0)

    price_start = (pd.Timestamp(opt_start) - pd.Timedelta(days=PRICE_WARMUP_DAYS)).strftime('%Y-%m-%d')
    price_cache_key = hashlib.md5(f"{sorted(shortlist_tickers)}{price_start}".encode()).hexdigest()
    results_cache_key = hashlib.md5(
        f"{setups}{opt_start}{opt_end}{shortlist_tickers}{opt_costs}{opt_validation_pct}".encode()
    ).hexdigest()


    def _play_data(symbol: str, prices: pd.DataFrame, start=None, end=None) -> dp.PlayData:
        """Build backtest arrays, keeping ex-dates within ``start``..``end``."""
        return dp.play_data(prices, dividend_events.get(symbol), start, end)


    if run_optimizer:
        st.session_state.pop('dp_optimizer_results', None)
        st.session_state.pop('dp_optimizer_results_cache_key', None)

        if st.session_state.get('dp_price_cache_key') == price_cache_key and 'dp_prices' in st.session_state:
            price_histories = st.session_state['dp_prices']
            st.info(f"Using cached price data for {len(price_histories)} stocks. Only setup parameters changed.")
        else:
            price_histories = fetch_price_histories(shortlist_tickers, price_start, "Fetching price history...")
            st.session_state['dp_prices'] = price_histories
            st.session_state['dp_price_cache_key'] = price_cache_key

        if not price_histories:
            st.error("No price data returned. Make sure the shortlist is non-empty and price history is available.")
        else:
            ex_dates_in_range = sum(
                int(events['date'].between(pd.Timestamp(opt_start), pd.Timestamp(opt_end)).sum())
                for symbol, events in dividend_events.items() if symbol in price_histories
            )
            fetch_cols = st.columns(3)
            fetch_cols[0].metric("Stocks with Prices", f"{len(price_histories)}")
            fetch_cols[1].metric("Expected Stocks", f"{len(shortlist_tickers)}")
            fetch_cols[2].metric("Ex-dates in Range", f"{ex_dates_in_range:,}")

            results = []
            progress_bar = st.progress(0.0, text="Testing setups...")
            symbols = list(price_histories)
            for completed, symbol in enumerate(symbols, start=1):
                data = _play_data(symbol, price_histories[symbol], opt_start, opt_end)
                results.extend(dp.evaluate_stock_setups(symbol, data, setups, split_date, opt_costs))
                progress_bar.progress(
                    completed / len(symbols),
                    text=f"Testing setups... {completed * len(setups):,}/{len(symbols) * len(setups):,}",
                )
            progress_bar.empty()

            if results:
                results_df = pd.DataFrame(results)
                st.session_state['dp_optimizer_results'] = results_df
                st.session_state['dp_optimizer_results_cache_key'] = results_cache_key
                st.success(
                    f"Setup search complete: {len(setups):,} setups tested across "
                    f"{results_df['Symbol'].nunique()} stocks."
                )
            else:
                st.warning("No usable backtests were produced. Try an earlier start date or a looser shortlist.")


    # ── Display Results ────────────────────────────────────────────────────────────
    recommended_setup = None
    stock_setup_map = {}
    setup_was_applied = False
    results_are_current = (
        'dp_optimizer_results' in st.session_state
        and st.session_state.get('dp_optimizer_results_cache_key') == results_cache_key
    )
    if 'dp_optimizer_results' in st.session_state and not results_are_current:
        st.session_state.pop('dp_active_setup', None)
        st.session_state.pop('dp_stock_setups', None)
        st.session_state.pop('dp_applied_setup_signature', None)
        st.info("Search settings changed. Run **Find Best Setup** to refresh the results.")

    col_cfg = {
        "Symbol": st.column_config.TextColumn("Symbol"),
        "Setup": st.column_config.TextColumn("Setup", width="large"),
        "Passes Gates": st.column_config.CheckboxColumn("Passes Gates"),
        "Setup Score": st.column_config.ProgressColumn("Setup Score", format="%.1f", min_value=0, max_value=100),
        "Stock Setup Score": st.column_config.ProgressColumn("Stock Setup Score", format="%.1f", min_value=0, max_value=100),
        "Validation Avg Trade": st.column_config.NumberColumn("Holdout Avg Trade", format="%.2f%%"),
        "Validation Win Rate": st.column_config.NumberColumn("Holdout Win Rate", format="%.1f%%"),
        "Validation Trades": st.column_config.NumberColumn("Holdout Trades", format="%d"),
        "Validation Profit Factor": st.column_config.NumberColumn("Profit Factor", help="Gross holdout gains / gross holdout losses", format="%.2f"),
        "Validation t-stat": st.column_config.NumberColumn("t-stat", help="Mean holdout trade divided by its standard error", format="%.2f"),
        "Return / Month": st.column_config.NumberColumn("Return / Month", help="Holdout average trade scaled to a 30-day holding", format="%.2f%%"),
        "Avg Days": st.column_config.NumberColumn("Avg Days Held", format="%.0f"),
        "Positive Stocks %": st.column_config.ProgressColumn("Positive Stocks", format="%.1f%%", min_value=0, max_value=100),
        "Stocks Validated": st.column_config.NumberColumn("Stocks Validated", format="%d"),
        "Train Avg Trade": st.column_config.NumberColumn("Train Avg Trade", format="%.2f%%"),
        "Train Trades": st.column_config.NumberColumn("Train Trades", format="%d"),
        "Generalization Gap": st.column_config.NumberColumn("Holdout - Train", format="%.2f%%"),
        "Avg Trade": st.column_config.NumberColumn("Avg Trade (all)", format="%.2f%%"),
        "Win Rate": st.column_config.NumberColumn("Win Rate (all)", format="%.1f%%"),
        "Trades": st.column_config.NumberColumn("Trades (all)", format="%d"),
        "Avg Drawdown": st.column_config.NumberColumn("Avg Drawdown", format="%.2f%%"),
    }

    if results_are_current:
        results_df = st.session_state['dp_optimizer_results'].copy()
        setup_df = dp.summarize_setups(results_df, setups, MIN_PROFIT_FACTOR, MIN_T_STAT)
        qualified_df, min_validated_stocks, min_validation_trades = dp.qualify_setups(setup_df, results_df)
        gate_description = (
            f"a positive after-cost average trade in both train and holdout, holdout profit factor "
            f"≥ {MIN_PROFIT_FACTOR:.1f}, t-stat ≥ {MIN_T_STAT:.1f}, and at least {min_validation_trades} "
            f"holdout trades across {min_validated_stocks} stocks"
        )

        if qualified_df.empty:
            st.session_state.pop('dp_active_setup', None)
            st.session_state.pop('dp_applied_setup_signature', None)
            st.warning(
                f"No setup passed the profitability gates ({gate_description}). Nothing was applied to the "
                "backtest or scanner. Review **All Setups**, try a longer date range, or widen the shortlist."
            )

        best_stock_df = dp.best_stock_setups(results_df, setups, MIN_STOCK_VALIDATION_TRADES, MIN_STOCK_VALIDATION_WIN_RATE)
        stock_setup_map = {
            row['Symbol']: {
                **dp.setup_from_row(row),
                'validation_avg_trade': float(row['Validation Avg Trade']),
                'validation_trades': int(row['Validation Trades']),
            }
            for _, row in best_stock_df.iterrows()
        }
        st.session_state['dp_stock_setups'] = stock_setup_map

        tab_setup, tab_stocks, tab_all = st.tabs(["Recommended Setup", "Stock Breakdown", "All Setups"])

        if not qualified_df.empty:
            recommended = qualified_df.iloc[0]
            recommended_setup = {
                **dp.setup_from_row(recommended),
                'score': float(recommended['Setup Score']),
                'validation_avg_trade': float(recommended['Validation Avg Trade']),
            }
            setup_signature = (results_cache_key, int(recommended['Setup ID']))
            if st.session_state.get('dp_applied_setup_signature') != setup_signature:
                st.session_state['dp_active_setup'] = recommended_setup
                st.session_state['dp_applied_setup_signature'] = setup_signature
                for key, value in {
                    'dpb_buy_fee': float(opt_buy_fee),
                    'dpb_sell_fee': float(opt_sell_fee),
                    'dpb_sell_tax': float(opt_sell_tax),
                    'dpb_div_tax': float(opt_dividend_tax),
                }.items():
                    st.session_state[key] = value
                st.session_state.pop('dp_single_setup_signature', None)
                setup_was_applied = True

            with tab_setup:
                st.caption(
                    "Only setups that pass the absolute profitability gates are eligible; among those, the "
                    "recommendation ranks holdout return, return per month held, win rate, breadth across "
                    "stocks, and statistical confidence."
                )
                if setup_was_applied:
                    st.success("Applied this setup to the single-stock backtest and upcoming scanner.")
                st.markdown(f"**{recommended['Setup']}**")
                metric_cols = st.columns(5)
                metric_cols[0].metric("Setup Score", f"{recommended['Setup Score']:.1f}/100")
                metric_cols[1].metric("Holdout Avg Trade", f"{recommended['Validation Avg Trade']:.2f}%")
                metric_cols[2].metric("Holdout Win Rate", f"{recommended['Validation Win Rate']:.1f}%")
                metric_cols[3].metric("Positive Stocks", f"{recommended['Positive Stocks %']:.1f}%")
                metric_cols[4].metric("Avg Days Held", f"{recommended['Avg Days']:.0f}")
                st.caption(f"Stock-specific setups available for {len(stock_setup_map)} tickers.")
                st.dataframe(
                    qualified_df.head(10), column_config=col_cfg,
                    column_order=[
                        'Setup Score', 'Setup', 'Validation Avg Trade', 'Return / Month',
                        'Validation Win Rate', 'Validation Profit Factor', 'Validation t-stat',
                        'Positive Stocks %', 'Validation Trades', 'Stocks Validated', 'Avg Days',
                        'Generalization Gap',
                    ],
                    hide_index=True, width='stretch',
                )
        else:
            with tab_setup:
                st.info("No setup passed the profitability gates, so none is recommended.")

        with tab_stocks:
            use_best_stock_setup = st.checkbox(
                "Use best setup for each stock", value=False, key="dp_use_best_stock_setup",
                help="Choose each stock's highest-scoring setup on held-out dividends.",
                disabled=recommended_setup is None,
            ) or recommended_setup is None
            if use_best_stock_setup:
                stock_breakdown_df = best_stock_df
                st.caption(
                    "Each row uses that stock's highest-scoring setup on held-out dividends. Only setups with at "
                    f"least {MIN_STOCK_VALIDATION_TRADES} holdout trades, a {MIN_STOCK_VALIDATION_WIN_RATE:.0f}% "
                    "holdout win rate, and a positive average trade in both train and holdout are considered. "
                    "Picking the best of many setups per stock flatters these numbers."
                )
                stock_breakdown_order = [
                    'Symbol', 'Stock Setup Score', 'Setup', 'Validation Avg Trade', 'Validation Win Rate',
                    'Validation Trades', 'Return / Month', 'Avg Days', 'Avg Drawdown', 'Avg Trade', 'Trades',
                ]
            else:
                stock_breakdown_df = results_df[results_df['Setup ID'] == int(recommended['Setup ID'])].sort_values(
                    ['Validation Avg Trade', 'Validation Trades'], ascending=False,
                )
                st.caption("Per-stock results for the single recommended setup.")
                stock_breakdown_order = [
                    'Symbol', 'Validation Avg Trade', 'Validation Win Rate', 'Validation Trades',
                    'Train Avg Trade', 'Train Trades', 'Avg Trade', 'Win Rate', 'Trades', 'Avg Days', 'Avg Drawdown',
                ]
            st.dataframe(
                stock_breakdown_df, column_config=col_cfg, column_order=stock_breakdown_order,
                hide_index=True, width='stretch',
            )

        with tab_all:
            st.caption(f"To be recommended, a setup needs {gate_description}.")
            st.dataframe(
                setup_df, column_config=col_cfg,
                column_order=[
                    'Passes Gates', 'Setup Score', 'Setup', 'Validation Avg Trade', 'Return / Month',
                    'Validation Win Rate', 'Validation Profit Factor', 'Validation t-stat',
                    'Positive Stocks %', 'Validation Trades', 'Stocks Validated', 'Avg Days',
                    'Train Avg Trade', 'Train Trades', 'Generalization Gap',
                ],
                hide_index=True, width='stretch',
            )

        st.download_button(
            "Download Setup Rankings CSV",
            data=df_to_csv(setup_df),
            file_name="dividend_play_setup_rankings.csv",
            mime="text/csv",
        )


    # ═══════════════════════════════════════════════════════════════════════════════
    # SINGLE STOCK BACKTEST
    # ═══════════════════════════════════════════════════════════════════════════════

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Run a Dividend Play Backtest on a Specific Ticker", divider='red')

    active_setup = st.session_state.get('dp_active_setup')
    stock_setup_map = st.session_state.get('dp_stock_setups', {})

    main_col1, main_col2 = st.columns([1, 2])
    with main_col1:
        stock_list = shortlist_tickers or sorted(dividend_events) or ['BBRI.JK']
        default_idx = stock_list.index('BBRI.JK') if 'BBRI.JK' in stock_list else 0
        stock = st.selectbox('Stock Ticker', options=stock_list, index=default_idx, key="dp_single_stock")
    with main_col2:
        date_cols = st.columns(2)
        start_date = date_cols[0].date_input('Start Date', value=datetime.date(2015, 1, 1), key="dp_single_start")
        end_date = date_cols[1].date_input('End Date', value=datetime.date.today(), key="dp_single_end")

    stock_specific_setup = stock_setup_map.get(stock)
    single_setup = stock_specific_setup or active_setup or BASELINE_SETUP
    single_setup_signature = (
        stock, st.session_state.get('dp_applied_setup_signature'), tuple(sorted(dp.setup_from_row(single_setup).items())),
    )
    if st.session_state.get('dp_single_setup_signature') != single_setup_signature:
        for key, value in {
            'dpb_window': (single_setup['entry_end_days'], single_setup['entry_start_days']),
            'dpb_entry_rule': single_setup['entry_rule'],
            'dpb_ma': single_setup['dip_ma_days'],
            'dpb_dip': single_setup['dip_pct'],
            'dpb_exit_rule': single_setup['exit_rule'],
            'dpb_hold': single_setup['hold_after_days'],
            'dpb_tp': single_setup['take_profit_pct'],
            'dpb_sl': single_setup['stop_loss_pct'],
        }.items():
            st.session_state[key] = value
        st.session_state['dp_single_setup_signature'] = single_setup_signature
    st.session_state.setdefault('dpb_buy_fee', DEFAULT_BUY_FEE)
    st.session_state.setdefault('dpb_sell_fee', DEFAULT_SELL_FEE)
    st.session_state.setdefault('dpb_sell_tax', DEFAULT_SELL_TAX)
    st.session_state.setdefault('dpb_div_tax', DEFAULT_DIVIDEND_TAX)

    with st.expander("Setup Parameters & Costs", expanded=False):
        if stock_specific_setup:
            st.info(f"Using the stock-specific holdout result for **{stock}**: {dp.format_setup(stock_specific_setup)}.")
        elif active_setup:
            st.info(f"Using the global recommendation: {dp.format_setup(active_setup)}.")
        else:
            st.info("Using the baseline setup. Run **Find Best Setup** to apply validated parameters.")

        entry_cols = st.columns(4)
        bt_window = entry_cols[0].slider(
            "Entry Window (days before ex)", min_value=5, max_value=270, step=5, key="dpb_window",
        )
        bt_entry_rule = entry_cols[1].selectbox(
            "Entry Rule", list(dp.ENTRY_LABELS), format_func=dp.ENTRY_LABELS.get, key="dpb_entry_rule",
            help="The dip rule buys on the first close below the moving average inside the window, "
                 "or on the last day of the window if no dip happens.",
        )
        bt_ma = entry_cols[2].number_input("Dip Moving Average", min_value=5, max_value=250, step=5, key="dpb_ma")
        bt_dip = entry_cols[3].number_input("Dip Below Average %", min_value=0.0, max_value=30.0, step=0.5, key="dpb_dip")

        st.markdown("#### Exit Rules & Costs")
        exit_cols = st.columns(4)
        bt_exit_rule = exit_cols[0].selectbox(
            "Exit Rule", list(dp.EXIT_LABELS), format_func=dp.EXIT_LABELS.get, key="dpb_exit_rule",
        )
        bt_hold = exit_cols[1].number_input(
            "Days Held After Ex", min_value=1, max_value=120, step=1, key="dpb_hold",
            help="Only used when holding through the ex-date.",
        )
        bt_tp = exit_cols[2].number_input("Take Profit %", min_value=0.0, max_value=100.0, step=1.0, key="dpb_tp",
                                          help="0 disables it.")
        bt_sl = exit_cols[3].number_input("Stop Loss %", min_value=0.0, max_value=100.0, step=1.0, key="dpb_sl",
                                          help="0 disables it.")
        cost_cols = st.columns(4)
        bt_buy_fee = cost_cols[0].number_input("Buy Fee %", min_value=0.0, max_value=5.0, step=0.01, key="dpb_buy_fee")
        bt_sell_fee = cost_cols[1].number_input("Sell Fee %", min_value=0.0, max_value=5.0, step=0.01, key="dpb_sell_fee")
        bt_sell_tax = cost_cols[2].number_input("Sell Tax %", min_value=0.0, max_value=5.0, step=0.01, key="dpb_sell_tax")
        bt_div_tax = cost_cols[3].number_input("Dividend Tax %", min_value=0.0, max_value=30.0, step=1.0, key="dpb_div_tax")

    single_params = {
        'entry_start_days': int(bt_window[1]), 'entry_end_days': int(bt_window[0]),
        'entry_rule': bt_entry_rule, 'dip_ma_days': int(bt_ma), 'dip_pct': float(bt_dip),
        'exit_rule': bt_exit_rule, 'hold_after_days': int(bt_hold),
        'take_profit_pct': float(bt_tp), 'stop_loss_pct': float(bt_sl),
        'buy_fee_pct': float(bt_buy_fee), 'sell_fee_pct': float(bt_sell_fee) + float(bt_sell_tax),
        'dividend_tax_pct': float(bt_div_tax),
    }

    st.divider()

    single_start = pd.Timestamp(start_date) - pd.Timedelta(days=PRICE_WARMUP_DAYS)


    def _cached_prices(symbol: str, start: pd.Timestamp) -> pd.DataFrame | None:
        """Prices already loaded by the optimizer, scanner, or an earlier run."""
        for key in ('dp_prices', 'dp_scan_prices', 'dp_single_prices'):
            frame = st.session_state.get(key, {}).get(symbol)
            if frame is not None and not frame.empty and frame['date'].min() <= start + pd.Timedelta(days=10):
                return frame
        return None


    single_prices = _cached_prices(stock, single_start)
    if single_prices is None:
        st.caption("Press **Run Dividend Play Backtest** to load this stock. Stocks loaded by the optimizer or "
                   "scanner update instantly when selected.")
    if st.button('Run Dividend Play Backtest', key="dp_run_single") and stock and single_prices is None:
        with st.spinner("Fetching price history..."):
            single_prices = fetch_price_history(stock, single_start.strftime('%Y-%m-%d'))
        st.session_state.setdefault('dp_single_prices', {})[stock] = single_prices

    if single_prices is not None:
        render_backtest(
            stock, single_prices, dividend_events.get(stock, pd.DataFrame(columns=['date', 'dividend'])),
            single_params, pd.Timestamp(start_date), pd.Timestamp(end_date),
        )


    # ═══════════════════════════════════════════════════════════════════════════════
    # UPCOMING DIVIDEND PLAY SCANNER
    # ═══════════════════════════════════════════════════════════════════════════════

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Upcoming Dividend Play Scanner", divider='orange')
    st.caption(
        "Estimate each shortlist stock's next ex-date from last year's schedule, then check whether today is "
        "inside its entry window and whether the dip rule is active. Confirm the announced cum and ex-dates "
        "before trading."
    )

    opening_within = st.number_input(
        "Show Windows Opening Within (days)", min_value=7, max_value=180, value=45, step=7, key="dp_opening_within",
    )
    if stock_setup_map:
        st.caption(
            f"Stock-specific validated setups apply to {len(stock_setup_map)} tickers; others use the "
            f"{'global recommendation' if active_setup else 'baseline setup'}."
        )
    elif active_setup:
        st.caption("All stocks use the global recommendation.")
    else:
        st.caption("All stocks use the baseline setup. Run the optimizer to apply validated parameters.")

    scan_button = st.button("Scan Upcoming Dividend Plays", type="primary", disabled=len(shortlist_tickers) == 0)

    if scan_button:
        today = pd.Timestamp(datetime.date.today())
        cached_prices = st.session_state.get('dp_prices', {})

        def _is_usable(frame):
            return (
                frame is not None and not frame.empty
                and frame['date'].max() >= today - pd.Timedelta(days=7)
                and frame['date'].min() <= pd.Timestamp(price_start) + pd.Timedelta(days=10)
            )

        stale = [ticker for ticker in shortlist_tickers if not _is_usable(cached_prices.get(ticker))]
        # Full history, so every stock also gets a backtest of the setup it will be traded with.
        fresh = fetch_price_histories(stale, price_start, "Loading price history...") if stale else {}
        st.session_state['dp_scan_prices'] = fresh

        scan_rows = []
        for ticker in shortlist_tickers:
            prices = fresh.get(ticker, cached_prices.get(ticker))
            divs = dividend_events.get(ticker)
            if prices is None or prices.empty or divs is None:
                continue
            stock_setup = stock_setup_map.get(ticker)
            setup = stock_setup or active_setup or BASELINE_SETUP
            nxt = dp.next_play_event(divs, setup, today)
            if nxt is None:
                continue
            stats = dp.summarize_trades(dp.backtest_dividend_play(
                _play_data(ticker, prices, opt_start, opt_end), **dp.setup_from_row(setup), **opt_costs,
            ))
            status = dp.entry_status(
                prices, nxt['expected_date'], setup['entry_start_days'], setup['entry_end_days'],
                setup['dip_ma_days'], setup['dip_pct'], as_of=today,
            )
            signal = dp.scan_signal(status, setup['entry_rule'], opening_within)
            scan_rows.append({
                'Symbol': ticker,
                'Signal': signal,
                'Validated': stock_setup is not None or active_setup is not None,
                'Parameter Source': 'Stock-specific' if stock_setup else ('Global' if active_setup else 'Baseline'),
                'Expected Ex-date': nxt['expected_date'].date(),
                'Window Opens': status['window_start'].date(),
                'Window Closes': status['window_end'].date(),
                'Days to Window': status['days_to_window'],
                'Current Price': status['close'],
                'Vs MA': status['vs_sma'],
                'Next Dividend Yield': nxt['dividend'] / status['close'] * 100 if status['close'] > 0 else np.nan,
                'Expected Trade': setup.get('validation_avg_trade', np.nan),
                'Trades': stats['trades'],
                'Win Rate': stats.get('win_rate', np.nan),
                'Avg Return': stats.get('avg_return', np.nan),
                'Median Return': stats.get('median_return', np.nan),
                'Avg Days': stats.get('avg_days', np.nan),
                'Return / Month': stats.get('return_per_month', np.nan),
                'CAGR': stats.get('cagr', np.nan),
                'Setup': dp.format_setup(setup),
                'History Through': prices['date'].max().date(),
            })

        if not scan_rows:
            st.warning("Could not evaluate any stock. Verify the price and dividend sources.")
        else:
            scan_df = pd.DataFrame(scan_rows)
            scan_df = scan_df.assign(order=scan_df['Signal'].map(dp.SIGNAL_ORDER)).sort_values(
                ['order', 'Validated', 'Expected Trade', 'Days to Window'],
                ascending=[True, False, False, True], na_position='last',
            ).drop(columns='order')
            st.session_state['dp_scan_results'] = scan_df

    scan_df = st.session_state.get('dp_scan_results')
    if scan_df is not None and 'Trades' not in scan_df.columns:
        # Results from an older version of this page; rescan to add backtest columns.
        scan_df = None
    if scan_df is not None and not scan_df.empty:
        buy_df = scan_df[scan_df['Signal'] == 'Buy Zone']
        window_df = scan_df[scan_df['Signal'] == 'In Window - Wait for Dip']
        soon_df = scan_df[scan_df['Signal'] == 'Opening Soon']
        traded = scan_df[scan_df['Trades'] > 0]
        total_trades = int(traded['Trades'].sum())
        result_cols = st.columns(4)
        result_cols[0].metric("Trades Simulated", f"{total_trades:,}",
                              help="Every shortlist stock backtested with the setup it will be traded with")
        if total_trades:
            weights = traded['Trades'] / total_trades
            result_cols[1].metric("Win Rate", _pct((traded['Win Rate'] * weights).sum(), 0, signed=False))
            result_cols[2].metric("Average Return per Trade", _pct((traded['Avg Return'] * weights).sum()))
            result_cols[3].metric("Average Holding", f"{(traded['Avg Days'] * weights).sum():.0f} days")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Buy Zone", len(buy_df), help="Inside the entry window and the entry rule is met")
        summary_cols[1].metric("In Window", len(window_df), help="Inside the entry window, waiting for a dip")
        summary_cols[2].metric("Opening Soon", len(soon_df))
        summary_cols[3].metric("Stocks Tracked", len(scan_df))

        scanner_cfg = {
            'Signal': st.column_config.TextColumn('Signal'),
            'Validated': st.column_config.CheckboxColumn('Validated', help='Setup came from the holdout optimizer'),
            'Expected Ex-date': st.column_config.DateColumn('Expected Ex-date', help="Last year's ex-date rolled forward"),
            'Window Opens': st.column_config.DateColumn('Window Opens'),
            'Window Closes': st.column_config.DateColumn('Window Closes'),
            'Days to Window': st.column_config.NumberColumn('Days to Window', format='%d'),
            'Current Price': st.column_config.NumberColumn('Price', format='%.0f'),
            'Vs MA': st.column_config.NumberColumn('Price vs MA', format='%+.1f%%'),
            'Next Dividend Yield': st.column_config.NumberColumn('Next Dividend Yield', format='%.2f%%'),
            'Expected Trade': st.column_config.NumberColumn('Holdout Avg Trade', format='%.2f%%'),
            'Trades': st.column_config.NumberColumn('Trades', format='%d'),
            'Win Rate': st.column_config.ProgressColumn('Win Rate', min_value=0, max_value=100, format='%.0f%%'),
            'Avg Return': st.column_config.NumberColumn('Avg Return', format='%+.1f%%'),
            'Median Return': st.column_config.NumberColumn('Median Return', format='%+.1f%%'),
            'Avg Days': st.column_config.NumberColumn('Avg Days', format='%.0f'),
            'Return / Month': st.column_config.NumberColumn('Return / Month', format='%+.2f%%',
                                                            help='Average return scaled to a 30-day holding'),
            'CAGR': st.column_config.NumberColumn('Compounded CAGR', format='%+.1f%%',
                                                  help='Annualized growth if every trade were taken back to back'),
            'Setup': st.column_config.TextColumn('Setup', width='large'),
            'History Through': st.column_config.DateColumn('History Through'),
        }
        actionable_df = pd.concat([buy_df, window_df])
        if not buy_df.empty:
            st.success(f"{len(buy_df)} stock(s) are in their entry window with the entry rule met.")
        scan_tabs = st.tabs(["Actionable Now", "Opening Soon", "All Stocks"])
        for tab, frame, empty_text in zip(
            scan_tabs,
            [actionable_df, soon_df, scan_df],
            ["No stock is inside its entry window today.", "No entry window opens soon.", "No stocks tracked."],
        ):
            with tab:
                if frame.empty:
                    st.info(empty_text)
                else:
                    st.dataframe(
                        frame, column_config=scanner_cfg, hide_index=True, width='stretch',
                        column_order=[
                            'Symbol', 'Signal', 'Expected Ex-date', 'Window Opens', 'Window Closes', 'Vs MA',
                            'Next Dividend Yield', 'Trades', 'Win Rate', 'Avg Return', 'Median Return',
                            'Avg Days', 'Return / Month', 'CAGR', 'Expected Trade', 'Validated',
                            'Parameter Source', 'Current Price', 'Days to Window', 'Setup', 'History Through',
                        ],
                    )
        st.caption(
            '**Buy Zone** - inside the entry window with the entry rule met. **In Window** - buying is allowed, '
            'waiting for a dip. **Opening Soon** - the window opens within the chosen number of days. Backtest '
            'columns use the setup each stock is traded with, over the optimizer date range.'
        )
        st.download_button(
            "Download Scanner CSV",
            data=df_to_csv(scan_df),
            file_name=f"dividend_play_scanner_{datetime.date.today().isoformat()}.csv",
            mime="text/csv",
        )

    st.warning(
        "Backtests model fees and dividend tax but not slippage or partial fills, and future ex-dates are "
        "estimates. Confirm announced dates before trading."
    )
