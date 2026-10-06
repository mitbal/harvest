import datetime
import logging
import os
import secrets

import pandas as pd
import streamlit as st

import harvest.trading_store as store
import harvest.trading_ui as ui
from harvest.premium import current_email, get_connection, require_premium


TOOL_LABELS = {'dividend_play': 'Dividend Play', 'swing': 'Swing Trading', 'long_term': 'Long Term'}
TOOL_ICONS = {'dividend_play': 'payments', 'swing': 'bolt', 'long_term': 'trending_up'}
logger = logging.getLogger(__name__)

st.set_page_config(page_title='My Trades - Panen Dividen')
ui.inject_styles()
ui.hero('Trading · Journal', 'My trades',
        'Track the trades you take from the trading tools, count down to each sell-by date and get '
        'Telegram reminders before you need to sell.', 'menu_book')

email = current_email()
if email is None:
    st.button('Log in with Google', icon=':material/login:', type='primary', on_click=st.login, args=('google',))
    st.stop()
require_premium()

client = get_connection()
if 'trades_toast' in st.session_state:
    st.toast(st.session_state.pop('trades_toast'), icon=':material/check:')


def _short(symbol: str) -> str:
    return symbol.removesuffix('.JK')


def _day(value) -> str:
    date = pd.Timestamp(value)
    return f'{date:%a} {date.day} {date:%b} {date:%Y}'


def _idr(value) -> str:
    return f'Rp {value:,.0f}'


def _countdown(exit_by) -> str:
    if exit_by is None or pd.isna(exit_by):
        return 'No sell-by date'
    sessions = store.sessions_until(exit_by)
    if sessions == 0:
        return 'Sell-by date has passed'
    if sessions == 1:
        return 'Sell today'
    return f'{sessions} sessions left, sell by {_day(exit_by)}'


@st.dialog('Close position')
def close_dialog(position: pd.Series) -> None:
    st.markdown(f"**{_short(position['symbol'])}** · {int(position['lots'])} lots bought at "
                f"{_idr(position['entry_price'])}")
    exit_price = st.number_input('Sell price (Rp)', min_value=1.0, value=float(position['entry_price']),
                                 step=5.0, format='%.0f')
    closed_at = st.date_input('Sold on', value=datetime.date.today(), min_value=position['entry_date'],
                              max_value=datetime.date.today())
    if st.button('Close position', type='primary'):
        store.close_position(client, email, int(position['id']), closed_at, exit_price)
        st.rerun()


@st.dialog('Remove position')
def remove_dialog(position: pd.Series) -> None:
    st.markdown(f"Remove **{_short(position['symbol'])}** from your trades? This can't be undone.")
    if st.button('Remove', type='primary'):
        store.delete_position(client, email, int(position['id']))
        st.rerun()


try:
    settings = store.get_settings(client, email)
    positions = store.list_positions(client, email)
except Exception:
    logger.exception('Could not load trades')
    st.warning('Your trades are unavailable right now. Please try again in a few minutes.')
    st.stop()

open_positions = positions[positions['closed_at'].isna()]
closed_positions = positions[positions['closed_at'].notna()]

def _holding(position: pd.Series) -> str:
    """Progress through the holding period, in IDX sessions, with the countdown as its legend."""
    countdown = _countdown(position['exit_by'])
    total = len(store.idx_trading_days(position['entry_date'], position['exit_by']))
    left = store.sessions_until(position['exit_by'])
    tone = 'down' if left == 0 else ('warn' if left <= 2 else '')
    return ui.progress(1 - left / total if total else 1.0, ui.esc(countdown),
                       f'{total - left} of {total} sessions held' if total else '', tone)


def render_position(position: pd.Series) -> None:
    rule = position['exit_rule'] or {}
    tool = position['tool']
    countdown = _countdown(position['exit_by'])
    urgent = countdown in ('Sell today', 'Sell-by date has passed')
    lots, entry = int(position['lots']), float(position['entry_price'])
    left = store.sessions_until(position['exit_by'])
    if urgent:
        status = ui.chip(countdown, 'warn lg' if countdown == 'Sell today' else 'down lg', 'alarm')
    elif rule.get('kind') == 'valuation':
        status = ui.chip('Watching valuation', 'brand lg', 'visibility')
    elif position['exit_by'] is not None and not pd.isna(position['exit_by']):
        status = ui.chip(f'{left} sessions left', 'warn lg' if left <= 2 else 'brand lg', 'timer')
    else:
        status = ''
    tiles = [('Bought at', ui.esc(_idr(entry)), 'per share'),
             ('Size', f'{lots} lots', ui.esc(f'about {_idr(entry * lots * 100)}'))]
    if rule.get('stop_price'):
        tiles.append(('Stop', ui.esc(_idr(rule['stop_price'])), f"{(rule['stop_price'] / entry - 1) * 100:+.1f}%", 'down'))
    if rule.get('target_price'):
        tiles.append(('Target', ui.esc(_idr(rule['target_price'])), f"{(rule['target_price'] / entry - 1) * 100:+.1f}%", 'up'))
    parts = [
        ui.card_head(position['symbol'], f"{TOOL_LABELS.get(tool, tool)} · bought {_day(position['entry_date'])}", status,
                     icon_name=TOOL_ICONS.get(tool, 'candlestick_chart')),
        ui.cells(tiles),
    ]
    if rule.get('kind') == 'valuation':
        # Long Term positions have no sell-by date; the nightly scan watches the exit conditions.
        parts.append(ui.strip(f"Exit when P/E reaches its mean of {rule['pe_mean']:.1f} (it was "
                              f"{rule['pe_at_entry']:.1f}) or a new quarter turns growth negative", icon_name='visibility'))
    elif position['exit_by'] is not None and not pd.isna(position['exit_by']):
        parts.append(_holding(position))
    if rule.get('setup_text'):
        cum = rule.get('key_date')
        parts.append(ui.foot(ui.esc(rule['setup_text'] + (
            f"; cum date {'~' if rule.get('key_date_estimated') else ''}{_day(cum)}" if cum else ''))))
    with ui.card(f"pos-{position['id']}", urgent=urgent):
        ui.render(''.join(parts))
        with st.container(horizontal=True):
            if st.button('Close', key=f"close_{position['id']}", icon=':material/sell:'):
                close_dialog(position)
            if st.button('Remove', key=f"remove_{position['id']}", type='tertiary', icon=':material/delete:'):
                remove_dialog(position)


# ── Open positions ─────────────────────────────────────────────────────────── #
ui.section('Open positions', len(open_positions), 'Soonest sell-by first')
if open_positions.empty:
    ui.empty_state('No open positions', 'Use "Track this trade" on a trading card to add one.', 'bookmark_add')
ordered = open_positions.assign(_by=pd.to_datetime(open_positions['exit_by'])).sort_values('_by', na_position='last')
for start in range(0, len(ordered), 2):
    for column, (_, position) in zip(st.columns(2), ordered.iloc[start:start + 2].iterrows()):
        with column:
            render_position(position)

# ── Closed positions ───────────────────────────────────────────────────────── #
ui.section('Closed positions', len(closed_positions))
if closed_positions.empty:
    ui.empty_state('No closed trades yet', 'Closed trades and their results appear here.', 'history')
else:
    results = pd.DataFrame([
        store.position_result(position, settings['buy_fee_pct'], settings['sell_fee_pct'])
        for _, position in closed_positions.iterrows()
    ], index=closed_positions.index)
    closed = closed_positions.join(results)
    wins = int((closed['return_pct'] > 0).sum())
    summary = st.container(horizontal=True)
    summary.metric('Closed trades', len(closed))
    summary.metric('Won', f'{wins} of {len(closed)}')
    total, cost = closed['profit'].sum(), (closed['entry_price'] * closed['lots'] * 100).sum()
    summary.metric('Total result', _idr(total), help='After your fees, including dividends received',
                   delta=f'{total / cost * 100:+.1f}% on cost' if cost else None)
    table = pd.DataFrame({
        'Stock': closed['symbol'].map(_short),
        'Bought': pd.to_datetime(closed['entry_date']),
        'Sold': pd.to_datetime(closed['closed_at']),
        'Lots': closed['lots'].astype(int),
        'Buy price': closed['entry_price'].astype(float),
        'Sell price': closed['exit_price'].astype(float),
        'Return': closed['return_pct'],
        'Result': closed['profit'],
    })
    st.dataframe(
        table, hide_index=True, width='stretch',
        column_config={
            'Bought': st.column_config.DateColumn(format='D MMM YYYY'),
            'Sold': st.column_config.DateColumn(format='D MMM YYYY'),
            'Buy price': st.column_config.NumberColumn(format='%,.0f'),
            'Sell price': st.column_config.NumberColumn(format='%,.0f'),
            'Return': st.column_config.NumberColumn(format='%+.2f%%', help='After your fees, including dividends'),
            'Result': st.column_config.NumberColumn('Result (Rp)', format='%,.0f'),
        },
    )

# ── Settings ───────────────────────────────────────────────────────────────── #
ui.section('Settings')
with st.form('trade_settings'):
    fee_cols = st.columns(2)
    buy_fee = fee_cols[0].number_input('Buy fee %', min_value=0.0, max_value=5.0, step=0.01,
                                       value=float(settings['buy_fee_pct']))
    sell_fee = fee_cols[1].number_input('Sell fee and tax %', min_value=0.0, max_value=5.0, step=0.01,
                                        value=float(settings['sell_fee_pct']))
    size_cols = st.columns(2)
    capital = size_cols[0].number_input(
        'Trading capital (Rp)', min_value=0.0, step=1_000_000.0, format='%.0f',
        value=float(settings.get('capital_idr') or 0.0),
        help='Used to size swing trades. Stays private to your account.')
    risk = size_cols[1].number_input(
        'Risk per trade %', min_value=0.1, max_value=10.0, step=0.1,
        value=float(settings.get('risk_pct') or 1.0),
        help='How much of your capital a stop-out may lose')
    prefs = settings['alert_prefs']
    new_signals = st.toggle('Alert me about every new Dividend Play entry window',
                            value=bool(prefs.get('new_signals')))
    watch = st.multiselect('Stocks to alert me about', options=sorted(set(prefs.get('watch', []))),
                           default=sorted(set(prefs.get('watch', []))), format_func=_short,
                           help='Add stocks with "Alert me" on the Dividend Play page')
    if st.form_submit_button('Save settings', type='primary'):
        store.save_settings(client, email, {
            'buy_fee_pct': buy_fee, 'sell_fee_pct': sell_fee,
            'capital_idr': capital or None, 'risk_pct': risk,
            'alert_prefs': {**prefs, 'new_signals': new_signals, 'watch': watch},
        })
        st.session_state['trades_toast'] = 'Settings saved.'
        st.rerun()

def _bot_username() -> str | None:
    name = os.environ.get('TELEGRAM_BOT_USERNAME')
    if not name:
        try:
            name = st.secrets.get('TELEGRAM_BOT_USERNAME')
        except Exception:
            name = None
    return name


bot = _bot_username()
with st.container(border=True):
    ui.render(f'<div class="hv-section" style="margin:0">{ui.icon("send")}<h3 style="font-size:1.05rem">'
              'Telegram alerts</h3></div>')
    if settings.get('telegram_chat_id'):
        st.markdown('Linked. Alerts and sell-by reminders arrive in your Telegram chat.')
        if st.button('Unlink Telegram', icon=':material/link_off:'):
            store.save_settings(client, email, {'telegram_chat_id': None})
            st.rerun()
    elif not bot:
        st.caption('Telegram alerts are not available yet.')
    elif settings.get('telegram_link_token'):
        st.markdown('Open the bot and press **Start** to finish linking. Then reload this page.')
        st.link_button('Open Telegram', f"https://t.me/{bot}?start={settings['telegram_link_token']}",
                       icon=':material/send:', type='primary')
    elif st.button('Link Telegram', icon=':material/send:', type='primary'):
        store.save_settings(client, email, {'telegram_link_token': secrets.token_urlsafe(16)})
        st.rerun()

st.caption('Results use your fees and are for your own records. Not financial advice.')
