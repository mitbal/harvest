"""My Trades: the trading hub.

Tonight's signals from every trading tool and the dates coming up, then the trades you are
tracking with their sell-by countdowns, your closed results and settings. The signal sections
only read the published store (``trading_signals``, ``trading_runs``) and are open to everyone
with the tools' free-tier rules; the journal needs a login and supporter access.
"""
import datetime
import logging
import os
import secrets

import pandas as pd
import streamlit as st

import harvest.position as pos
import harvest.trading_store as store
import harvest.trading_ui as ui
from harvest.premium import current_email, get_connection, has_access, require_premium


TOOLS = {
    'dividend_play': {'label': 'Dividend Play', 'icon': 'payments', 'page': 'apps/trading/dividend_play.py',
                      'note': 'Entry window open now'},
    'swing': {'label': 'Swing Trading', 'icon': 'bolt', 'page': 'apps/trading/swing_trading.py',
              'note': 'Buy at the next open'},
    'long_term': {'label': 'Long Term', 'icon': 'trending_up', 'page': 'apps/trading/backtester.py',
                  'note': 'Top candidates of the valuation screen'},
}
TOOL_LABELS = {tool: info['label'] for tool, info in TOOLS.items()}
TOOL_ICONS = {tool: info['icon'] for tool, info in TOOLS.items()}
# Full-access cards per tool; the tool page has the rest.
CARDS_PER_TOOL = {'dividend_play': 4, 'swing': 4, 'long_term': 3}
COMING_UP_DAYS = 45
SELL_SOON_SESSIONS = 5
logger = logging.getLogger(__name__)

st.set_page_config(page_title='My Trades - Panen Dividen')
ui.inject_styles()
ui.hero('Trading · My trades', 'My trades',
        'What to act on after the last close across Dividend Play, Swing Trading and Long Term, the dates '
        'coming up, and the trades you track with their sell-by countdowns and Telegram reminders.', 'menu_book')

email = current_email()
access = has_access()
if 'trades_toast' in st.session_state:
    st.toast(st.session_state.pop('trades_toast'), icon=':material/check:')


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=10 * 60, show_spinner=False)
def _load_tool(tool: str) -> dict:
    client = get_connection()
    run = store.latest_run(client, tool, 'scan')
    return {
        'run': run,
        # The latest run's own signals, so a night with no signals doesn't show the night before.
        'signals': store.read_signals(client, tool, run['as_of']) if run else None,
        'stale': store.is_stale(client, tool),
    }


def load_tool(tool: str) -> dict | None:
    """Latest scan run and its signals for one tool; None when the store is unreachable."""
    try:
        return _load_tool(tool)
    except Exception:
        logger.exception('Could not load published %s signals', tool)
        return None


def act_now(tool: str, signals: pd.DataFrame | None) -> pd.DataFrame:
    """Signals to act on, in the order the tool's own Today view shows them."""
    if signals is None or signals.empty:
        return pd.DataFrame()
    if tool == 'long_term':
        ranked = signals.assign(_rank=signals['detail'].map(lambda d: d.get('rank') or 0))
        return ranked.sort_values('_rank')
    actionable = signals[signals['status'] == 'actionable']
    expected = actionable['detail'].map(lambda d: (d.get('holdout') or {}).get('avg_return') or 0)
    return actionable.assign(_expected=expected).sort_values('_expected', ascending=False)


def coming_up(published: dict, positions: pd.DataFrame | None, today: datetime.date | None = None) -> pd.DataFrame:
    """Dated events in the next ``COMING_UP_DAYS``: windows opening, near-setups and your exits."""
    today = today or datetime.date.today()
    horizon = today + datetime.timedelta(days=COMING_UP_DAYS)
    events = []
    dividend = (published.get('dividend_play') or {}).get('signals')
    if dividend is not None:
        for _, signal in dividend[dividend['status'] == 'opening_soon'].iterrows():
            opens = ui.first_session(signal['detail']['window_start'])
            estimated = bool(signal['key_date_estimated'])
            events.append({'date': opens, 'symbol': signal['symbol'], 'tool': 'dividend_play',
                           'event': 'Entry window opens' + (' (ex-date estimated)' if estimated else '')})
    swing = published.get('swing') or {}
    if swing.get('signals') is not None:
        watch = swing['signals'][swing['signals']['status'] == 'opening_soon']
        if not watch.empty:
            session = ui.next_session(swing['run']['as_of'])
            events.extend({'date': session, 'symbol': symbol, 'tool': 'swing',
                           'event': 'Close to a setup, could trigger at this close'} for symbol in watch['symbol'])
    if positions is not None:
        for _, position in positions.iterrows():
            rule = position['exit_rule'] or {}
            if position['exit_by'] is not None and not pd.isna(position['exit_by']):
                events.append({'date': position['exit_by'], 'symbol': position['symbol'], 'tool': position['tool'],
                               'event': 'Sell by, your position'})
            if rule.get('key_date'):
                events.append({'date': pd.Timestamp(rule['key_date']).date(), 'symbol': position['symbol'],
                               'tool': position['tool'], 'event': 'Cum date, your position'})
    frame = pd.DataFrame(events, columns=['date', 'symbol', 'tool', 'event'])
    if frame.empty:
        return frame
    frame['date'] = pd.to_datetime(frame['date']).dt.date
    frame = frame[(frame['date'] >= today) & (frame['date'] <= horizon)]
    return frame.sort_values(['date', 'symbol']).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CARDS
# Short versions of each tool's Today cards. The full card, with Track this trade,
# stays on the tool page.
# ═══════════════════════════════════════════════════════════════════════════════

def _price(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _record(holdout: dict) -> str:
    trades = holdout.get('trades') or 0
    if not trades:
        return 'No holdout trades yet'
    avg = holdout.get('avg_return')
    return (f"Holdout {holdout.get('wins', 0)} of {trades} won"
            + (f', {avg:+.1f}% average after costs' if avg is not None else ''))


def _avg_badge(holdout: dict) -> str:
    avg = holdout.get('avg_return')
    return ui.chip(f'{avg:+.1f}% avg', 'up lg' if avg > 0 else 'down lg', 'trending_up') if avg is not None else ''


def _dividend_parts(signal: pd.Series) -> list[str]:
    detail = signal['detail']
    estimated = bool(signal['key_date_estimated'])
    holdout = detail.get('holdout') or {}
    parts = [
        ui.card_head(signal['symbol'], detail.get('company'), _avg_badge(holdout), icon_name='payments'),
        ui.cells([
            ('Entry at or below', ui.esc(ui.idr(ui.tick_floor(signal['entry_price']))), 'limit price'),
            ('Cum date', ui.esc(ui.day(signal['key_date'], estimated)), 'estimated' if estimated else 'announced'),
            ('Sell by', ui.esc(ui.day(signal['exit_by'], estimated)), 'at the latest'),
        ]),
    ]
    window = detail.get('window') or {}
    if window.get('of'):
        parts.append(ui.progress(
            window['day'] / window['of'], f"Day <b>{window['day']}</b> of {window['of']} in the entry window",
            f"{window['sessions_left']} sessions left",
        ))
    parts.append(ui.foot(ui.winbar(holdout.get('wins'), holdout.get('trades')) + ui.esc(_record(holdout))))
    return parts


def _swing_parts(signal: pd.Series) -> list[str]:
    detail = signal['detail']
    holdout = detail.get('holdout') or {}
    stop, target = _price(signal['stop_price']), _price(signal['target_price'])
    tiles = [('Buy at open', ui.esc(ui.day(signal['key_date'])), 'next session')]
    if stop is not None:
        tiles.append(('Stop', f"−{detail['stop_pct']:g}%", f'about {ui.esc(ui.idr(stop))}', 'down'))
    if target is not None:
        tiles.append(('Target', f"+{detail['target_pct']:g}%", f'about {ui.esc(ui.idr(target))}', 'up'))
    tiles.append(('Sell by', ui.esc(ui.day(signal['exit_by'])), f"{detail.get('max_hold')} sessions max"))
    return [
        ui.card_head(signal['symbol'], detail.get('company'), _avg_badge(holdout), icon_name='bolt'),
        ui.cells(tiles),
        ui.foot(ui.winbar(holdout.get('wins'), holdout.get('trades')) + ui.esc(_record(holdout))),
    ]


def _long_term_parts(signal: pd.Series) -> list[str]:
    d = signal['detail']
    score = d['scores']['score']
    ring = f'<div class="hv-ring" style="--p:{score:.0f}" title="Score out of 100"><span>{score:.0f}</span></div>'
    pe, upside = d['pe'], d['upside']
    below = pos.gap_to_median(pe['discount'])
    return [
        ui.card_head(signal['symbol'], d.get('company'), ring, rank=d.get('rank'), icon_name='trending_up'),
        ui.cells([
            ('Price', ui.esc(ui.idr(signal['entry_price'])), 'last close'),
            ('P/E', f"{pe['current']:.1f}", ui.esc(f"{abs(below):.0f}% {'below' if below >= 0 else 'above'} median")),
            ('Upside to mean', f"{upside['to_mean']:+.0f}%", f'about {ui.esc(ui.idr(signal["target_price"]))}',
             'up' if upside['to_mean'] > 0 else 'down'),
        ]),
        ui.foot(ui.esc(f"{d.get('sector') or 'Unknown sector'}. Exit when the discount closes or growth turns negative.")),
    ]


CARD_PARTS = {'dividend_play': _dividend_parts, 'swing': _swing_parts, 'long_term': _long_term_parts}


def render_card(tool: str, signal: pd.Series) -> None:
    symbol, page = signal['symbol'], TOOLS[tool]['page']
    with ui.card(f'{tool}-{symbol}'):
        ui.render(''.join(CARD_PARTS[tool](signal)))
        with st.container(horizontal=True):
            if st.button(f"Open in {TOOLS[tool]['label']}", key=f'open_{tool}_{symbol}', type='primary',
                         icon=':material/arrow_forward:'):
                st.switch_page(page)
            if st.button('See evidence', key=f'evidence_{tool}_{symbol}', icon=':material/query_stats:'):
                st.switch_page(page, query_params={'view': ui.VIEW_EVIDENCE, 'stock': symbol})


def tool_header(tool: str, data: dict | None, count: int) -> None:
    """Tool name, how many signals and when its scan last ran."""
    info = TOOLS[tool]
    run = (data or {}).get('run')
    if data is None:
        status = '<span class="hv-status stale"><span class="dot"></span>Unavailable right now</span>'
    elif run is None:
        status = '<span class="hv-status stale"><span class="dot"></span>Not published yet</span>'
    else:
        text = f"Updated {ui.day(run['as_of'])}" + (' · may be out of date' if data['stale'] else '')
        status = f'<span class="hv-status{" stale" if data["stale"] else ""}"><span class="dot"></span>{ui.esc(text)}</span>'
    ui.render(f'<div class="hv-tool"><span class="name">{ui.icon(info["icon"])}{ui.esc(info["label"])}</span>'
              f'<span class="count">{count}</span>'
              f'<span class="note">{ui.esc(info["note"])}</span>{status}</div>')


# ═══════════════════════════════════════════════════════════════════════════════
# JOURNAL
# The trades you track, with close and remove dialogs.
# ═══════════════════════════════════════════════════════════════════════════════

def _day(value) -> str:
    date = pd.Timestamp(value)
    return f'{date:%a} {date.day} {date:%b} {date:%Y}'


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
    st.markdown(f"**{ui.short(position['symbol'])}** · {int(position['lots'])} lots bought at "
                f"{ui.idr(position['entry_price'])}")
    exit_price = st.number_input('Sell price (Rp)', min_value=1.0, value=float(position['entry_price']),
                                 step=5.0, format='%.0f')
    closed_at = st.date_input('Sold on', value=datetime.date.today(), min_value=position['entry_date'],
                              max_value=datetime.date.today())
    if st.button('Close position', type='primary'):
        store.close_position(client, email, int(position['id']), closed_at, exit_price)
        st.rerun()


@st.dialog('Remove position')
def remove_dialog(position: pd.Series) -> None:
    st.markdown(f"Remove **{ui.short(position['symbol'])}** from your trades? This can't be undone.")
    if st.button('Remove', type='primary'):
        store.delete_position(client, email, int(position['id']))
        st.rerun()


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
    tiles = [('Bought at', ui.esc(ui.idr(entry)), 'per share'),
             ('Size', f'{lots} lots', ui.esc(f'about {ui.idr(entry * lots * 100)}'))]
    if rule.get('stop_price'):
        tiles.append(('Stop', ui.esc(ui.idr(rule['stop_price'])), f"{(rule['stop_price'] / entry - 1) * 100:+.1f}%", 'down'))
    if rule.get('target_price'):
        tiles.append(('Target', ui.esc(ui.idr(rule['target_price'])), f"{(rule['target_price'] / entry - 1) * 100:+.1f}%", 'up'))
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


def render_journal(settings: dict, positions: pd.DataFrame) -> None:
    open_positions = positions[positions['closed_at'].isna()]
    closed_positions = positions[positions['closed_at'].notna()]

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
        summary.metric('Total result', ui.idr(total), help='After your fees, including dividends received',
                       delta=f'{total / cost * 100:+.1f}% on cost' if cost else None)
        table = pd.DataFrame({
            'Stock': closed['symbol'].map(ui.short),
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
                               default=sorted(set(prefs.get('watch', []))), format_func=ui.short,
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


def render_telegram(settings: dict) -> None:
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


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE
# ═══════════════════════════════════════════════════════════════════════════════

published = {tool: load_tool(tool) for tool in TOOLS}
ranked = {tool: act_now(tool, (data or {}).get('signals')) for tool, data in published.items()}

client, settings, positions = None, None, None
if email and access:
    try:
        client = get_connection()
        settings, positions = store.get_settings(client, email), store.list_positions(client, email)
    except Exception:
        logger.exception('Could not load trades')
open_positions = None if positions is None else positions[positions['closed_at'].isna()]
events = coming_up(published, open_positions)

metrics = st.container(horizontal=True)
metrics.metric('Signals today', len(ranked['dividend_play']) + len(ranked['swing']),
               help='Dividend Play entry windows open now plus swing setups for the next open')
metrics.metric('Long term candidates', len(ranked['long_term']))
if open_positions is None:
    metrics.metric('Your positions', '–', help='Log in to track your trades' if email is None else None)
else:
    selling = int(sum(0 < store.sessions_until(by) <= SELL_SOON_SESSIONS
                      for by in open_positions['exit_by'].dropna()))
    metrics.metric('Your positions', len(open_positions),
                   delta=f'{selling} to sell within {SELL_SOON_SESSIONS} sessions' if selling else None,
                   delta_color='off', delta_arrow='off')
metrics.metric(f'Coming up in {COMING_UP_DAYS} days', len(events))

# ── Act now ─────────────────────────────────────────────────────────────── #
ui.section('Act now', sum(len(rows) for rows in ranked.values()), 'From the scans after the last close')
if all(data is None for data in published.values()):
    st.warning('Signals are unavailable right now. Please try again in a few minutes.')
hidden = []
for tool, data in published.items():
    rows = ranked[tool]
    tool_header(tool, data, len(rows))
    if rows.empty:
        if data is not None and data['run'] is not None:
            st.caption('Nothing to act on today.' if tool != 'long_term'
                       else 'No company passes every rule of the screen today.')
        continue
    shown = rows.head(CARDS_PER_TOOL[tool] if access else 1)
    for start in range(0, len(shown), 2):
        for column, (_, signal) in zip(st.columns(2), shown.iloc[start:start + 2].iterrows()):
            with column:
                render_card(tool, signal)
    if not access:
        hidden.extend(rows['symbol'].iloc[1:])
    elif len(rows) > len(shown) and st.button(
        f"See all {len(rows)} in {TOOLS[tool]['label']}", key=f'all_{tool}', type='tertiary',
        icon=':material/arrow_forward:',
    ):
        st.switch_page(TOOLS[tool]['page'])
if hidden:
    ui.unlock_prompt(list(dict.fromkeys(hidden)), noun='signal', page=TOOLS['dividend_play']['page'])

# ── Coming up ───────────────────────────────────────────────────────────── #
ui.section('Coming up', len(events), f'Next {COMING_UP_DAYS} days')
if events.empty:
    ui.empty_state('Nothing scheduled', 'Entry windows, near-setups and your sell-by dates show up here.',
                   'event_upcoming')
else:
    st.dataframe(
        pd.DataFrame({
            'Date': pd.to_datetime(events['date']),
            'Stock': events['symbol'].map(ui.short),
            'What': events['event'],
            'Tool': events['tool'].map(lambda t: TOOLS.get(t, {}).get('label', t)),
        }),
        hide_index=True, width='stretch',
        column_config={'Date': st.column_config.DateColumn(format='ddd D MMM')},
    )

# ── Your trades ─────────────────────────────────────────────────────────── #
if email is None:
    ui.section('Your trades')
    ui.empty_state('Log in to track your trades',
                   'Track a trade from any tool and it shows up here with its sell-by countdown.', 'login')
    st.button('Log in with Google', icon=':material/login:', type='primary', on_click=st.login, args=('google',))
else:
    require_premium()
    if positions is None:
        ui.section('Your trades')
        st.warning('Your trades are unavailable right now. Please try again in a few minutes.')
    else:
        render_journal(settings, positions)
        render_telegram(settings)

st.caption('Signals come from historical backtests and a valuation screen. Check dates with IDX before trading. '
           'Results use your fees and are for your own records. Not financial advice.')
