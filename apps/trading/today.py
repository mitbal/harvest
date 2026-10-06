"""Trading Today: tonight's signals from every trading tool, your open positions and what is coming up.

The page only reads the published store (``trading_signals``, ``trading_runs``, ``positions``);
the cards are short versions of each tool's Today cards and link to the tool for the full card.
"""
import datetime
import logging

import pandas as pd
import streamlit as st

import harvest.position as pos
import harvest.trading_store as store
import harvest.trading_ui as ui
from harvest.premium import current_email, get_connection, has_access


TOOLS = {
    'dividend_play': {'label': 'Dividend Play', 'icon': 'payments', 'page': 'apps/trading/dividend_play.py',
                      'note': 'Entry window open now'},
    'swing': {'label': 'Swing Trading', 'icon': 'bolt', 'page': 'apps/trading/swing_trading.py',
              'note': 'Buy at the next open'},
    'long_term': {'label': 'Long Term', 'icon': 'trending_up', 'page': 'apps/trading/backtester.py',
                  'note': 'Top candidates of the valuation screen'},
}
MY_TRADES_PAGE = 'apps/trading/my_trades.py'
# Full-access cards per tool; the tool page has the rest.
CARDS_PER_TOOL = {'dividend_play': 4, 'swing': 4, 'long_term': 3}
COMING_UP_DAYS = 45
SELL_SOON_SESSIONS = 5
logger = logging.getLogger(__name__)

st.set_page_config(page_title='Trading Today - Panen Dividen')
ui.inject_styles()
ui.hero('Trading · Today', 'Trading today',
        'What to act on after the last close across Dividend Play, Swing Trading and Long Term, '
        'your open positions and the dates coming up.', 'today')


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


@st.cache_data(ttl=60, show_spinner=False)
def load_open_positions(email: str) -> pd.DataFrame:
    return store.list_positions(get_connection(), email, open_only=True)


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


def _countdown(position: pd.Series) -> str:
    if (position['exit_rule'] or {}).get('kind') == 'valuation':
        return 'Watching valuation'
    if position['exit_by'] is None or pd.isna(position['exit_by']):
        return 'No sell-by date'
    sessions = store.sessions_until(position['exit_by'])
    if sessions == 0:
        return 'Sell-by date has passed'
    if sessions == 1:
        return 'Sell today'
    return f"{sessions} sessions left, sell by {ui.day(position['exit_by'])}"


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE
# ═══════════════════════════════════════════════════════════════════════════════

access = has_access()
email = current_email()
published = {tool: load_tool(tool) for tool in TOOLS}
if all(data is None for data in published.values()):
    st.warning('Signals are unavailable right now. Please try again in a few minutes.')
    st.stop()

ranked = {tool: act_now(tool, (data or {}).get('signals')) for tool, data in published.items()}
positions = None
if email:
    try:
        positions = load_open_positions(email)
    except Exception:
        logger.exception('Could not load positions')
events = coming_up(published, positions)

metrics = st.container(horizontal=True)
metrics.metric('Signals today', len(ranked['dividend_play']) + len(ranked['swing']),
               help='Dividend Play entry windows open now plus swing setups for the next open')
metrics.metric('Long term candidates', len(ranked['long_term']))
if positions is None:
    metrics.metric('Your positions', '–', help='Log in to track your trades' if email is None else None)
else:
    selling = int(sum(0 < store.sessions_until(by) <= SELL_SOON_SESSIONS
                      for by in positions['exit_by'].dropna()))
    metrics.metric('Your positions', len(positions),
                   delta=f'{selling} to sell within {SELL_SOON_SESSIONS} sessions' if selling else None,
                   delta_color='off', delta_arrow='off')
metrics.metric(f'Coming up in {COMING_UP_DAYS} days', len(events))

# ── Act now ─────────────────────────────────────────────────────────────── #
ui.section('Act now', sum(len(rows) for rows in ranked.values()), 'From the scans after the last close')
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

# ── Your positions ──────────────────────────────────────────────────────── #
ui.section('Your positions', None if positions is None else len(positions), 'Soonest sell-by first')
if email is None:
    ui.empty_state('Log in to see your positions',
                   'Track a trade from any tool and it shows up here with its sell-by countdown.', 'login')
    st.button('Log in with Google', icon=':material/login:', on_click=st.login, args=('google',), key='hub_login')
elif positions is None:
    st.warning('Your positions are unavailable right now. Please try again in a few minutes.')
elif positions.empty:
    ui.empty_state('No open positions', 'Use "Track this trade" on a tool\'s card to add one.', 'bookmark_add')
else:
    countdowns = positions.apply(_countdown, axis=1)
    for symbol, text in zip(positions['symbol'], countdowns):
        if text in ('Sell today', 'Sell-by date has passed'):
            ui.render(ui.strip(f'<b>{ui.esc(ui.short(symbol))}</b>: {ui.esc(text)}', 'warn', 'alarm'))
    st.dataframe(
        pd.DataFrame({
            'Stock': positions['symbol'].map(ui.short),
            'Tool': positions['tool'].map(lambda t: TOOLS.get(t, {}).get('label', t)),
            'Bought': pd.to_datetime(positions['entry_date']),
            'Lots': positions['lots'].astype(int),
            'Buy price': positions['entry_price'].astype(float),
            'Exit': countdowns,
        }),
        hide_index=True, width='stretch',
        column_config={
            'Bought': st.column_config.DateColumn(format='D MMM YYYY'),
            'Buy price': st.column_config.NumberColumn(format='%,.0f'),
        },
    )
    if st.button('Open My Trades', icon=':material/menu_book:', key='hub_my_trades'):
        st.switch_page(MY_TRADES_PAGE)

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

st.caption('Signals come from historical backtests and a valuation screen. Check dates with IDX before trading. '
           'Not financial advice.')
