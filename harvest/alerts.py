"""Telegram alerts sent by the nightly scan.

Alerts, each sent at most once per event (``alert_log``):

- ``signal``: an entry window opened for a stock the user watches ("Alert me")
  or, when they opted into new signals, any stock.
- ``exit``: a tracked position must be sold by the next trading session, or
  a Long Term position's reason to hold is gone (P/E back at its historical
  mean, or a new quarter turned trailing growth negative).

Streamlit-free; the bot token comes from ``TELEGRAM_BOT_TOKEN``.
"""
import datetime
import logging
import os
from typing import Callable, Optional

import pandas as pd
import requests

import harvest.trading_store as store
from harvest.donations import active_supporters


logger = logging.getLogger(__name__)

TELEGRAM_API = 'https://api.telegram.org/bot{token}/{method}'
SITE_URL = 'https://panendividen.com'
TOOL = 'dividend_play'
EXIT_REMINDER_TOOLS = ('dividend_play', 'swing')


def bot_token() -> Optional[str]:
    return os.environ.get('TELEGRAM_BOT_TOKEN')


def call_telegram(method: str, payload: dict, token: Optional[str] = None) -> dict:
    """POST to the Bot API; raises on HTTP errors or ``ok: false``."""
    token = token or bot_token()
    if not token:
        raise RuntimeError('TELEGRAM_BOT_TOKEN is not set')
    response = requests.post(TELEGRAM_API.format(token=token, method=method), json=payload, timeout=15)
    response.raise_for_status()
    body = response.json()
    if not body.get('ok'):
        raise RuntimeError(f"Telegram {method} failed: {body.get('description')}")
    return body


def send_message(chat_id, text: str, token: Optional[str] = None) -> None:
    call_telegram('sendMessage', {'chat_id': chat_id, 'text': text, 'disable_web_page_preview': True}, token)


def _day(value, estimated: bool = False) -> str:
    date = pd.Timestamp(value)
    return f"{'~' if estimated else ''}{date:%a} {date.day} {date:%b}"


def _short(symbol: str) -> str:
    return symbol.removesuffix('.JK')


def signal_message(signal: dict) -> str:
    detail = signal.get('detail') or {}
    estimated = bool(signal.get('key_date_estimated'))
    holdout = detail.get('holdout') or {}
    lines = [
        f"Dividend Play: the entry window for {_short(signal['symbol'])} is open.",
        f"Entry at or below Rp {signal['entry_price']:,.0f}, sell by {_day(signal['exit_by'], estimated)}, "
        f"cum date {_day(signal['key_date'], estimated)}.",
    ]
    if holdout.get('trades'):
        lines.append(f"Holdout record: {holdout.get('wins')} of {holdout['trades']} trades won.")
    lines.append(f"{SITE_URL}/dividend_play?stock={signal['symbol']}")
    lines.append('Historical backtest, not financial advice.')
    return '\n'.join(lines)


def exit_message(position: pd.Series, next_session: datetime.date) -> str:
    exit_by = pd.Timestamp(position['exit_by']).date()
    when = 'the next session' if exit_by == next_session else 'this date has passed'
    return '\n'.join([
        f"Reminder: sell {_short(position['symbol'])} by {_day(exit_by)} ({when}).",
        f"You tracked {int(position['lots'])} lots bought at Rp {float(position['entry_price']):,.0f}.",
        f"{SITE_URL}/my_trades",
    ])


def _next_session(as_of) -> datetime.date:
    start = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    return store.idx_trading_days(start, start + pd.Timedelta(days=14))[0]


def send_dividend_play_alerts(
    client,
    as_of,
    signals: list[dict],
    token: Optional[str] = None,
    send: Callable[[str, str, Optional[str]], None] = send_message,
) -> dict:
    """Send tonight's signal and exit alerts; returns counts. Never raises for one bad chat."""
    token = token or bot_token()
    if not token:
        return {'skipped': 'TELEGRAM_BOT_TOKEN is not set'}
    stats = {'signal': 0, 'exit': 0, 'already_sent': 0, 'failed': 0}

    def deliver(subscriber: dict, kind: str, ref: str, text: str) -> None:
        email = subscriber['email']
        if not store.log_alert(client, email, kind, ref):
            stats['already_sent'] += 1
            return
        try:
            send(subscriber['telegram_chat_id'], text, token)
            stats[kind] += 1
        except Exception:
            logger.exception('Telegram %s alert failed for one subscriber', kind)
            store.forget_alert(client, email, kind, ref)
            stats['failed'] += 1

    subscribers = {row['email']: row for row in store.alert_subscribers(client)}
    allowed = active_supporters(client)
    actionable = [signal for signal in signals if signal['status'] == 'actionable']
    for email, subscriber in subscribers.items():
        if allowed is not None and email not in allowed:
            continue
        prefs = subscriber.get('alert_prefs') or {}
        watched = set(prefs.get('watch') or [])
        for signal in actionable:
            if prefs.get('new_signals') or signal['symbol'] in watched:
                window = (signal.get('detail') or {}).get('window_start', '')
                deliver(subscriber, 'signal', f"{TOOL}:{signal['symbol']}:{window}", signal_message(signal))

    # Exit reminders go to anyone who tracked the trade, supporter or not, for every tool.
    next_session = _next_session(as_of)
    positions = pd.concat([store.open_positions_all(client, tool) for tool in EXIT_REMINDER_TOOLS], ignore_index=True)
    for _, position in positions.iterrows():
        subscriber = subscribers.get(position['email'])
        if subscriber is None or position['exit_by'] is None or pd.isna(position['exit_by']):
            continue
        if pd.Timestamp(position['exit_by']).date() <= next_session:
            ref = f"{position['id']}:{pd.Timestamp(position['exit_by']).date().isoformat()}"
            deliver(subscriber, 'exit', ref, exit_message(position, next_session))
    return stats


def long_term_exit_reasons(position: pd.Series, row: dict) -> list[tuple[str, str]]:
    """(alert ref, message line) for each exit condition a Long Term position meets today."""
    reasons = []
    symbol = _short(position['symbol'])
    pe, pe_mean = row.get('Current PE'), row.get('PE Mean')
    if pe is not None and pe_mean is not None and pd.notna(pe) and pd.notna(pe_mean) and pe >= pe_mean:
        reasons.append((
            f"{position['id']}:pe_mean",
            f"Long Term: {symbol}'s P/E is back at its historical mean ({pe:.1f} vs a mean of {pe_mean:.1f}). "
            "The discount you bought has closed.",
        ))
    statement = row.get('Statement Date')
    entry = position.get('entry_date')
    if statement is not None and pd.notna(statement) and entry is not None and pd.notna(entry) \
            and pd.Timestamp(statement).date() > pd.Timestamp(entry).date():
        falling = [
            f"{label} {row[column]:+.1f}%"
            for label, column in (('earnings', 'Earnings Growth TTM'), ('revenue', 'Revenue Growth TTM'))
            if pd.notna(row.get(column)) and row[column] < 0
        ]
        if falling:
            reasons.append((
                f"{position['id']}:growth:{pd.Timestamp(statement).date().isoformat()}",
                f"Long Term: {symbol}'s latest quarter ({_day(statement)} {pd.Timestamp(statement).year}) turned "
                f"trailing twelve-month growth negative ({', '.join(falling)}).",
            ))
    return reasons


def long_term_exit_message(position: pd.Series, line: str) -> str:
    return '\n'.join([
        line,
        f"You tracked {int(position['lots'])} lots bought at Rp {float(position['entry_price']):,.0f}. "
        "Review whether the reason you bought still holds.",
        f"{SITE_URL}/my_trades",
        'Screen result, not financial advice.',
    ])


def send_long_term_exit_alerts(
    client,
    as_of,
    rows: dict[str, dict],
    positions: pd.DataFrame,
    token: Optional[str] = None,
    send: Callable[[str, str, Optional[str]], None] = send_message,
) -> dict:
    """Exit alerts for open Long Term positions from tonight's screen rows (keyed by symbol)."""
    token = token or bot_token()
    if not token:
        return {'skipped': 'TELEGRAM_BOT_TOKEN is not set'}
    stats = {'exit': 0, 'already_sent': 0, 'failed': 0}
    subscribers = {row['email']: row for row in store.alert_subscribers(client)}
    for _, position in positions.iterrows():
        subscriber = subscribers.get(position['email'])
        row = rows.get(position['symbol'])
        if subscriber is None or row is None:
            continue
        for ref, line in long_term_exit_reasons(position, row):
            if not store.log_alert(client, position['email'], 'exit', ref):
                stats['already_sent'] += 1
                continue
            try:
                send(subscriber['telegram_chat_id'], long_term_exit_message(position, line), token)
                stats['exit'] += 1
            except Exception:
                logger.exception('Telegram Long Term exit alert failed for one subscriber')
                store.forget_alert(client, position['email'], 'exit', ref)
                stats['failed'] += 1
    return stats


# ── Linking a chat ──────────────────────────────────────────────────────── #

def _mask(email: str) -> str:
    name, _, domain = email.partition('@')
    return f"{name[:2]}{'*' * max(len(name) - 2, 1)}@{domain}"


def link_chat(client, token: str, chat_id) -> str:
    """Attach a Telegram chat to the account that created ``token``; returns the reply text."""
    settings = store.settings_for_link_token(client, token)
    if settings is None:
        return ('This link has expired. Open My Trades on panendividen.com and press '
                '"Link Telegram" to get a new one.')
    store.save_settings(client, settings['email'], {'telegram_chat_id': str(chat_id), 'telegram_link_token': None})
    return f"Linked. Dividend Play alerts for {_mask(settings['email'])} will arrive here. Send /stop to turn them off."


def unlink_chat(client, chat_id) -> str:
    for settings in store.settings_for_chat(client, chat_id):
        store.save_settings(client, settings['email'], {'telegram_chat_id': None})
    return 'Alerts stopped. You can link again from My Trades on panendividen.com.'
