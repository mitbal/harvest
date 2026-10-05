"""Read and write the trading tables: runs, setups, signals and positions.

Every function takes the Supabase client as its first argument, so the
pipeline (supabase-py ``Client``) and the app (``st.connection``) share this
code. Both expose the same ``client.table(...)`` query builder. The module
stays Streamlit-free.

Table definitions live in ``pipeline/premium_trading.sql``.
"""
import datetime
import json
import math
from typing import Iterable, Optional

import holidays
import numpy as np
import pandas as pd


TOOLS = ('dividend_play', 'swing', 'long_term')
RUN_KINDS = ('optimize', 'scan')
RUN_STATUSES = ('running', 'ok', 'failed')
SIGNAL_STATUSES = ('actionable', 'opening_soon', 'candidate')

SIGNAL_COLUMNS = [
    'tool', 'symbol', 'as_of', 'status', 'entry_price', 'stop_price', 'target_price',
    'exit_by', 'key_date', 'key_date_estimated', 'setup_scope', 'detail',
]
SIGNAL_DATE_COLUMNS = ('as_of', 'exit_by', 'key_date')
POSITION_COLUMNS = [
    'id', 'email', 'tool', 'symbol', 'entry_date', 'entry_price', 'lots', 'exit_rule',
    'exit_by', 'closed_at', 'exit_price', 'created_at',
]
POSITION_DATE_COLUMNS = ('entry_date', 'exit_by', 'closed_at')
POSITION_FIELDS = {'tool', 'symbol', 'entry_date', 'entry_price', 'lots', 'exit_rule', 'exit_by'}
POSITION_UPDATABLE = {'entry_date', 'entry_price', 'lots', 'exit_rule', 'exit_by', 'closed_at', 'exit_price'}


# ── Value helpers ─────────────────────────────────────────────────────────── #

def _check(value: str, allowed: Iterable[str], name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}; got {value!r}")
    return value


def _to_date(value) -> Optional[datetime.date]:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return pd.Timestamp(value).date()


def _jsonable(value):
    """Plain JSON value: dates as ISO strings, numpy scalars unwrapped, NaN and inf as None."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if pd.isna(value):
        return None
    return json.loads(json.dumps(value, default=str))


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _frame(rows: list[dict], columns: list[str], date_columns: Iterable[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
    for column in date_columns:
        frame[column] = pd.to_datetime(frame[column], errors='coerce').dt.date
    return frame


def _email(email: str) -> str:
    if not email or not email.strip():
        raise ValueError('email is required')
    return email.strip().lower()


# ── IDX trading days ──────────────────────────────────────────────────────── #

def idx_closed_days(years: Iterable[int]) -> set[datetime.date]:
    """Weekday closures: public holidays, cuti bersama and the 31 Dec bourse holiday."""
    years = list(years)
    closed = set(holidays.Indonesia(years=years, categories=('public', 'government')))
    closed.update(datetime.date(year, 12, 31) for year in years)
    return closed


def idx_trading_days(start, end) -> list[datetime.date]:
    """IDX trading days from ``start`` to ``end``, both inclusive."""
    start, end = _to_date(start), _to_date(end)
    if start > end:
        return []
    closed = idx_closed_days(range(start.year, end.year + 1))
    return [
        day.date() for day in pd.bdate_range(start, end)
        if day.date() not in closed
    ]


# ── Runs ──────────────────────────────────────────────────────────────────── #

def start_run(client, tool: str, kind: str, as_of) -> int:
    """Record a run as ``running`` and return its id."""
    row = {
        'tool': _check(tool, TOOLS, 'tool'),
        'kind': _check(kind, RUN_KINDS, 'kind'),
        'as_of': _to_date(as_of).isoformat(),
        'status': 'running',
        'started_at': _now(),
    }
    data = client.table('trading_runs').insert(row).execute().data
    return int(data[0]['id'])


def finish_run(client, run_id: int, status: str = 'ok', stats: Optional[dict] = None) -> None:
    """Mark a run ``ok`` or ``failed`` with optional coverage stats."""
    _check(status, ('ok', 'failed'), 'status')
    (
        client.table('trading_runs')
        .update({'status': status, 'stats': _jsonable(stats or {}), 'finished_at': _now()})
        .eq('id', run_id)
        .execute()
    )


def latest_run(client, tool: str, kind: str, status: Optional[str] = 'ok') -> Optional[dict]:
    """Most recent run by ``as_of`` (then id); ``status=None`` includes running and failed runs."""
    query = client.table('trading_runs').select('*').eq('tool', tool).eq('kind', kind)
    if status is not None:
        query = query.eq('status', status)
    rows = query.order('as_of', desc=True).order('id', desc=True).limit(1).execute().data
    return rows[0] if rows else None


def is_stale(
    client,
    tool: str,
    max_trading_days: int = 1,
    today=None,
) -> bool:
    """Whether the latest successful scan is missing ``max_trading_days`` or more sessions.

    The scan for a session runs that night, so today's session does not count
    yet: with the default, Tuesday shows Monday's scan as fresh, and Wednesday
    shows it as stale. No successful scan at all is stale.
    """
    run = latest_run(client, tool, 'scan')
    if run is None:
        return True
    as_of = _to_date(run['as_of'])
    today = _to_date(today) or datetime.date.today()
    missed = [day for day in idx_trading_days(as_of, today) if as_of < day < today]
    return len(missed) >= max_trading_days


# ── Setups ────────────────────────────────────────────────────────────────── #

def write_setups(client, tool: str, setups: list[dict], run_id: Optional[int] = None) -> int:
    """Replace the tool's published setups with ``setups``; returns the number written.

    Each setup has ``scope`` ('global' or a symbol), ``params`` and ``holdout``.
    Scopes missing from ``setups`` are deleted, so a stock that no longer
    passes the gates loses its per-stock setup.
    """
    _check(tool, TOOLS, 'tool')
    rows = [
        {
            'tool': tool,
            'scope': str(setup['scope']),
            'params': _jsonable(setup['params']),
            'holdout': _jsonable(setup['holdout']),
            'run_id': run_id,
        }
        for setup in setups
    ]
    if rows:
        client.table('trading_setups').upsert(rows, on_conflict='tool,scope').execute()
    stale = client.table('trading_setups').delete().eq('tool', tool)
    scopes = [row['scope'] for row in rows]
    if scopes:
        stale = stale.not_.in_('scope', scopes)
    stale.execute()
    return len(rows)


def read_setups(client, tool: str) -> dict[str, dict]:
    """Published setups keyed by scope: ``{'global': {...}, 'BBRI.JK': {...}}``."""
    rows = client.table('trading_setups').select('*').eq('tool', tool).execute().data or []
    return {
        row['scope']: {'params': row['params'], 'holdout': row['holdout'], 'run_id': row.get('run_id')}
        for row in rows
    }


# ── Signals ───────────────────────────────────────────────────────────────── #

def latest_signal_date(client, tool: str) -> Optional[datetime.date]:
    rows = (
        client.table('trading_signals').select('as_of').eq('tool', tool)
        .order('as_of', desc=True).limit(1).execute().data
    )
    return _to_date(rows[0]['as_of']) if rows else None


def write_signals(client, tool: str, as_of, signals: list[dict]) -> int:
    """Publish the signals of one scan day; returns the number written.

    Earlier days are the public track record and are never touched: writing a
    day older than the latest stored one raises. Rerunning the same day
    replaces that day's rows, including removing symbols no longer signalled.
    """
    _check(tool, TOOLS, 'tool')
    as_of = _to_date(as_of)
    latest = latest_signal_date(client, tool)
    if latest is not None and as_of < latest:
        raise ValueError(f'{tool} already has signals for {latest}; refusing to write {as_of}')

    rows = []
    for signal in signals:
        row = {column: _jsonable(signal.get(column)) for column in SIGNAL_COLUMNS}
        row.update({'tool': tool, 'as_of': as_of.isoformat(), 'symbol': str(signal['symbol'])})
        _check(row['status'], SIGNAL_STATUSES, 'status')
        rows.append(row)
    symbols = [row['symbol'] for row in rows]
    if len(set(symbols)) != len(symbols):
        raise ValueError('signals contain the same symbol twice')

    if rows:
        client.table('trading_signals').upsert(rows, on_conflict='tool,symbol,as_of').execute()
    stale = client.table('trading_signals').delete().eq('tool', tool).eq('as_of', as_of.isoformat())
    if symbols:
        stale = stale.not_.in_('symbol', symbols)
    stale.execute()
    return len(rows)


def read_signals(client, tool: str, as_of=None) -> pd.DataFrame:
    """Signals for one day (default: the latest day with signals), date columns as dates."""
    as_of = _to_date(as_of) or latest_signal_date(client, tool)
    if as_of is None:
        return _frame([], SIGNAL_COLUMNS, SIGNAL_DATE_COLUMNS)
    rows = (
        client.table('trading_signals').select('*')
        .eq('tool', tool).eq('as_of', as_of.isoformat())
        .order('symbol').execute().data or []
    )
    return _frame(rows, SIGNAL_COLUMNS, SIGNAL_DATE_COLUMNS)


# ── Positions ─────────────────────────────────────────────────────────────── #
# Every query filters by the owner's email, so one user can never read or change
# another user's positions even when given their id.

def create_position(client, email: str, position: dict) -> dict:
    """Track a new open position; returns the stored row."""
    missing = {'tool', 'symbol', 'entry_date', 'entry_price', 'lots', 'exit_rule'} - set(position)
    if missing:
        raise ValueError(f"position is missing {', '.join(sorted(missing))}")
    unknown = set(position) - POSITION_FIELDS
    if unknown:
        raise ValueError(f"unknown position fields: {', '.join(sorted(unknown))}")
    _check(position['tool'], TOOLS, 'tool')
    if int(position['lots']) <= 0:
        raise ValueError('lots must be positive')
    row = {key: _jsonable(value) for key, value in position.items()}
    row['email'] = _email(email)
    row['lots'] = int(position['lots'])
    return client.table('positions').insert(row).execute().data[0]


def list_positions(client, email: str, open_only: bool = False) -> pd.DataFrame:
    """The user's positions, open first by ``exit_by`` then closed by ``closed_at``."""
    query = client.table('positions').select('*').eq('email', _email(email))
    if open_only:
        query = query.is_('closed_at', 'null')
    rows = query.execute().data or []
    frame = _frame(rows, POSITION_COLUMNS, POSITION_DATE_COLUMNS)
    if frame.empty:
        return frame
    frame['_closed'] = frame['closed_at'].notna()
    frame = frame.sort_values(
        ['_closed', 'exit_by', 'closed_at', 'id'], ascending=[True, True, False, True], na_position='last'
    )
    return frame.drop(columns='_closed').reset_index(drop=True)


def update_position(client, email: str, position_id: int, changes: dict) -> Optional[dict]:
    """Change editable fields of one of the user's positions; None if it isn't theirs."""
    unknown = set(changes) - POSITION_UPDATABLE
    if unknown:
        raise ValueError(f"cannot update {', '.join(sorted(unknown))}")
    if not changes:
        raise ValueError('nothing to update')
    if 'lots' in changes and int(changes['lots']) <= 0:
        raise ValueError('lots must be positive')
    rows = (
        client.table('positions')
        .update({key: _jsonable(value) for key, value in changes.items()})
        .eq('id', position_id).eq('email', _email(email))
        .execute().data
    )
    return rows[0] if rows else None


def close_position(client, email: str, position_id: int, closed_at, exit_price: float) -> Optional[dict]:
    """Record the exit of one of the user's positions."""
    return update_position(
        client, email, position_id, {'closed_at': _to_date(closed_at), 'exit_price': float(exit_price)}
    )


def delete_position(client, email: str, position_id: int) -> bool:
    """Remove one of the user's positions; False if it isn't theirs."""
    rows = (
        client.table('positions').delete()
        .eq('id', position_id).eq('email', _email(email))
        .execute().data
    )
    return bool(rows)


# ── User settings and alerts ──────────────────────────────────────────────── #

SETTINGS_DEFAULTS = {
    'buy_fee_pct': 0.15,
    'sell_fee_pct': 0.25,
    'capital_idr': None,
    'risk_pct': 1.0,
    'telegram_chat_id': None,
    'telegram_link_token': None,
    'alert_prefs': {},
}
SETTINGS_UPDATABLE = set(SETTINGS_DEFAULTS)


def get_settings(client, email: str) -> dict:
    """The user's settings, with defaults for anything not saved yet."""
    rows = client.table('user_settings').select('*').eq('email', _email(email)).execute().data
    saved = rows[0] if rows else {}
    settings = {**SETTINGS_DEFAULTS, **{k: v for k, v in saved.items() if v is not None}}
    settings['alert_prefs'] = dict(settings.get('alert_prefs') or {})
    settings['email'] = _email(email)
    return settings


def save_settings(client, email: str, changes: dict) -> dict:
    """Upsert some of the user's settings; returns the stored row."""
    unknown = set(changes) - SETTINGS_UPDATABLE
    if unknown:
        raise ValueError(f"cannot update {', '.join(sorted(unknown))}")
    row = {key: _jsonable(value) for key, value in changes.items()}
    row['email'] = _email(email)
    return client.table('user_settings').upsert(row, on_conflict='email').execute().data[0]


def settings_for_link_token(client, token: str) -> Optional[dict]:
    """The settings row waiting to be linked to Telegram with this token."""
    if not token:
        return None
    rows = client.table('user_settings').select('*').eq('telegram_link_token', token).execute().data
    return rows[0] if rows else None


def settings_for_chat(client, chat_id: str) -> list[dict]:
    return client.table('user_settings').select('*').eq('telegram_chat_id', str(chat_id)).execute().data or []


def alert_subscribers(client) -> list[dict]:
    """Every user with a linked Telegram chat (server side only)."""
    rows = client.table('user_settings').select('*').execute().data or []
    return [row for row in rows if row.get('telegram_chat_id')]


def open_positions_all(client, tool: str) -> pd.DataFrame:
    """Open positions of every user for one tool (server side only)."""
    rows = (
        client.table('positions').select('*').eq('tool', tool).is_('closed_at', 'null')
        .execute().data or []
    )
    return _frame(rows, POSITION_COLUMNS, POSITION_DATE_COLUMNS)


def log_alert(client, email: str, kind: str, ref: str) -> bool:
    """Record an alert; False when the same alert was already sent (one alert per event)."""
    inserted = client.table('alert_log').upsert(
        {'email': _email(email), 'kind': kind, 'ref': ref},
        on_conflict='email,kind,ref', ignore_duplicates=True,
    ).execute().data
    return bool(inserted)


def forget_alert(client, email: str, kind: str, ref: str) -> None:
    """Undo ``log_alert`` when sending failed, so the next run retries."""
    client.table('alert_log').delete().eq('email', _email(email)).eq('kind', kind).eq('ref', ref).execute()


def position_result(position, buy_fee_pct: float, sell_fee_pct: float, dividend_tax_pct: float = 10.0) -> dict:
    """Return after fees of a closed position, adding the dividend when it was held past the cum date.

    ``position`` is a row of :func:`list_positions`. Returns ``return_pct`` and
    ``profit`` (IDR, 100 shares per lot); NaN for open positions.
    """
    if position['closed_at'] is None or pd.isna(position['closed_at']) or position['exit_price'] is None:
        return {'return_pct': np.nan, 'profit': np.nan, 'dividend': 0.0}
    rule = position['exit_rule'] or {}
    dividend = 0.0
    key_date = _to_date(rule.get('key_date'))
    if rule.get('next_dividend') and key_date and _to_date(position['closed_at']) > key_date:
        dividend = float(rule['next_dividend']) * (1 - dividend_tax_pct / 100)
    cost = float(position['entry_price']) * (1 + buy_fee_pct / 100)
    proceeds = float(position['exit_price']) * (1 - sell_fee_pct / 100) + dividend
    shares = int(position['lots']) * 100
    return {
        'return_pct': (proceeds / cost - 1) * 100,
        'profit': (proceeds - cost) * shares,
        'dividend': dividend * shares,
    }


def sessions_until(exit_by, today=None) -> int:
    """IDX trading sessions from today through ``exit_by`` (0 once it has passed)."""
    today = _to_date(today) or datetime.date.today()
    exit_by = _to_date(exit_by)
    if exit_by is None or exit_by < today:
        return 0
    return len(idx_trading_days(today, exit_by))
