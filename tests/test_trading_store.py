import datetime

import numpy as np
import pandas as pd
import pytest

import harvest.trading_store as ts
from tests.fake_supabase import FakeSupabase


@pytest.fixture
def client():
    return FakeSupabase()


def _signal(symbol, status='actionable', **extra):
    return {'symbol': symbol, 'status': status, **extra}


def _setup(scope, win_rate=60.0):
    return {'scope': scope, 'params': {'entry_start_days': 10}, 'holdout': {'win_rate': win_rate}}


def _position(**overrides):
    position = {
        'tool': 'dividend_play',
        'symbol': 'BBRI.JK',
        'entry_date': datetime.date(2026, 10, 1),
        'entry_price': 4000,
        'lots': 10,
        'exit_rule': {'sell_on': 'cum_date'},
        'exit_by': datetime.date(2026, 10, 20),
    }
    position.update(overrides)
    return position


# ── Trading days ─────────────────────────────────────────────────────────── #

def test_idx_trading_days_skip_weekends_holidays_and_year_end():
    days = ts.idx_trading_days('2026-12-24', '2027-01-04')

    assert datetime.date(2026, 12, 25) not in days  # Christmas
    assert datetime.date(2026, 12, 26) not in days  # Saturday
    assert datetime.date(2026, 12, 31) not in days  # IDX year-end holiday
    assert datetime.date(2027, 1, 1) not in days  # New Year
    assert datetime.date(2026, 12, 28) in days
    assert days[-1] == datetime.date(2027, 1, 4)


def test_idx_trading_days_empty_when_start_after_end():
    assert ts.idx_trading_days('2026-10-06', '2026-10-05') == []


# ── Runs ─────────────────────────────────────────────────────────────────── #

def test_run_lifecycle(client):
    run_id = ts.start_run(client, 'dividend_play', 'scan', '2026-10-05')
    assert ts.latest_run(client, 'dividend_play', 'scan') is None
    assert ts.latest_run(client, 'dividend_play', 'scan', status=None)['status'] == 'running'

    ts.finish_run(client, run_id, stats={'analyzed': np.int64(42), 'coverage': float('nan')})
    run = ts.latest_run(client, 'dividend_play', 'scan')

    assert run['id'] == run_id
    assert run['as_of'] == '2026-10-05'
    assert run['status'] == 'ok'
    assert run['stats'] == {'analyzed': 42, 'coverage': None}
    assert run['finished_at'] is not None


def test_latest_run_prefers_newest_as_of_then_id(client):
    first = ts.start_run(client, 'swing', 'scan', '2026-10-02')
    second = ts.start_run(client, 'swing', 'scan', '2026-10-05')
    rerun = ts.start_run(client, 'swing', 'scan', '2026-10-05')
    other_tool = ts.start_run(client, 'dividend_play', 'scan', '2026-10-06')
    for run_id in (first, second, rerun, other_tool):
        ts.finish_run(client, run_id)
    ts.finish_run(client, rerun, status='failed')

    assert ts.latest_run(client, 'swing', 'scan')['id'] == second
    assert ts.latest_run(client, 'swing', 'scan', status='failed')['id'] == rerun
    assert ts.latest_run(client, 'swing', 'optimize') is None


def test_run_arguments_are_validated(client):
    with pytest.raises(ValueError, match='tool'):
        ts.start_run(client, 'crypto', 'scan', '2026-10-05')
    with pytest.raises(ValueError, match='kind'):
        ts.start_run(client, 'swing', 'backtest', '2026-10-05')
    with pytest.raises(ValueError, match='status'):
        ts.finish_run(client, 1, status='running')


# ── Staleness ────────────────────────────────────────────────────────────── #

def _scan(client, as_of, status='ok'):
    ts.finish_run(client, ts.start_run(client, 'dividend_play', 'scan', as_of), status=status)


def test_is_stale_without_any_scan(client):
    assert ts.is_stale(client, 'dividend_play', today='2026-10-06')


@pytest.mark.parametrize('today, expected', [
    ('2026-10-05', False),  # Monday: the Monday scan only runs tonight
    ('2026-10-06', False),  # Tuesday: Monday's scan is the latest expected
    ('2026-10-07', True),   # Wednesday: Tuesday's scan is missing
])
def test_is_stale_counts_missed_sessions(client, today, expected):
    _scan(client, '2026-10-05')

    assert ts.is_stale(client, 'dividend_play', today=today) is expected


def test_is_stale_skips_weekends_and_holidays(client):
    _scan(client, '2026-12-24')  # Thursday before Christmas

    # Fri 25 Dec is a holiday and the weekend follows, so Monday still shows Thursday's scan.
    assert not ts.is_stale(client, 'dividend_play', today='2026-12-28')
    assert ts.is_stale(client, 'dividend_play', today='2026-12-29')


def test_is_stale_honours_tolerance_and_ignores_failed_runs(client):
    _scan(client, '2026-10-05')
    _scan(client, '2026-10-07', status='failed')

    assert ts.is_stale(client, 'dividend_play', today='2026-10-08')
    assert not ts.is_stale(client, 'dividend_play', max_trading_days=3, today='2026-10-08')


# ── Setups ───────────────────────────────────────────────────────────────── #

def test_write_and_read_setups(client):
    run_id = ts.start_run(client, 'dividend_play', 'optimize', '2026-10-04')
    written = ts.write_setups(client, 'dividend_play', [_setup('global'), _setup('BBRI.JK', np.float64(75.0))], run_id)
    ts.write_setups(client, 'swing', [_setup('global')])

    setups = ts.read_setups(client, 'dividend_play')

    assert written == 2
    assert set(setups) == {'global', 'BBRI.JK'}
    assert setups['BBRI.JK']['holdout'] == {'win_rate': 75.0}
    assert setups['global']['run_id'] == run_id


def test_write_setups_replaces_previous_scopes(client):
    ts.write_setups(client, 'dividend_play', [_setup('global'), _setup('BBRI.JK'), _setup('TLKM.JK')])
    ts.write_setups(client, 'swing', [_setup('BBRI.JK')])

    ts.write_setups(client, 'dividend_play', [_setup('global', 55.0), _setup('BBRI.JK')])

    setups = ts.read_setups(client, 'dividend_play')
    assert set(setups) == {'global', 'BBRI.JK'}
    assert setups['global']['holdout'] == {'win_rate': 55.0}
    assert set(ts.read_setups(client, 'swing')) == {'BBRI.JK'}


def test_write_setups_with_nothing_clears_the_tool(client):
    ts.write_setups(client, 'dividend_play', [_setup('global')])

    assert ts.write_setups(client, 'dividend_play', []) == 0
    assert ts.read_setups(client, 'dividend_play') == {}


# ── Signals ──────────────────────────────────────────────────────────────── #

def test_write_and_read_signals(client):
    ts.write_signals(client, 'dividend_play', datetime.date(2026, 10, 5), [
        _signal('TLKM.JK', 'opening_soon', key_date=pd.Timestamp('2026-11-02'), key_date_estimated=True),
        _signal(
            'BBRI.JK', entry_price=np.float64(4000.0), exit_by=datetime.date(2026, 10, 20),
            detail={'window_day': np.int64(3), 'avg_return': float('inf'), 'yield': 6.1},
        ),
    ])

    signals = ts.read_signals(client, 'dividend_play')

    assert list(signals.columns) == ts.SIGNAL_COLUMNS
    assert signals['symbol'].tolist() == ['BBRI.JK', 'TLKM.JK']
    bbri = signals.iloc[0]
    assert bbri['as_of'] == datetime.date(2026, 10, 5)
    assert bbri['exit_by'] == datetime.date(2026, 10, 20)
    assert bbri['entry_price'] == 4000.0
    assert bbri['detail'] == {'window_day': 3, 'avg_return': None, 'yield': 6.1}
    assert signals.iloc[1]['key_date'] == datetime.date(2026, 11, 2)
    assert bool(signals.iloc[1]['key_date_estimated'])


def test_read_signals_defaults_to_latest_day_and_reads_history(client):
    ts.write_signals(client, 'dividend_play', '2026-10-02', [_signal('ASII.JK')])
    ts.write_signals(client, 'dividend_play', '2026-10-05', [_signal('BBRI.JK')])

    assert ts.read_signals(client, 'dividend_play')['symbol'].tolist() == ['BBRI.JK']
    assert ts.read_signals(client, 'dividend_play', as_of='2026-10-02')['symbol'].tolist() == ['ASII.JK']
    assert ts.read_signals(client, 'swing').empty


def test_rerunning_a_day_replaces_only_that_day(client):
    ts.write_signals(client, 'dividend_play', '2026-10-02', [_signal('ASII.JK'), _signal('BBRI.JK')])
    ts.write_signals(client, 'dividend_play', '2026-10-05', [_signal('BBRI.JK'), _signal('TLKM.JK')])

    ts.write_signals(client, 'dividend_play', '2026-10-05', [_signal('BBRI.JK', 'opening_soon')])

    today = ts.read_signals(client, 'dividend_play')
    assert today['symbol'].tolist() == ['BBRI.JK']
    assert today['status'].tolist() == ['opening_soon']
    history = ts.read_signals(client, 'dividend_play', as_of='2026-10-02')
    assert history['symbol'].tolist() == ['ASII.JK', 'BBRI.JK']


def test_past_signal_days_are_never_overwritten(client):
    ts.write_signals(client, 'dividend_play', '2026-10-05', [_signal('BBRI.JK')])

    with pytest.raises(ValueError, match='refusing'):
        ts.write_signals(client, 'dividend_play', '2026-10-02', [_signal('BBRI.JK')])
    # Another tool keeps its own history.
    ts.write_signals(client, 'swing', '2026-10-02', [_signal('BBRI.JK')])


def test_signals_are_validated(client):
    with pytest.raises(ValueError, match='status'):
        ts.write_signals(client, 'dividend_play', '2026-10-05', [_signal('BBRI.JK', 'buy')])
    with pytest.raises(ValueError, match='twice'):
        ts.write_signals(client, 'dividend_play', '2026-10-05', [_signal('BBRI.JK'), _signal('BBRI.JK')])
    assert ts.read_signals(client, 'dividend_play').empty


# ── Positions ────────────────────────────────────────────────────────────── #

def test_position_lifecycle(client):
    created = ts.create_position(client, ' Trader@Example.com', _position())
    assert created['email'] == 'trader@example.com'
    assert created['entry_date'] == '2026-10-01'

    ts.update_position(client, 'trader@example.com', created['id'], {'lots': 12})
    closed = ts.close_position(client, 'trader@example.com', created['id'], '2026-10-19', 4300)

    assert closed['lots'] == 12
    assert closed['closed_at'] == '2026-10-19'
    assert closed['exit_price'] == 4300.0
    assert ts.list_positions(client, 'trader@example.com', open_only=True).empty

    assert ts.delete_position(client, 'trader@example.com', created['id'])
    assert ts.list_positions(client, 'trader@example.com').empty


def test_list_positions_orders_open_by_exit_then_closed(client):
    email = 'trader@example.com'
    late = ts.create_position(client, email, _position(symbol='TLKM.JK', exit_by='2026-11-01'))
    soon = ts.create_position(client, email, _position(symbol='BBRI.JK', exit_by='2026-10-10'))
    no_exit = ts.create_position(client, email, _position(symbol='ASII.JK', exit_by=None))
    old = ts.create_position(client, email, _position(symbol='BMRI.JK'))
    recent = ts.create_position(client, email, _position(symbol='BBNI.JK'))
    ts.close_position(client, email, old['id'], '2026-09-01', 5000)
    ts.close_position(client, email, recent['id'], '2026-09-20', 5000)

    positions = ts.list_positions(client, email)

    assert positions['id'].tolist() == [soon['id'], late['id'], no_exit['id'], recent['id'], old['id']]
    assert positions.loc[0, 'exit_by'] == datetime.date(2026, 10, 10)
    assert ts.list_positions(client, email, open_only=True)['id'].tolist() == [
        soon['id'], late['id'], no_exit['id']
    ]


def test_positions_are_private_to_their_owner(client):
    mine = ts.create_position(client, 'me@example.com', _position())
    ts.create_position(client, 'other@example.com', _position(symbol='TLKM.JK'))

    assert ts.list_positions(client, 'me@example.com')['symbol'].tolist() == ['BBRI.JK']
    assert ts.update_position(client, 'other@example.com', mine['id'], {'lots': 1}) is None
    assert ts.close_position(client, 'other@example.com', mine['id'], '2026-10-05', 1) is None
    assert not ts.delete_position(client, 'other@example.com', mine['id'])
    assert ts.list_positions(client, 'me@example.com').loc[0, 'lots'] == 10


def test_positions_are_validated(client):
    with pytest.raises(ValueError, match='missing exit_rule'):
        ts.create_position(client, 'me@example.com', {k: v for k, v in _position().items() if k != 'exit_rule'})
    with pytest.raises(ValueError, match='unknown position fields: email'):
        ts.create_position(client, 'me@example.com', _position(email='other@example.com'))
    with pytest.raises(ValueError, match='lots'):
        ts.create_position(client, 'me@example.com', _position(lots=0))
    with pytest.raises(ValueError, match='email'):
        ts.create_position(client, '  ', _position())
    position = ts.create_position(client, 'me@example.com', _position())
    with pytest.raises(ValueError, match='cannot update email'):
        ts.update_position(client, 'me@example.com', position['id'], {'email': 'other@example.com'})
    with pytest.raises(ValueError, match='nothing'):
        ts.update_position(client, 'me@example.com', position['id'], {})


# ── Settings, alert log and results ─────────────────────────────────────── #

def test_settings_default_then_saved(client):
    assert ts.get_settings(client, 'Me@Example.com')['buy_fee_pct'] == 0.15

    ts.save_settings(client, 'me@example.com', {'sell_fee_pct': 0.3, 'alert_prefs': {'watch': ['BBRI.JK']}})

    settings = ts.get_settings(client, 'me@example.com')
    assert settings['sell_fee_pct'] == 0.3 and settings['buy_fee_pct'] == 0.15
    assert settings['alert_prefs'] == {'watch': ['BBRI.JK']}
    with pytest.raises(ValueError, match='cannot update email'):
        ts.save_settings(client, 'me@example.com', {'email': 'x@example.com'})


def test_alert_log_records_each_event_once(client):
    assert ts.log_alert(client, 'me@example.com', 'exit', '1:2026-10-06')
    assert not ts.log_alert(client, 'me@example.com', 'exit', '1:2026-10-06')
    ts.forget_alert(client, 'me@example.com', 'exit', '1:2026-10-06')
    assert ts.log_alert(client, 'me@example.com', 'exit', '1:2026-10-06')


def test_link_token_and_subscriber_lookups(client):
    ts.save_settings(client, 'a@example.com', {'telegram_link_token': 'tok'})
    ts.save_settings(client, 'b@example.com', {'telegram_chat_id': '42'})

    assert ts.settings_for_link_token(client, 'tok')['email'] == 'a@example.com'
    assert ts.settings_for_link_token(client, '') is None
    assert [row['email'] for row in ts.alert_subscribers(client)] == ['b@example.com']
    assert ts.settings_for_chat(client, 42)[0]['email'] == 'b@example.com'


def test_open_positions_all_spans_users(client):
    ts.create_position(client, 'a@example.com', _position())
    closed = ts.create_position(client, 'b@example.com', _position())
    ts.close_position(client, 'b@example.com', closed['id'], '2026-10-10', 4100)
    ts.create_position(client, 'c@example.com', _position(tool='swing'))

    assert ts.open_positions_all(client, 'dividend_play')['email'].tolist() == ['a@example.com']


def test_position_result_with_fees_and_dividend():
    base = {'entry_price': 1000.0, 'lots': 2, 'closed_at': datetime.date(2026, 12, 1), 'exit_price': 1100.0,
            'exit_rule': {'key_date': '2026-11-20', 'next_dividend': 50.0}}

    result = ts.position_result(pd.Series(base), buy_fee_pct=0.15, sell_fee_pct=0.25, dividend_tax_pct=10)

    cost, proceeds = 1000 * 1.0015, 1100 * 0.9975 + 45
    assert result['return_pct'] == pytest.approx((proceeds / cost - 1) * 100)
    assert result['profit'] == pytest.approx((proceeds - cost) * 200)
    assert result['dividend'] == pytest.approx(45 * 200)
    before_cum = ts.position_result(pd.Series({**base, 'closed_at': datetime.date(2026, 11, 20)}), 0.15, 0.25)
    assert before_cum['dividend'] == 0.0
    assert np.isnan(ts.position_result(pd.Series({**base, 'closed_at': None}), 0.15, 0.25)['return_pct'])


def test_sessions_until_counts_trading_days():
    assert ts.sessions_until('2026-10-09', today='2026-10-05') == 5
    assert ts.sessions_until('2026-10-05', today='2026-10-05') == 1
    assert ts.sessions_until('2026-10-02', today='2026-10-05') == 0
