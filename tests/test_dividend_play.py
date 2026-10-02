import numpy as np
import pandas as pd

import harvest.dividend_play as dp


def _prices(start='2022-01-01', end='2024-12-31', ex_dates=(), low=90.0, high=110.0):
    """Daily prices that bottom 90 days before each ex-date and peak just before it."""
    dates = pd.bdate_range(start, end)
    close = pd.Series(100.0, index=dates)
    for ex in pd.to_datetime(list(ex_dates)):
        days = (ex - dates).days
        runup = (days > 0) & (days <= 90)
        close[runup] = high - (high - low) * days[runup] / 90
        close[(days <= 0) & (days > -30)] = high - 5
    return pd.DataFrame({'date': dates, 'close': close.values})


def test_cum_exit_captures_runup_without_dividend():
    ex_dates = ['2023-06-15', '2024-06-14']
    prices = _prices(ex_dates=ex_dates)
    divs = pd.DataFrame({'date': ex_dates, 'adjDividend': [5.0, 5.0]})

    trades = dp.backtest_dividend_play(
        prices, divs, entry_start_days=90, entry_end_days=60,
        entry_rule=dp.ENTRY_FIXED, exit_rule=dp.EXIT_CUM,
    )

    assert len(trades) == 2
    assert (trades['exit_date'] < trades['ex_date']).all()
    assert not trades['collected_dividend'].any()
    assert (trades['dividend_return'] == 0).all()
    assert (trades['total_return'] > 15).all()


def test_after_exit_collects_dividend():
    ex_dates = ['2023-06-15']
    prices = _prices(ex_dates=ex_dates)
    divs = pd.DataFrame({'date': ex_dates, 'dividend': [5.0]})

    trades = dp.backtest_dividend_play(
        prices, divs, entry_start_days=90, entry_end_days=60,
        entry_rule=dp.ENTRY_FIXED, exit_rule=dp.EXIT_AFTER, hold_after_days=5,
    )

    trade = trades.iloc[0]
    assert trade['collected_dividend']
    assert trade['exit_date'] >= pd.Timestamp('2023-06-20')
    assert np.isclose(trade['dividend_return'], 5.0 / trade['entry_price'] * 100)
    assert np.isclose(trade['total_return'], trade['price_return'] + trade['dividend_return'])


def test_take_profit_exits_early():
    ex_dates = ['2023-06-15']
    prices = _prices(ex_dates=ex_dates)
    divs = pd.DataFrame({'date': ex_dates, 'dividend': [5.0]})

    trades = dp.backtest_dividend_play(
        prices, divs, entry_start_days=90, entry_end_days=60,
        entry_rule=dp.ENTRY_FIXED, exit_rule=dp.EXIT_CUM, take_profit_pct=5.0,
    )

    trade = trades.iloc[0]
    assert trade['exit_reason'] == 'Take profit'
    assert 5.0 <= trade['total_return'] < 7.0


def test_dip_rule_falls_back_to_window_end():
    dates = pd.bdate_range('2023-01-01', '2023-08-31')
    prices = pd.DataFrame({'date': dates, 'close': np.linspace(100, 150, len(dates))})
    divs = pd.DataFrame({'date': ['2023-07-14'], 'dividend': [2.0]})

    trades = dp.backtest_dividend_play(
        prices, divs, entry_start_days=90, entry_end_days=30,
        entry_rule=dp.ENTRY_DIP, dip_ma_days=20, dip_pct=1.0,
    )

    trade = trades.iloc[0]
    assert not trade['dip_triggered']
    assert trade['entry_date'] <= pd.Timestamp('2023-06-14')
    assert trade['entry_date'] >= pd.Timestamp('2023-06-10')


def test_events_without_enough_history_are_skipped():
    prices = _prices(start='2023-05-01', ex_dates=['2023-06-15'])
    divs = pd.DataFrame({'date': ['2023-06-15'], 'dividend': [5.0]})

    trades = dp.backtest_dividend_play(prices, divs, entry_start_days=90, entry_end_days=30)

    assert trades.empty


def test_estimate_next_ex_dates_rolls_last_year_forward():
    divs = dp.prepare_dividends(pd.DataFrame({
        'date': ['2023-04-10', '2024-04-12', '2024-11-20'],
        'dividend': [10.0, 12.0, 4.0],
    }))

    upcoming = dp.estimate_next_ex_dates(divs, as_of='2025-01-15')

    assert upcoming['expected_date'].tolist() == [pd.Timestamp('2025-04-12'), pd.Timestamp('2025-11-20')]
    assert upcoming['dividend'].tolist() == [12.0, 4.0]


def test_entry_status_phases():
    prices = pd.DataFrame({'date': pd.bdate_range('2025-01-01', periods=60), 'close': 100.0})
    kwargs = dict(entry_start_days=90, entry_end_days=30, dip_ma_days=20, dip_pct=0.0)

    assert dp.entry_status(prices, '2025-06-30', as_of='2025-03-01', **kwargs)['status'] == 'Waiting'
    assert dp.entry_status(prices, '2025-06-30', as_of='2025-05-01', **kwargs)['status'] == 'Buy zone - dip active'
    assert dp.entry_status(prices, '2025-06-30', as_of='2025-06-15', **kwargs)['status'] == 'Run-up phase'


def test_summarize_trades():
    trades = pd.DataFrame({
        'entry_date': pd.to_datetime(['2023-01-01', '2024-01-01']),
        'exit_date': pd.to_datetime(['2023-03-01', '2024-03-01']),
        'total_return': [10.0, -5.0],
        'days_held': [59, 60],
        'max_drawdown': [-2.0, -6.0],
    })

    stats = dp.summarize_trades(trades)

    assert stats['trades'] == 2
    assert stats['win_rate'] == 50.0
    assert np.isclose(stats['compounded'], 4.5)


def test_costs_and_dividend_tax_reduce_returns():
    ex_dates = ['2023-06-15']
    prices = _prices(ex_dates=ex_dates)
    divs = pd.DataFrame({'date': ex_dates, 'dividend': [5.0]})
    kwargs = dict(entry_start_days=90, entry_end_days=60, entry_rule=dp.ENTRY_FIXED,
                  exit_rule=dp.EXIT_AFTER, hold_after_days=5)

    gross = dp.backtest_dividend_play(prices, divs, **kwargs).iloc[0]
    net = dp.backtest_dividend_play(prices, divs, buy_fee_pct=0.15, sell_fee_pct=0.25,
                                    dividend_tax_pct=10, **kwargs).iloc[0]

    assert net['price_return'] < gross['price_return']
    assert np.isclose(net['dividend_return'], 4.5 / (gross['entry_price'] * 1.0015) * 100)


def _search_fixture():
    ex_dates = ['2019-06-14', '2020-06-15', '2021-06-15', '2022-06-15', '2023-06-15', '2024-06-14']
    data = dp.PlayData(
        _prices(start='2018-06-01', ex_dates=ex_dates),
        pd.DataFrame({'date': ex_dates, 'dividend': [3.0] * len(ex_dates)}),
    )
    setups = dp.build_setup_grid(
        windows=[(90, 60), (30, 10)],
        entries=[(dp.ENTRY_FIXED, 20, 0.0)],
        exits=[(dp.EXIT_CUM, 0)],
        exit_limits=[(0.0, 0.0)],
    )
    return data, setups


def test_evaluate_stock_setups_splits_by_ex_date():
    data, setups = _search_fixture()

    rows = dp.evaluate_stock_setups('AAAA.JK', data, setups, '2022-01-01', {
        'buy_fee_pct': 0.0, 'sell_fee_pct': 0.0, 'dividend_tax_pct': 0.0,
    })

    assert len(setups) == 2
    assert [row['Setup ID'] for row in rows] == [0, 1]
    assert rows[0]['Train Trades'] == 3
    assert rows[0]['Validation Trades'] == 3
    # Buying at the 90-day low captures more of the run-up than buying 30 days out.
    assert rows[0]['Validation Avg Trade'] > rows[1]['Validation Avg Trade'] > 0


def test_summarize_and_best_stock_setups_rank_the_stronger_window_first():
    data, setups = _search_fixture()
    costs = {'buy_fee_pct': 0.0, 'sell_fee_pct': 0.0, 'dividend_tax_pct': 0.0}
    results = pd.DataFrame(
        dp.evaluate_stock_setups('AAAA.JK', data, setups, '2022-01-01', costs)
        + dp.evaluate_stock_setups('BBBB.JK', data, setups, '2022-01-01', costs)
    )

    summary = dp.summarize_setups(results, setups, min_profit_factor=1.2, min_t_stat=0.0)
    best = dp.best_stock_setups(results, setups, min_validation_trades=3, min_win_rate=60)

    assert summary.iloc[0]['Setup ID'] == 0
    assert summary.iloc[0]['Passes Gates']
    assert summary.iloc[0]['Setup'].startswith('Buy 90-60d before ex')
    assert set(best['Symbol']) == {'AAAA.JK', 'BBBB.JK'}
    assert (best['entry_start_days'] == 90).all()
