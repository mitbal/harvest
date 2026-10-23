import numpy as np
import pandas as pd
import pytest

import harvest.swing as sw


def _closes(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# ── Regime ──────────────────────────────────────────────────────────────── #

def test_compute_regime_series_labels_crossovers():
    index_df = pd.DataFrame({"close": np.r_[np.linspace(200, 100, 60), np.linspace(100, 300, 60)]},
                            index=pd.bdate_range("2024-01-01", periods=120))

    regime = sw.compute_regime_series(index_df, fast_sma=5, slow_sma=20)

    assert regime.iloc[30] == "bear"
    assert regime.iloc[-1] == "bull"
    assert sw.get_current_regime(regime) == "bull"
    assert sw.get_regime_for_date(regime, regime.index[30]) == "bear"
    assert sw.get_regime_for_date(regime, pd.Timestamp("2000-01-01")) is None


def test_compute_regime_series_needs_enough_history():
    index_df = pd.DataFrame({"close": [1.0] * 10}, index=pd.bdate_range("2024-01-01", periods=10))

    assert sw.compute_regime_series(index_df, 5, 20).empty
    assert sw.get_current_regime(pd.Series(dtype=str)) is None


# ── Indicators ──────────────────────────────────────────────────────────── #

def test_wilder_rsi_bounds_and_extremes():
    rising = sw.wilder_rsi(_closes(np.arange(1.0, 40.0)), 14)
    falling = sw.wilder_rsi(_closes(np.arange(40.0, 1.0, -1)), 14)

    assert rising.iloc[:14].isna().all()
    assert rising.iloc[-1] == pytest.approx(100.0)
    assert falling.iloc[-1] == pytest.approx(0.0)


def test_close_supertrend_flips_with_the_trend():
    close = _closes(np.r_[np.linspace(100, 150, 40), np.linspace(150, 90, 40)])

    line, direction = sw.close_supertrend(close, period=5, multiplier=2.0)

    assert direction.iloc[39] == 1
    assert line.iloc[39] < close.iloc[39]
    assert direction.iloc[-1] == -1
    assert line.iloc[-1] > close.iloc[-1]


def test_dividend_adjust_scales_history_before_ex_date():
    close = _closes([100.0, 100.0, 95.0, 96.0])
    dividends = pd.DataFrame({"date": [close.index[2]], "dividend": [5.0]})

    adjusted = sw.dividend_adjust(close, dividends)

    assert adjusted.iloc[:2].tolist() == pytest.approx([95.0, 95.0])
    assert adjusted.iloc[2:].tolist() == [95.0, 96.0]
    assert sw.dividend_adjust(close, None) is close


def test_dividend_adjust_ignores_implausible_events():
    close = _closes([100.0, 40.0])
    dividends = pd.DataFrame({"date": [close.index[1]], "dividend": [60.0]})

    assert sw.dividend_adjust(close, dividends).tolist() == [100.0, 40.0]


# ── Costs ───────────────────────────────────────────────────────────────── #

@pytest.mark.parametrize("price, tick", [
    (50, 1), (199, 1), (200, 2), (499, 2), (500, 5), (1999, 5), (2000, 10), (4999, 10), (5000, 25), (9000, 25),
])
def test_idx_tick_size_follows_fraction_table(price, tick):
    assert sw.idx_tick_size([price])[0] == tick


def test_build_trade_fees_charges_entries_and_exits():
    prices = _closes([1000.0, 1010.0, 1020.0])
    entries = pd.Series([True, False, False], index=prices.index)
    exits = pd.Series([False, False, True], index=prices.index)

    fees = sw.build_trade_fees(prices, entries, exits, 0.15, 0.15, 0.1, slippage_ticks=1.0)

    assert fees.iloc[0] == pytest.approx(0.0015 + 5 / 1000)
    assert fees.iloc[1] == 0.0
    assert fees.iloc[2] == pytest.approx(0.0025 + 5 / 1020)


def test_market_breadth_needs_minimum_universe():
    grouped = {f"S{i}": _closes(np.linspace(100, 120, 30)) for i in range(3)}

    assert sw.build_market_breadth(grouped, window=5, min_stocks=4).empty
    breadth = sw.build_market_breadth(grouped, window=5, min_stocks=3)
    assert breadth.iloc[-1] == 100.0


# ── Ranking and selection ───────────────────────────────────────────────── #

def test_format_strategy_params_trims_zeros():
    text = sw.format_strategy_params(sw.STRATEGY_BOLLINGER_LOWER, 20, 2.0, 14, 40.5)

    assert text == "Band Period: 20, Std Dev: 2, RSI Period: 14, Max RSI: 40.5"


def _result(symbol, strategy=sw.STRATEGY_REGIME_RSI, param_a=14.0, **overrides):
    row = {
        "Symbol": symbol, "Strategy": strategy,
        "Param A": param_a, "Param B": 30.0, "Param C": 50.0, "Param D": 45.0,
        "Max Hold": 5, "Profit Target": 5.0, "Stop Loss": 2.0,
        "Avg Trade": 0.5, "# Trades": 20,
        "Validation Return": 4.0, "Validation Win Rate": 60.0, "Validation Drawdown": -3.0,
        "Validation Trades": 12, "Validation Avg Trade": 0.4,
        "Validation Gross Win": 9.0, "Validation Gross Loss": 4.2, "Validation Sum Sq": 20.0,
    }
    row.update(overrides)
    return row


def test_summarize_strategies_pools_across_stocks_and_ranks_gated_first():
    results = pd.DataFrame([
        _result("A.JK"), _result("B.JK"),
        _result("A.JK", param_a=7.0, **{"Validation Avg Trade": -0.2, "Validation Gross Win": 1.0}),
        _result("B.JK", param_a=7.0, **{"Validation Avg Trade": -0.2, "Validation Gross Win": 1.0}),
    ])

    summary = sw.summarize_strategies(results)

    assert len(summary) == 2
    best = summary.iloc[0]
    assert best["Param A"] == 14.0
    assert best["Stocks Validated"] == 2
    assert best["Validation Trades"] == 24
    assert best["Validation Profit Factor"] == pytest.approx(18.0 / 8.4)
    assert best["Parameters"].startswith("RSI Period: 14")
    assert not summary.iloc[1]["Passes Gates"]


def test_select_stock_strategies_only_overrides_when_holdout_beats_global():
    global_strategy = {
        "strategy_type": sw.STRATEGY_REGIME_RSI, "param_a": 14.0, "param_b": 30.0, "param_c": 50.0,
        "param_d": 45.0, "max_hold": 5, "profit_target": 5.0, "stop_loss": 2.0,
    }
    results = pd.DataFrame([
        _result("A.JK"),
        _result("A.JK", param_a=7.0, **{"Validation Return": 9.0}),
        _result("B.JK"),
        _result("B.JK", param_a=7.0, **{"Validation Return": 2.0}),
    ])

    selections = sw.select_stock_strategies(results, global_strategy)

    assert set(selections) == {"A.JK"}
    assert selections["A.JK"]["param_a"] == 7.0
    assert selections["A.JK"]["return_uplift"] == pytest.approx(5.0)


def test_scanner_plan_prefers_stock_then_global_then_baseline():
    stock = {
        "strategy_type": sw.STRATEGY_REGIME_RSI, "param_a": 7, "param_b": 25, "param_c": 20, "param_d": 40,
        "max_hold": 3, "profit_target": 4.0, "stop_loss": 1.5, "score": 80.0,
        "return_uplift": 2.0, "validation_avg_trade": 0.6,
    }
    active = {**stock, "strategy_type": sw.STRATEGY_BOLLINGER_LOWER, "score": 70.0}

    plan = {item["strategy_type"]: item for item in sw.scanner_strategy_plan(stock, active)}

    assert list(plan) == sw.STRATEGY_TYPES
    assert plan[sw.STRATEGY_REGIME_RSI]["parameter_source"] == "Stock-specific optimized"
    assert plan[sw.STRATEGY_BOLLINGER_LOWER]["parameter_source"] == "Global optimized"
    baseline = plan[sw.STRATEGY_RSI_RECOVERY]
    assert baseline["parameter_source"] == "Baseline preset"
    assert not baseline["validated"]
    assert baseline["params"] == tuple(float(v) for v in sw.SEARCH_PRESETS["Quick"]["strategy_params"][sw.STRATEGY_RSI_RECOVERY][0])


# ── Evaluation and scanner helpers ──────────────────────────────────────── #

def test_evaluate_stock_strategies_splits_train_and_holdout():
    close = _closes(100 + np.sin(np.arange(400) / 3) * 8)
    regime = pd.Series("bull", index=close.index)
    combos = [(sw.STRATEGY_REGIME_RSI, 7, 40, 5, 45, 5, 5.0, 2.0)]

    split = sw.holdout_split_date(close.index[0], close.index[-1], 30)
    trades = []
    rows, status = sw.evaluate_stock_strategies(
        "TEST.JK", close, regime, combos, split, 0.15, 0.15, 0.1, None, {}, slippage_ticks=0.5, trades_out=trades,
    )

    assert status == "ok"
    assert rows[0]["Symbol"] == "TEST.JK"
    assert rows[0]["# Trades"] > 0
    assert rows[0]["Validation Trades"] > 0
    assert len(trades) == rows[0]["# Trades"] + rows[0]["Validation Trades"]
    assert {t["combo"] for t in trades} == {0}
    assert all(t["exit_date"] >= t["entry_date"] and t["entry_price"] > 0 for t in trades)


@pytest.mark.parametrize("series, status", [(None, "missing_data"), (_closes([100.0] * 30), "insufficient_history")])
def test_evaluate_stock_strategies_reports_unusable_data(series, status):
    combos = [(sw.STRATEGY_REGIME_RSI, 14, 30, 50, 45, 5, 5.0, 2.0)]

    assert sw.evaluate_stock_strategies(
        "TEST.JK", series, pd.Series(dtype=str), combos, "2024-01-20", 0.15, 0.15, 0.1, None, {},
    ) == ([], status)


def test_clean_price_series_sorts_and_dedupes():
    series = pd.Series([3.0, "bad", 1.0, 2.0], index=["2024-01-03", "2024-01-02", "2024-01-01", "2024-01-03"])

    clean = sw.clean_price_series(series)

    assert clean.index.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-01-03"]
    assert clean.tolist() == [1.0, 2.0]
    assert sw.clean_price_series(None).empty


def test_regime_rsi_watch_uses_bear_threshold_outside_bull_regime():
    params = (14.0, 30.0, 50.0, 45.0)

    assert sw.is_near_setup(sw.STRATEGY_REGIME_RSI, 33.0, params, 5.0, trend_ok=False, regime="bear")
    assert not sw.is_near_setup(sw.STRATEGY_REGIME_RSI, 40.0, params, 5.0, regime="bear")
    assert not sw.is_near_setup(sw.STRATEGY_REGIME_RSI, np.nan, params, 5.0, regime="bull")


# ── OHLC fills ──────────────────────────────────────────────────────────── #

def _bars(rows, start="2024-01-01"):
    """OHLC frame from (open, high, low, close) tuples on business days."""
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=pd.bdate_range(start, periods=len(rows)))


def _simulate(rows, setup_on=0, exit_on=None, max_hold=5, target=5.0, stop=2.0):
    frame = _bars(rows)
    setup = pd.Series(False, index=frame.index)
    setup.iloc[setup_on] = True
    strategy_exit = pd.Series(False, index=frame.index)
    if exit_on is not None:
        strategy_exit.iloc[exit_on] = True
    return sw.simulate_trades(frame, setup, strategy_exit, max_hold, target, stop, has_ohlc=True)


def test_entry_is_the_next_open():
    trades = _simulate([(100, 101, 99, 100), (102, 103, 101.5, 102), (102, 103, 101.5, 102)], max_hold=1)

    assert trades[0]["entry_i"] == 1 and trades[0]["entry_price"] == 102
    assert trades[0]["exit_reason"] == "Max hold" and trades[0]["exit_i"] == 2


def test_stop_fills_at_the_stop_or_the_gap_open():
    intrabar = _simulate([(100, 100, 100, 100), (100, 101, 99.5, 100), (100, 100.5, 97.0, 99)])
    gap = _simulate([(100, 100, 100, 100), (100, 101, 99.5, 100), (95, 96, 94, 95)])

    assert intrabar[0]["exit_reason"] == "Stop" and intrabar[0]["exit_price"] == pytest.approx(98.0)
    assert gap[0]["exit_reason"] == "Stop (gap)" and gap[0]["exit_price"] == 95


def test_target_fills_at_the_target_or_the_gap_open():
    intrabar = _simulate([(100, 100, 100, 100), (100, 101, 99.5, 100), (101, 106, 100.5, 104)])
    gap = _simulate([(100, 100, 100, 100), (100, 101, 99.5, 100), (108, 109, 107, 108)])

    assert intrabar[0]["exit_reason"] == "Target" and intrabar[0]["exit_price"] == pytest.approx(105.0)
    assert gap[0]["exit_reason"] == "Target (gap)" and gap[0]["exit_price"] == 108


def test_stop_wins_when_both_levels_are_touched():
    trades = _simulate([(100, 100, 100, 100), (100, 101, 99.5, 100), (100, 106, 97, 103)])

    assert trades[0]["exit_reason"] == "Stop"


def test_entry_day_checks_only_the_stop():
    stopped = _simulate([(100, 100, 100, 100), (100, 101, 97.5, 99)])
    no_target = _simulate([(100, 100, 100, 100), (100, 106, 99.5, 105), (104, 104.5, 103, 104)], max_hold=1)

    assert stopped[0]["entry_i"] == stopped[0]["exit_i"] == 1
    assert no_target[0]["exit_reason"] == "Max hold"


def test_strategy_exit_fills_at_the_next_open():
    trades = _simulate([(100, 100, 100, 100), (100, 101, 99.5, 100), (100, 101, 99.5, 100.5), (101, 102, 100, 101)],
                       exit_on=2)

    assert trades[0]["exit_reason"] == "Strategy exit"
    assert trades[0]["exit_i"] == 3 and trades[0]["exit_price"] == 101


def test_entry_on_the_last_bar_is_left_out():
    assert _simulate([(100, 100, 100, 100), (100, 101, 99.5, 100)], setup_on=1) == []


def test_ohlc_frame_falls_back_to_closes_and_clamps_bad_bars():
    frame, has_ohlc = sw.ohlc_frame(pd.Series([1.0, 2.0], index=pd.bdate_range("2024-01-01", periods=2)))
    assert not has_ohlc and frame["open"].tolist() == [1.0, 2.0]

    bad = _bars([(100, 99, 101, 100)])
    clean, has_ohlc = sw.ohlc_frame(bad)
    assert has_ohlc and clean.iloc[0]["high"] == 101 and clean.iloc[0]["low"] == 99


def test_backtest_costs_follow_the_vectorbt_trade_return():
    rows = [(100, 100, 100, 100)] * 60 + [(100, 101, 99.5, 100), (101, 106, 100.5, 104)]
    frame = _bars(rows)
    regime = pd.Series("bull", index=frame.index)
    with patch_components(frame, setup_at=59):
        trades, _ = sw.backtest_trades(frame, regime, sw.STRATEGY_REGIME_RSI, 14, 30, 50, 45, 5, 5.0, 2.0,
                                       buy_fee=0.15, sell_fee=0.15, sell_tax=0.1)

    trade = trades.iloc[0]
    assert trade["exit_price"] == pytest.approx(105.0)
    assert trade["return_pct"] == pytest.approx((105 * (1 - 0.0025) / 100 - 1.0015) * 100)


from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def patch_components(frame, setup_at):
    setup = pd.Series(False, index=frame.index)
    setup.iloc[setup_at] = True
    blank = pd.Series(False, index=frame.index)
    with patch.object(sw, "strategy_components", return_value=(setup, blank, pd.Series(0.0, index=frame.index))):
        yield


# ── Walk-forward and gates ──────────────────────────────────────────────── #

def _trades(combo, start, end, mean, seed, per_month=8):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq=f"{30 // per_month}D")
    return pd.DataFrame({"combo": combo, "symbol": "X", "entry_date": dates,
                         "return_pct": mean + rng.normal(0, 1.0, len(dates))})


def test_walk_forward_records_the_choice_made_on_past_data():
    # Combo 0 shines until 2022 and then stops working; combo 1 is steadily positive.
    trades = pd.concat([
        _trades(0, "2016-01-01", "2021-12-31", 2.0, 1), _trades(0, "2022-01-01", "2025-12-31", -1.0, 2),
        _trades(1, "2016-01-01", "2025-12-31", 0.6, 3),
    ])

    result = sw.walk_forward(trades, "2022-01-01", "2025-12-31")

    picks = [fold["combo"] for fold in result["folds"]]
    assert picks[0] == 0          # chosen on 2019-2021, when combo 0 looked best
    assert picks[-1] == 1         # the walk-forward notices and switches
    assert result["final_combo"] == 1
    # The record includes the losing fold(s) of combo 0: below combo 1's own mean.
    assert result["record"]["avg_trade"] < 0.6
    assert result["record"]["trades"] == sum(fold["trades"] for fold in result["folds"])
    assert len(result["trades"]) == result["record"]["trades"]
    assert result["trades"]["entry_date"].min() >= pd.Timestamp("2022-01-01")


def test_walk_forward_needs_enough_training_trades():
    trades = _trades(0, "2024-01-01", "2025-12-31", 1.0, 4, per_month=1)

    result = sw.walk_forward(trades, "2025-01-01", "2025-12-31")

    assert result["final_combo"] is None
    assert result["record"]["trades"] == 0


def test_walk_forward_gates():
    good = {"trades": 40, "avg_trade": 0.5, "profit_factor": 1.5, "t_stat": 2.5}
    assert sw.passes_walk_forward_gates(good)
    for key, value in [("trades", 20), ("avg_trade", -0.1), ("profit_factor", 1.1), ("t_stat", 1.5)]:
        assert not sw.passes_walk_forward_gates({**good, key: value})


def test_required_t_stat_grows_with_the_number_of_tries():
    assert sw.required_t_stat(1) == sw.MIN_T_STAT
    assert sw.required_t_stat(22) == pytest.approx(2.84, abs=0.01)
    assert sw.required_t_stat(200) > sw.required_t_stat(22)


def test_select_stock_strategies_applies_the_t_stat_bar():
    global_strategy = {
        "strategy_type": sw.STRATEGY_REGIME_RSI, "param_a": 14.0, "param_b": 30.0, "param_c": 50.0,
        "param_d": 45.0, "max_hold": 5, "profit_target": 5.0, "stop_loss": 2.0,
    }
    results = pd.DataFrame([
        _result("A.JK"),
        _result("A.JK", param_a=7.0, **{"Validation Return": 9.0}),
    ])
    t = sw.stock_holdout_t_stat(results.iloc[[1]]).iloc[0]

    assert "A.JK" in sw.select_stock_strategies(results, global_strategy, min_t_stat=t - 0.01)
    assert sw.select_stock_strategies(results, global_strategy, min_t_stat=t + 0.01) == {}


# ── IDX price rules and sizing ──────────────────────────────────────────── #

@pytest.mark.parametrize("price, direction, expected", [
    (1234.0, "down", 1230.0), (1234.0, "up", 1235.0), (6012.0, "down", 6000.0), (199.4, "nearest", 199.0),
])
def test_round_to_tick(price, direction, expected):
    assert sw.round_to_tick(price, direction) == expected


def test_price_limits_follow_the_2026_and_2027_rules():
    assert sw.price_limits(1000, "2026-10-05") == (850.0, 1250.0)      # ARB 15%, ARA 25%
    assert sw.price_limits(1000, "2027-01-04") == (750.0, 1250.0)      # symmetric from 2027
    assert sw.price_limits(8000, "2026-10-05") == (6800.0, 9600.0)     # ARA 20% above Rp 5,000
    assert sw.price_limits(150, "2027-01-04") == (98.0, 202.0)         # 35% up to Rp 200
    assert sw.price_limits(8, "2026-10-05") == (7.0, 9.0)              # Rp 1 band for Rp 1-10


def test_position_size_by_risk_and_cash():
    sized = sw.position_size(capital=100_000_000, risk_pct=1.0, entry=1000, stop=980)
    # Rp 1,000,000 at risk / (Rp 20 x 100 shares) = 500 lots, but the capital buys only 1,000 lots.
    assert sized == {"lots": 500, "risk_idr": 1_000_000.0, "cost_idr": 50_000_000.0}
    capped = sw.position_size(capital=10_000_000, risk_pct=5.0, entry=1000, stop=999)
    assert capped["lots"] == 100  # the cash limit binds
    assert sw.position_size(10_000_000, 1.0, 1000, 1000)["lots"] == 0


# ── Shortlist and helpers shared with the pipeline ──────────────────────── #

def test_price_volatility_ignores_implausible_moves():
    dates = pd.bdate_range("2024-01-01", periods=61)
    calm = 100 * np.cumprod(np.r_[1, np.full(60, 1.01)])
    jumpy = calm.copy()
    jumpy[30:] *= 2  # a 100% jump is a corporate action, not volatility
    prices = pd.concat([
        pd.DataFrame({"symbol": "CALM", "date": dates, "close": calm}),
        pd.DataFrame({"symbol": "JUMP", "date": dates, "close": jumpy}),
    ])

    vol = sw.price_volatility(prices).set_index("stock")

    assert vol.loc["CALM", "Daily Volatility"] == pytest.approx(0.0, abs=1e-9)
    assert vol.loc["JUMP", "Volatility Observations"] == 59


def test_build_shortlist_needs_liquidity_and_movement():
    universe = pd.DataFrame({
        "stock": ["BBRI.JK", "WILD.JK", "THIN.JK", "TINY.JK", "CALM.JK"],
        "mktCap": [500e12, 2e12, 2e12, 0.5e12, 2e12],
        "price": [4000.0, 500.0, 500.0, 500.0, 500.0],
        "volAvg": [1e8, 1e8, 1e3, 1e8, 1e8],
    })
    volatility = pd.DataFrame({
        "stock": universe["stock"], "Daily Volatility": [1.5, 4.0, 4.0, 4.0, 0.5],
        "Volatility Observations": 60,
    })

    shortlist = sw.build_shortlist(universe, volatility)

    assert shortlist["stock"].tolist() == ["BBRI.JK", "WILD.JK"]  # blue chip pinned first
    assert sw.build_shortlist(universe, volatility, pin_blue_chips=False)["stock"].tolist() == ["WILD.JK", "BBRI.JK"]


def test_dividend_adjust_ohlc_scales_every_price():
    frame = _bars([(100, 101, 99, 100), (100, 101, 99, 100), (95, 96, 94, 95)])
    dividends = pd.DataFrame({"date": [frame.index[2]], "dividend": [5.0]})

    adjusted = sw.dividend_adjust_ohlc(frame, dividends)

    assert adjusted.iloc[0].tolist() == pytest.approx([95.0, 95.95, 94.05, 95.0])
    assert adjusted.iloc[2].tolist() == [95, 96, 94, 95]


def test_combo_setup_and_presets():
    combos = sw.preset_combos("Quick")
    setup = sw.combo_setup(combos[0])

    assert len(combos) == len(sw.STRATEGY_TYPES)
    assert list(setup) == list(sw.SETUP_KEYS)
    assert setup["strategy_type"] == combos[0][0] and isinstance(setup["max_hold"], int)


# ── Limit-entry strategies ──────────────────────────────────────────────── #

def _limit_simulate(rows, limit, max_hold=2, stop=0.0, target=0.0):
    frame = _bars(rows)
    setup = pd.Series(False, index=frame.index)
    setup.iloc[0] = True
    limits = pd.Series(np.nan, index=frame.index)
    limits.iloc[0] = limit
    return sw.simulate_trades(frame, setup, pd.Series(False, index=frame.index), max_hold, target, stop,
                              has_ohlc=True, entry_limit=limits, exit_at_open=True)


def test_limit_entry_fills_only_below_the_limit_and_sells_at_the_open():
    rows = [(100, 100, 100, 100), (99, 99.5, 94, 96), (97, 99, 96, 98), (101, 102, 100, 101)]

    [trade] = _limit_simulate(rows, limit=95.0)
    gap_down = _limit_simulate([(100, 100, 100, 100), (93, 96, 92, 95)] + rows[2:], limit=95.0)

    assert trade["entry_i"] == 1 and trade["entry_price"] == 95.0
    # Held two sessions after the entry day's close: sold at the open of bar 3, not its close.
    assert trade["exit_i"] == 3 and trade["exit_price"] == 101 and trade["exit_reason"] == "Max hold"
    assert gap_down[0]["entry_price"] == 93  # an open below the limit fills at the open
    # Touching the limit is not enough, and an unfilled order is not carried to later sessions.
    assert _limit_simulate([(100, 100, 100, 100), (99, 99.5, 95, 96), (94, 95, 90, 92), (97, 98, 96, 97)],
                           limit=95.0) == []


def test_per_setup_stop_percentages_are_read_at_the_setup_close():
    frame = _bars([(100, 100, 100, 100), (100, 100.5, 96.5, 98), (98, 99, 97, 98)])
    setup = pd.Series([True, False, False], index=frame.index)
    stops = pd.Series([3.0, 50.0, 50.0], index=frame.index)

    [trade] = sw.simulate_trades(frame, setup, pd.Series(False, index=frame.index), 5, 0.0, stops, has_ohlc=True)

    assert trade["exit_reason"] == "Stop" and trade["exit_price"] == pytest.approx(97.0)


def _ohlcv(close, volume=None, start="2024-01-01"):
    close = np.asarray(close, dtype=float)
    volume = np.full(len(close), 1e8) if volume is None else np.asarray(volume, dtype=float)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": volume},
                        index=pd.bdate_range(start, periods=len(close)))


def _components(bars, strategy, params):
    regime = pd.Series("bear", index=bars.index)
    return sw.strategy_components(bars["close"], regime, None, None, strategy, *params, bars=bars)


def test_panic_dip_needs_a_sharp_drop_in_a_liquid_stock():
    close = np.r_[np.full(40, 1000.0), 950.0, 940.0]  # -5% then -1%
    setup, _, drop = _components(_ohlcv(close), sw.STRATEGY_PANIC_DIP, (1, 4, 1.0, 14))
    thin, _, _ = _components(_ohlcv(close, volume=np.full(len(close), 1e5)), sw.STRATEGY_PANIC_DIP, (1, 4, 1.0, 14))
    penny, _, _ = _components(_ohlcv(close / 10), sw.STRATEGY_PANIC_DIP, (1, 4, 1.0, 14))

    assert setup.tolist()[-2:] == [True, False] and drop.iloc[-2] == pytest.approx(-5.0)
    assert not thin.any()   # Rp 100 million a day is too thin to buy a panic
    assert not penny.any()  # under Rp 200 a tick is 1% or more


def test_volume_breakout_needs_a_new_high_on_heavy_volume():
    close = np.r_[np.linspace(900, 1000, 70), 1010.0, 1020.0]
    volume = np.r_[np.full(70, 1e7), 3e7, 1e7]
    setup, _, _ = _components(_ohlcv(close, volume), sw.STRATEGY_VOLUME_BREAKOUT, (60, 2, 1.0, 14))
    no_volume, _, _ = _components(_ohlcv(close, volume).drop(columns="volume"), sw.STRATEGY_VOLUME_BREAKOUT,
                                  (60, 2, 1.0, 14))

    assert setup.tolist()[-2:] == [True, False]
    assert not no_volume.any()


def test_limit_execution_is_a_tick_rounded_atr_below_the_close():
    bars = _ohlcv(np.full(30, 1000.0))  # true range 2% of the close every day
    limit, atr = sw.limit_execution(bars, sw.STRATEGY_PANIC_DIP, 1.5, 14)

    assert atr.iloc[-1] == pytest.approx(2.0)
    assert limit.iloc[-1] == 970.0
    assert sw.limit_execution(bars, sw.STRATEGY_REGIME_RSI, 1.5, 14) is None


def test_backtest_scales_limit_exits_by_atr():
    close = np.r_[np.full(40, 1000.0), 940.0]
    bars = _ohlcv(close)
    # The next session flushes through the limit and keeps falling to 3 ATRs below the fill.
    bars.loc[bars.index[-1] + pd.offsets.BDay(1)] = [930.0, 935.0, 840.0, 860.0, 1e8]
    bars.loc[bars.index[-1] + pd.offsets.BDay(1)] = [880.0, 890.0, 870.0, 880.0, 1e8]
    regime = pd.Series("bear", index=bars.index)

    trades, _ = sw.backtest_trades(bars, regime, sw.STRATEGY_PANIC_DIP, 1, 4, 1.0, 14, 2, 0.0, 3.0, 0.15, 0.15, 0.1)
    no_stop, _ = sw.backtest_trades(bars, regime, sw.STRATEGY_PANIC_DIP, 1, 4, 1.0, 14, 2, 0.0, 0.0, 0.15, 0.15, 0.1)

    assert trades.iloc[0]["exit_reason"] == "Stop" and trades.iloc[0]["sessions"] == 0
    assert no_stop.iloc[0]["exit_reason"] == "End of data"


def test_limit_strategies_search_their_own_exits():
    combos = sw.preset_combos("Balanced", [sw.STRATEGY_PANIC_DIP])
    exits = sw.strategy_exit_space(sw.STRATEGY_PANIC_DIP)

    assert {c[5] for c in combos} == set(exits["hold_days"])
    assert {(c[6], c[7]) for c in combos} == set(exits["exit_rules"])
    assert sw.strategy_exit_space(sw.STRATEGY_REGIME_RSI)["hold_days"] == sw.SEARCH_PRESETS["Balanced"]["hold_days"]
    assert sw.format_exit_rules(sw.STRATEGY_PANIC_DIP, 2, 0.0, 3.0) == "hold 2 (sell at the open), no target, stop 3 ATR"
    assert sw.format_exit_rules(sw.STRATEGY_REGIME_RSI, 5, 5.0, 2.0) == "hold 5, target 5%, stop 2%"


def test_walk_forward_playbook_keeps_every_family_that_passes():
    combos = [("A", 1, 0, 0, 0, 2, 0, 0), ("B", 1, 0, 0, 0, 2, 0, 0), ("C", 1, 0, 0, 0, 2, 0, 0)]
    trades = pd.concat([
        _trades(0, "2016-01-01", "2025-12-31", 0.8, 11), _trades(1, "2016-01-01", "2025-12-31", 1.2, 12),
        _trades(2, "2016-01-01", "2025-12-31", -0.5, 13),
    ])

    families = sw.walk_forward_playbook(trades, combos, "2022-01-01", "2025-12-31")

    assert [f["strategy_type"] for f in families] == ["B", "A"]  # best average trade first; C loses
    assert [sw.playbook_scope(rank) for rank in range(2)] == ["global", "global:2"]
    published = {"global:2": {"n": 2}, "BBRI.JK": {}, "global": {"n": 1}}
    assert [scope for scope, _ in sw.playbook(published)] == ["global", "global:2"]
