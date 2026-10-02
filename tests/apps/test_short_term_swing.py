import numpy as np
import pandas as pd

from apps.trading import day_trading as swing


def _market(length=260, regime="bull"):
    index = pd.bdate_range("2024-01-01", periods=length)
    return index, pd.Series(regime, index=index, dtype=str)


def test_all_strategy_components_return_aligned_series():
    index, regime = _market()
    trend = np.linspace(100, 150, len(index))
    close = pd.Series(trend + np.sin(np.arange(len(index)) / 4) * 4, index=index)
    benchmark = pd.Series(np.linspace(100, 120, len(index)), index=index)
    percentile = pd.Series(85.0, index=index)

    params = {
        swing.STRATEGY_REGIME_RSI: (14, 30, 50, 45),
        swing.STRATEGY_BOLLINGER_LOWER: (20, 2.0, 14, 40),
        swing.STRATEGY_RSI_RECOVERY: (14, 30, 0, 35),
        swing.STRATEGY_DONCHIAN: (20, 10, 50, 0.0),
        swing.STRATEGY_SQUEEZE: (20, 20, 10, 120),
        swing.STRATEGY_TREND_PULLBACK: (20, 50, 2.0, 5),
        swing.STRATEGY_RELATIVE_STRENGTH: (20, 2.0, 14, 40),
        swing.STRATEGY_CROSS_MOMENTUM: (20, 80, 14, 40),
        swing.STRATEGY_SUPERTREND_PULLBACK: (10, 3.0, 2.0, 0.5),
        swing.STRATEGY_BREADTH_RSI2: (2, 10, 20, 200),
    }
    breadth = pd.Series(10.0, index=index)

    for strategy, strategy_params in params.items():
        setup, exits, indicator = swing._strategy_components(
            close, regime, benchmark, percentile, strategy, *strategy_params,
            breadth=breadth,
        )
        assert setup.index.equals(close.index)
        assert exits.index.equals(close.index)
        assert indicator.index.equals(close.index)
        assert setup.dtype == bool
        assert exits.dtype == bool


def test_completed_setup_enters_on_next_session(monkeypatch):
    index, regime = _market(100, regime="bear")
    close = pd.Series(np.linspace(100, 90, len(index)), index=index)
    setup_date = index[70]

    def fake_components(*args, **kwargs):
        setup = pd.Series(False, index=index)
        setup.loc[setup_date] = True
        return setup, pd.Series(False, index=index), pd.Series(0.0, index=index)

    monkeypatch.setattr(swing, "_strategy_components", fake_components)
    signals = swing._build_regime_signals(
        close, regime, swing.STRATEGY_REGIME_RSI,
        14, 30, 50, 45, 5, 3.0, 2.0,
    )

    assert signals is not None
    _, entries, _, _ = signals
    assert not entries.loc[setup_date]
    assert entries.loc[index[71]]


def test_cross_sectional_momentum_ranks_each_date():
    index, _ = _market(80)
    grouped = {
        "LEADER": pd.Series(np.linspace(100, 180, len(index)), index=index),
        "LAGGARD": pd.Series(np.linspace(100, 105, len(index)), index=index),
    }

    ranks = swing._build_momentum_percentiles(grouped, {20})[20]

    assert ranks.loc[index[-1], "LEADER"] == 100.0
    assert ranks.loc[index[-1], "LAGGARD"] == 50.0


def test_squeeze_warmup_includes_bandwidth_history():
    warmup = swing._strategy_warmup(
        swing.STRATEGY_SQUEEZE, 20, 20, 10, 120,
    )

    assert warmup == 145


def test_supertrend_buys_bullish_pullback_and_exits_bearish_flip():
    index, regime = _market(17)
    close = pd.Series(
        [100, 100, 100, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 118, 116, 90],
        index=index,
        dtype=float,
    )

    setup, exits, support_distance = swing._strategy_components(
        close, regime, None, None, swing.STRATEGY_SUPERTREND_PULLBACK,
        3, 2.0, 1.0, 0.0,
    )

    assert support_distance.loc[index[14]] >= 0
    assert setup.loc[index[14]]
    assert exits.loc[index[-1]]
    assert support_distance.loc[index[-1]] < 0


def test_supertrend_support_proximity_only_triggers_once_while_price_remains_nearby():
    index, regime = _market(16)
    close = pd.Series(
        [100, 100, 100, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 120, 120],
        index=index,
        dtype=float,
    )

    setup, _, support_distance = swing._strategy_components(
        close, regime, None, None, swing.STRATEGY_SUPERTREND_PULLBACK,
        3, 2.0, 10.0, 10.0,
    )

    assert 0 <= support_distance.iloc[-1] <= 10
    assert setup.any()
    assert not setup.iloc[-1]
    assert not (setup & setup.shift(1, fill_value=False)).any()


def test_supertrend_watch_excludes_stale_support_zone_and_caps_width():
    params = (10.0, 3.0, 2.0, 0.5)

    assert not swing._is_near_setup(
        swing.STRATEGY_SUPERTREND_PULLBACK, 0.25, params, 5.0,
    )
    assert swing._is_near_setup(
        swing.STRATEGY_SUPERTREND_PULLBACK, 1.5, params, 5.0,
    )
    assert not swing._is_near_setup(
        swing.STRATEGY_SUPERTREND_PULLBACK, 3.0, params, 5.0,
    )


def test_scanner_plan_defaults_to_unvalidated_baseline_presets():
    plan = swing._scanner_strategy_plan(None, None)

    assert [item["strategy_type"] for item in plan] == swing.STRATEGY_TYPES
    assert all(item["profit_target"] == 5.0 for item in plan)
    assert all(item["parameter_source"] == "Baseline preset" for item in plan)
    assert not any(item["validated"] for item in plan)


def test_scanner_plan_uses_matching_optimized_parameters_only():
    active = {
        "strategy_type": swing.STRATEGY_BOLLINGER_LOWER,
        "param_a": 30,
        "param_b": 2.0,
        "param_c": 14,
        "param_d": 45,
        "max_hold": 7,
        "profit_target": 4.0,
        "stop_loss": 2.0,
        "score": 88.0,
        "validation_avg_trade": 0.8,
    }

    plan = swing._scanner_strategy_plan(None, active)
    bollinger = next(item for item in plan if item["strategy_type"] == swing.STRATEGY_BOLLINGER_LOWER)
    regime_rsi = next(item for item in plan if item["strategy_type"] == swing.STRATEGY_REGIME_RSI)

    assert bollinger["params"] == (30.0, 2.0, 14.0, 45.0)
    assert bollinger["parameter_source"] == "Global optimized"
    # The optimized target is kept rather than overwritten by the default.
    assert bollinger["profit_target"] == 4.0
    assert bollinger["validated"]
    assert bollinger["expected_trade"] == 0.8
    assert regime_rsi["parameter_source"] == "Baseline preset"


def test_dividend_adjustment_removes_ex_date_drop():
    index = pd.bdate_range("2024-01-01", periods=6)
    close = pd.Series([1000, 1000, 1000, 950, 950, 950], index=index, dtype=float)
    dividends = pd.DataFrame({"date": [index[3]], "dividend": [50.0]})

    adjusted = swing._dividend_adjust(close, dividends)

    assert adjusted.iloc[:3].tolist() == [950.0, 950.0, 950.0]
    assert adjusted.iloc[3:].tolist() == [950.0, 950.0, 950.0]
    assert adjusted.pct_change().abs().max() == 0


def test_dividend_adjustment_ignores_events_outside_history():
    index = pd.bdate_range("2024-01-01", periods=4)
    close = pd.Series([100, 101, 102, 103], index=index, dtype=float)
    dividends = pd.DataFrame({
        "date": [pd.Timestamp("2023-06-01"), pd.Timestamp("2025-06-01")],
        "dividend": [5.0, 5.0],
    })

    pd.testing.assert_series_equal(swing._dividend_adjust(close, dividends), close)


def test_slippage_uses_idx_tick_sizes():
    index = pd.RangeIndex(5)
    prices = pd.Series([100, 300, 1000, 3000, 10000], index=index, dtype=float)

    slippage = swing._slippage_fraction(prices, 1.0)

    assert np.allclose(slippage, [1 / 100, 2 / 300, 5 / 1000, 10 / 3000, 25 / 10000])
    assert (swing._slippage_fraction(prices, 0.0) == 0).all()


def test_market_breadth_counts_stocks_above_average():
    index = pd.bdate_range("2024-01-01", periods=30)
    rising = pd.Series(np.linspace(100, 130, len(index)), index=index)
    falling = pd.Series(np.linspace(130, 100, len(index)), index=index)
    grouped = {"UP1": rising, "UP2": rising, "DOWN": falling}

    breadth = swing._build_market_breadth(grouped, window=20, min_stocks=3)

    assert breadth.index[0] == index[19]
    assert np.isclose(breadth.iloc[-1], 200 / 3)
    assert swing._build_market_breadth(grouped, window=20, min_stocks=4).empty


def test_breadth_rsi2_needs_washed_out_market_and_exits_above_five_day_average():
    index, regime = _market(40)
    values = np.linspace(100, 120, len(index))
    values[30:33] = [112, 106, 101]  # sharp three-day dip
    values[33:] = [104, 108, 115, 118, 119, 120, 121]
    close = pd.Series(values, index=index)
    washed_out = pd.Series(10.0, index=index)
    healthy = pd.Series(70.0, index=index)

    setup, exits, indicator = swing._strategy_components(
        close, regime, None, None, swing.STRATEGY_BREADTH_RSI2,
        2, 10, 20, 0, breadth=washed_out,
    )
    assert setup.loc[index[32]]
    assert indicator.loc[index[32]] == 10.0
    assert exits.loc[index[35]]

    blocked, _, _ = swing._strategy_components(
        close, regime, None, None, swing.STRATEGY_BREADTH_RSI2,
        2, 10, 20, 0, breadth=healthy,
    )
    assert not blocked.any()

    no_breadth, _, _ = swing._strategy_components(
        close, regime, None, None, swing.STRATEGY_BREADTH_RSI2, 2, 10, 20, 0,
    )
    assert not no_breadth.any()


def _summary_row(**overrides):
    row = {
        "Validation Avg Trade": 0.8, "Train Avg Trade": 0.5,
        "Validation Profit Factor": 1.5, "Validation t-stat": 2.5,
    }
    row.update(overrides)
    return row


def test_profitability_gates_reject_losers_and_weak_evidence():
    summary = pd.DataFrame([
        _summary_row(),
        _summary_row(**{"Validation Avg Trade": -0.1}),
        _summary_row(**{"Train Avg Trade": -0.1}),
        _summary_row(**{"Validation Profit Factor": 1.1}),
        _summary_row(**{"Validation t-stat": 1.0}),
        _summary_row(**{"Validation t-stat": np.nan}),
    ])

    assert swing._passes_profitability_gates(summary).tolist() == [
        True, False, False, False, False, False,
    ]


def test_pooled_trade_stats_match_direct_calculation():
    trades_a = np.array([2.0, -1.0, 3.0])
    trades_b = np.array([1.0, -2.0])
    frame = pd.DataFrame([
        {
            "Validation Trades": len(t),
            "Validation Gross Win": t[t > 0].sum(),
            "Validation Gross Loss": -t[t < 0].sum(),
            "Validation Sum Sq": (t ** 2).sum(),
        }
        for t in (trades_a, trades_b)
    ])
    pooled = np.concatenate([trades_a, trades_b])

    profit_factor, t_stat = swing._pooled_trade_stats(frame, "Validation ")

    assert np.isclose(profit_factor, 6.0 / 3.0)
    expected_t = pooled.mean() / (pooled.std(ddof=1) / np.sqrt(len(pooled)))
    assert np.isclose(t_stat, expected_t)


def test_stock_overrides_require_enough_trades_and_positive_train_edge():
    base = {
        "Symbol": "AAAA.JK", "Validation Return": 5.0,
        "Validation Avg Trade": 0.5, "Avg Trade": 0.4,
        "Validation Trades": swing.MIN_STOCK_VALIDATION_TRADES,
    }
    group = pd.DataFrame([
        base,
        {**base, "Validation Trades": swing.MIN_STOCK_VALIDATION_TRADES - 1},
        {**base, "Avg Trade": -0.2},
        {**base, "Validation Avg Trade": -0.1},
    ])

    eligible = swing._eligible_stock_candidates(group)

    assert eligible.index.tolist() == [0]


def test_regime_rsi_watch_respects_bull_trend_filter(monkeypatch):
    monkeypatch.setattr(swing, "curr_regime", "bull")
    params = (14.0, 30.0, 50.0, 45.0)

    assert swing._is_near_setup(swing.STRATEGY_REGIME_RSI, 47.0, params, 5.0, trend_ok=True)
    assert not swing._is_near_setup(swing.STRATEGY_REGIME_RSI, 47.0, params, 5.0, trend_ok=False)


def test_run_regime_backtest_returns_pooled_trade_sums():
    index, regime = _market(120)
    close = pd.Series(100 + np.sin(np.arange(len(index)) / 3) * 8, index=index)

    result = swing._run_regime_backtest(
        close, regime, swing.STRATEGY_REGIME_RSI, 7, 40, 5, 45,
        max_hold=5, profit_target_pct=5.0, stop_loss_pct=2.0,
        buy_fee=0.15, sell_fee=0.15, sell_tax=0.1, slippage_ticks=1.0,
    )

    assert result is not None
    assert result["trades"] > 0
    net = (result["gross_win"] - result["gross_loss"]) / result["trades"]
    assert np.isclose(net, result["avg_trade"])
    assert result["sum_sq"] > 0
