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
    }

    for strategy, strategy_params in params.items():
        setup, exits, indicator = swing._strategy_components(
            close, regime, benchmark, percentile, strategy, *strategy_params,
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


def test_scanner_plan_evaluates_every_strategy_at_five_percent_target():
    plan = swing._scanner_strategy_plan(None, None)

    assert [item["strategy_type"] for item in plan] == swing.STRATEGY_TYPES
    assert all(item["profit_target"] == 5.0 for item in plan)
    assert all(item["parameter_source"] == "Baseline preset" for item in plan)


def test_scanner_plan_uses_matching_optimized_parameters_only():
    active = {
        "strategy_type": swing.STRATEGY_DONCHIAN,
        "param_a": 50,
        "param_b": 20,
        "param_c": 100,
        "param_d": 0.5,
        "max_hold": 7,
        "profit_target": 5.0,
        "stop_loss": 2.0,
        "score": 88.0,
    }

    plan = swing._scanner_strategy_plan(None, active)
    donchian = next(item for item in plan if item["strategy_type"] == swing.STRATEGY_DONCHIAN)
    regime_rsi = next(item for item in plan if item["strategy_type"] == swing.STRATEGY_REGIME_RSI)

    assert donchian["params"] == (50.0, 20.0, 100.0, 0.5)
    assert donchian["parameter_source"] == "Global optimized"
    assert regime_rsi["parameter_source"] == "Baseline preset"
