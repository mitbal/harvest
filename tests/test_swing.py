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

    rows, status = sw.evaluate_stock_strategies(
        "TEST.JK", close, regime, combos, 30, 0.15, 0.15, 0.1, None, {}, slippage_ticks=0.5,
    )

    assert status == "ok"
    assert rows[0]["Symbol"] == "TEST.JK"
    assert rows[0]["# Trades"] > 0
    assert rows[0]["Validation Trades"] > 0


@pytest.mark.parametrize("series, status", [(None, "missing_data"), (_closes([100.0] * 30), "insufficient_history")])
def test_evaluate_stock_strategies_reports_unusable_data(series, status):
    combos = [(sw.STRATEGY_REGIME_RSI, 14, 30, 50, 45, 5, 5.0, 2.0)]

    assert sw.evaluate_stock_strategies(
        "TEST.JK", series, pd.Series(dtype=str), combos, 30, 0.15, 0.15, 0.1, None, {},
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
