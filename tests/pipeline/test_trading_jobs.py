import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import harvest.dividend_play as dp
import harvest.trading_store as store
from pipeline import trading_data, trading_optimize, trading_scan
from tests.fake_supabase import FakeSupabase


AS_OF = pd.Timestamp("2026-10-05")  # a Monday
FIXED_SETUP = {**dp.BASELINE_SETUP, "entry_rule": dp.ENTRY_FIXED}
DIP_SETUP = {**dp.BASELINE_SETUP, "dip_ma_days": 20, "dip_pct": 0.0}
HOLDOUT = {"level": "global", "trades": 40, "win_rate": 75.0, "avg_trade": 4.2}


def _prices(closes, end=AS_OF):
    dates = pd.bdate_range(end=end, periods=len(closes))
    return pd.DataFrame({"date": dates, "close": np.asarray(closes, dtype=float)})


def _divs(*dates, amount=100.0):
    return pd.DataFrame({"date": pd.to_datetime(list(dates)), "dividend": amount})


def _signal(prices, divs, setup, **kwargs):
    return trading_scan.dividend_play_signal(
        "TEST.JK", prices, divs, setup, "global", HOLDOUT,
        {"companyName": "PT Test Tbk", "yield": 6.5}, AS_OF, **kwargs,
    )


# ── Dates ───────────────────────────────────────────────────────────────── #

def test_cum_date_is_the_trading_day_before_ex():
    assert trading_scan.cum_date("2026-11-04") == datetime.date(2026, 11, 3)
    # Monday ex-date: the cum date is the Friday before.
    assert trading_scan.cum_date("2026-11-09") == datetime.date(2026, 11, 6)
    # Ex-date after Christmas: skips the weekend, 25 Dec and the 24 Dec cuti bersama.
    assert trading_scan.cum_date("2026-12-28") == datetime.date(2026, 12, 23)


def test_planned_exit_follows_the_exit_rule():
    assert trading_scan.planned_exit(dp.BASELINE_SETUP, "2026-11-04") == datetime.date(2026, 11, 3)
    after = {**dp.BASELINE_SETUP, "exit_rule": dp.EXIT_AFTER, "hold_after_days": 5}
    # 4 Nov + 5 days is a Monday; a weekend target rolls to the next session.
    assert trading_scan.planned_exit(after, "2026-11-04") == datetime.date(2026, 11, 9)
    assert trading_scan.planned_exit({**after, "hold_after_days": 3}, "2026-11-04") == datetime.date(2026, 11, 9)


def test_window_progress_counts_sessions():
    progress = trading_scan.window_progress(pd.Timestamp("2026-09-28"), pd.Timestamp("2026-10-09"), AS_OF)

    assert progress == {"day": 6, "of": 10, "sessions_left": 4}


# ── Signals ─────────────────────────────────────────────────────────────── #

def test_fixed_entry_inside_window_is_actionable():
    # Last year's ex-date rolls forward to 4 Dec 2026: the 90-30 day window is 5 Sep - 4 Nov.
    signal = _signal(_prices(np.linspace(900, 1000, 120)), _divs("2025-12-04"), FIXED_SETUP)

    assert signal["status"] == "actionable"
    assert signal["entry_price"] == pytest.approx(1000.0)
    assert signal["stop_price"] is None and signal["target_price"] is None
    assert signal["key_date"] == datetime.date(2026, 12, 3)
    assert signal["exit_by"] == datetime.date(2026, 12, 3)
    assert signal["key_date_estimated"] is True
    detail = signal["detail"]
    assert detail["expected_ex_date"] == "2026-12-04"
    assert detail["based_on"] == "2025-12-04"
    assert detail["window"]["of"] == 43
    assert detail["window"]["day"] + detail["window"]["sessions_left"] == 43
    assert detail["holdout"] == {"level": "global", "trades": 40, "wins": 30, "win_rate": 75.0, "avg_return": 4.2}
    assert detail["next_yield"] == pytest.approx(10.0)
    assert detail["company"] == "PT Test Tbk"


def test_dip_rule_needs_the_dip_before_it_is_actionable():
    rising = _prices(np.linspace(900, 1000, 120))

    assert _signal(rising, _divs("2025-12-04"), DIP_SETUP) is None


def test_active_dip_is_actionable_at_the_dip_level_with_exits():
    closes = np.r_[np.full(100, 1000.0), np.linspace(1000, 900, 20)]
    setup = {**DIP_SETUP, "take_profit_pct": 10.0, "stop_loss_pct": 5.0}

    signal = _signal(_prices(closes), _divs("2025-12-04"), setup)

    sma = closes[-20:].mean()
    assert signal["status"] == "actionable"
    assert signal["entry_price"] == pytest.approx(sma)
    assert signal["target_price"] == pytest.approx(sma * 1.10)
    assert signal["stop_price"] == pytest.approx(sma * 0.95)
    assert signal["detail"]["dip"]["active"] is True


def test_window_opening_soon_and_later():
    prices = _prices(np.linspace(900, 1000, 120))
    # Ex-date 4 Feb 2027: the window opens 6 Nov 2026, 32 days away.
    soon = _signal(prices, _divs("2026-02-04"), FIXED_SETUP)

    assert soon["status"] == "opening_soon"
    assert soon["entry_price"] is None
    assert soon["detail"]["days_to_window"] == 32
    assert "window" not in soon["detail"]
    assert _signal(prices, _divs("2026-02-04"), FIXED_SETUP, opening_within=30) is None
    assert _signal(prices, _divs("2026-06-04"), FIXED_SETUP) is None


def test_announced_ex_date_beats_last_years_estimate():
    # Last year's 4 Dec dividend is already announced for 30 Nov this year.
    divs = _divs("2025-12-04", "2026-11-30")

    signal = _signal(_prices(np.linspace(900, 1000, 120)), divs, FIXED_SETUP)

    assert signal["key_date_estimated"] is False
    assert signal["detail"]["expected_ex_date"] == "2026-11-30"
    assert signal["key_date"] == datetime.date(2026, 11, 27)


def test_stopped_dividend_schedule_is_no_signal():
    # The last dividend was in 2020: rolling it forward would invent an ex-date.
    assert _signal(_prices(np.linspace(900, 1000, 120)), _divs("2020-12-04"), FIXED_SETUP) is None


# ── Scan job ────────────────────────────────────────────────────────────── #

def _universe(symbols):
    return pd.DataFrame({
        "stock": symbols, "companyName": [f"PT {s} Tbk" for s in symbols],
        "yield": 6.0, "numDividendYear": 10, "mktCap": 5e12, "price": 1000.0, "volAvg": 5e6,
    })


def _loaders(universe, events, histories):
    return (
        patch.object(trading_data, "load_universe", return_value=universe),
        patch.object(trading_data, "load_dividend_events", return_value=events),
        patch.object(trading_data, "load_price_histories", return_value=histories),
    )


def test_scan_publishes_signals_with_stock_setups_first():
    client = FakeSupabase()
    store.write_setups(client, "dividend_play", [
        {"scope": "global", "params": FIXED_SETUP, "holdout": HOLDOUT},
        {"scope": "OWN.JK", "params": {**FIXED_SETUP, "entry_start_days": 55}, "holdout": {**HOLDOUT, "level": "stock"}},
    ])
    rising = _prices(np.linspace(900, 1000, 120))
    stale = _prices(np.linspace(900, 1000, 120), end=AS_OF - pd.Timedelta(days=30))
    events = {s: _divs("2025-12-04") for s in ("GLOBAL.JK", "OWN.JK", "STALE.JK")}
    events["SOON.JK"] = _divs("2026-02-04")
    histories = {"GLOBAL.JK": rising, "OWN.JK": rising, "STALE.JK": stale, "SOON.JK": rising}
    universe = _universe(["GLOBAL.JK", "OWN.JK", "STALE.JK", "SOON.JK", "NODATA.JK"])
    events["NODATA.JK"] = _divs("2025-12-04")

    a, b, c = _loaders(universe, events, histories)
    with a, b, c:
        stats = trading_scan.dividend_play(client)

    signals = store.read_signals(client, "dividend_play").set_index("symbol")
    assert signals.loc["GLOBAL.JK", "status"] == "actionable"
    assert signals.loc["GLOBAL.JK", "setup_scope"] == "global"
    # OWN.JK's own 55-30 day window opens on 10 Oct; its setup beats the global one.
    assert signals.loc["OWN.JK", "status"] == "opening_soon"
    assert signals.loc["OWN.JK", "setup_scope"] == "OWN.JK"
    assert signals.loc["SOON.JK", "status"] == "opening_soon"
    assert "STALE.JK" not in signals.index
    assert stats["as_of"] == "2026-10-05"
    assert stats["actionable"] == 1 and stats["opening_soon"] == 2
    assert stats["stale_prices"] == 1 and stats["no_data"] == 1
    run = store.latest_run(client, "dividend_play", "scan")
    assert run["as_of"] == "2026-10-05" and run["stats"]["actionable"] == 1


def test_scan_without_setups_publishes_nothing():
    client = FakeSupabase()
    a, b, c = _loaders(_universe(["A.JK"]), {"A.JK": _divs("2025-12-04")},
                       {"A.JK": _prices(np.linspace(900, 1000, 120))})
    with a, b, c:
        stats = trading_scan.dividend_play(client)

    assert stats["no_setup"] == 1
    assert store.read_signals(client, "dividend_play").empty
    assert store.latest_run(client, "dividend_play", "scan")["status"] == "ok"


def test_scan_failure_is_recorded():
    client = FakeSupabase()
    store.write_setups(client, "dividend_play", [{"scope": "global", "params": FIXED_SETUP, "holdout": HOLDOUT}])
    a, b, c = _loaders(_universe(["A.JK"]), {"A.JK": _divs("2025-12-04")},
                       {"A.JK": _prices(np.linspace(900, 1000, 120))})
    with a, b, c, patch.object(trading_scan, "dividend_play_signal", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            trading_scan.dividend_play(client)

    run = store.latest_run(client, "dividend_play", "scan", status="failed")
    assert "boom" in run["stats"]["error"]


# ── Optimize job ────────────────────────────────────────────────────────── #

def _run_up_history(seed):
    """Prices that fall after each half-yearly ex-date and climb into the next one."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-01", AS_OF)
    ex_dates = pd.to_datetime([f"{year}-{month}-15" for year in range(2015, 2026) for month in ("06", "12")])
    days_to_ex = np.array([
        (ex_dates[ex_dates >= d].min() - d).days if (ex_dates >= d).any() else 365 for d in dates
    ])
    trend = 1000 * (1 + 0.15 * (1 - np.clip(days_to_ex, 0, 120) / 120))
    close = trend * (1 + rng.normal(0, 0.003, len(dates)))
    return pd.DataFrame({"date": dates, "close": close}), _divs(*ex_dates, amount=30.0)


def test_optimize_publishes_global_and_stock_setups():
    client = FakeSupabase()
    symbols = [f"S{i:02d}.JK" for i in range(25)]
    data = {s: _run_up_history(i) for i, s in enumerate(symbols)}
    a, b, c = _loaders(_universe(symbols), {s: d[1] for s, d in data.items()}, {s: d[0] for s, d in data.items()})
    with a, b, c:
        stats = trading_optimize.dividend_play(client, as_of=AS_OF, preset="Quick")

    setups = store.read_setups(client, "dividend_play")
    assert stats["stocks_tested"] == 25
    assert stats["global_setup"] is not None
    assert setups["global"]["holdout"]["level"] == "global"
    assert setups["global"]["holdout"]["avg_trade"] > 0
    assert set(setups["global"]["params"]) == set(dp.SETUP_FIELDS)
    stock_scopes = set(setups) - {"global"}
    assert stats["stock_setups"] == len(stock_scopes) > 0
    assert all(setups[s]["holdout"]["level"] == "stock" for s in stock_scopes)
    run = store.latest_run(client, "dividend_play", "optimize")
    assert run["stats"]["setups_searched"] == len(dp.build_setup_grid(**dp.SEARCH_PRESETS["Quick"]))
    assert setups["global"]["run_id"] == run["id"]


def test_optimize_failure_keeps_published_setups():
    client = FakeSupabase()
    store.write_setups(client, "dividend_play", [{"scope": "global", "params": FIXED_SETUP, "holdout": HOLDOUT}])
    a, b, c = _loaders(_universe([]), {}, {})
    with a, b, c:
        with pytest.raises(RuntimeError, match="shortlist is empty"):
            trading_optimize.dividend_play(client, as_of=AS_OF)

    assert "global" in store.read_setups(client, "dividend_play")
    assert store.latest_run(client, "dividend_play", "optimize", status="failed") is not None


def test_stock_holdout_stats_match_pooled_formula():
    returns = np.array([5.0, -2.0, 3.0, 4.0])
    row = pd.Series({
        "Validation Trades": 4, "Validation Win Rate": 75.0, "Validation Avg Trade": returns.mean(),
        "Validation Gains": 12.0, "Validation Losses": 2.0, "Validation Sum": returns.sum(),
        "Validation Sum Sq": (returns ** 2).sum(), "Avg Days": 40.0, "Return / Month": 1.9,
        "Train Avg Trade": 2.0, "Stock Setup Score": 80.0,
    })

    holdout = trading_optimize.dividend_play_stock_holdout(row)

    assert holdout["profit_factor"] == pytest.approx(6.0)
    assert holdout["t_stat"] == pytest.approx(returns.mean() / (returns.std(ddof=1) / 2))


# ── Command line ────────────────────────────────────────────────────────── #

@pytest.mark.parametrize("module", [trading_scan, trading_optimize])
def test_cli_requires_a_known_tool(module):
    assert module.build_parser().parse_args(["--tool", "dividend_play"]).tool == "dividend_play"
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--tool", "crypto"])


@pytest.mark.parametrize("module", [trading_scan, trading_optimize])
def test_cli_fails_without_environment(module):
    with patch.dict("os.environ", {}, clear=True):
        assert module.main(["--tool", "dividend_play"]) == 1


def test_cli_passes_as_of():
    with patch.object(trading_data, "validate_environment"), patch.object(
        trading_data, "get_supabase_client", return_value="client"
    ), patch.dict(trading_scan.TOOLS, {"dividend_play": lambda client, as_of: {"as_of": as_of}}) :
        assert trading_scan.main(["--tool", "dividend_play", "--as-of", "2026-10-02"]) == 0



# ── Swing jobs ──────────────────────────────────────────────────────────── #

import harvest.swing as sw


SWING_SETUP = {"strategy_type": sw.STRATEGY_REGIME_RSI, "param_a": 14.0, "param_b": 30.0, "param_c": 50.0,
               "param_d": 45.0, "max_hold": 5, "profit_target": 5.0, "stop_loss": 2.0}


def _swing_bars(closes, end=AS_OF):
    dates = pd.bdate_range(end=end, periods=len(closes))
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": closes, "high": closes * 1.01, "low": closes * 0.99, "close": closes}, index=dates)


def test_swing_signal_fires_with_card_prices():
    bars = _swing_bars(np.r_[np.full(100, 1000.0), np.linspace(1000, 800, 20)])  # steady fall: RSI very low
    regime = pd.Series("bear", index=bars.index)

    signal = trading_scan.swing_signal(
        "TEST.JK", bars, regime, SWING_SETUP, "global", {"level": "global", "trades": 50, "win_rate": 60.0, "avg_trade": 0.8},
        {"companyName": "PT Test Tbk"}, AS_OF, bars["close"], pd.Series(dtype=float),
    )

    assert signal["status"] == "actionable"
    assert signal["entry_price"] == 800.0
    assert signal["stop_price"] == 780.0 and signal["target_price"] == 840.0  # Rp 5 ticks between Rp 500 and 2,000
    assert signal["key_date"] == datetime.date(2026, 10, 6)  # entry at the next session's open
    assert signal["exit_by"] == datetime.date(2026, 10, 13)  # five sessions after the entry day
    detail = signal["detail"]
    assert detail["price_limits"] == {"floor": 680.0, "ceiling": 1000.0}
    assert detail["holdout"]["wins"] == 30


def test_swing_signal_needs_todays_bar_and_a_setup():
    falling = _swing_bars(np.linspace(1000, 800, 120), end=AS_OF - pd.Timedelta(days=3))
    flat = _swing_bars(np.full(120, 1000.0) + np.sin(np.arange(120)))
    for bars in (falling, flat):
        assert trading_scan.swing_signal(
            "TEST.JK", bars, pd.Series(dtype=str), SWING_SETUP, "global", {}, {}, AS_OF,
            bars["close"], pd.Series(dtype=float),
        ) is None


def _patch_swing_inputs(shortlist, histories, regime=None, breadth=None):
    index = next(iter(histories.values()))["close"]
    return patch.object(trading_optimize, "swing_inputs", return_value=(
        shortlist, histories, regime if regime is not None else pd.Series("bear", index=index.index),
        breadth if breadth is not None else pd.Series(dtype=float), index,
    ))


def _cycles(seed, years=8):
    """A price that dips to RSI lows and rebounds, giving Regime RSI a real edge."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=AS_OF, periods=252 * years)
    phase = np.arange(len(dates)) % 30
    close = 1000 * np.where(phase < 10, 1 - phase * 0.012, 0.88 + (phase - 10) * 0.009) * (1 + rng.normal(0, 0.002, len(dates)))
    return pd.DataFrame({"open": close, "high": close * 1.004, "low": close * 0.996, "close": close}, index=dates)


def test_swing_optimize_publishes_a_walk_forward_winner():
    client = FakeSupabase()
    symbols = [f"S{i}.JK" for i in range(6)]
    histories = {s: _cycles(i) for i, s in enumerate(symbols)}
    with _patch_swing_inputs(pd.DataFrame({"stock": symbols}), histories), \
            patch.object(sw, "preset_combos", return_value=[
                (sw.STRATEGY_REGIME_RSI, 14, 30, 50, 45, 5, 5.0, 2.0),
                (sw.STRATEGY_REGIME_RSI, 7, 30, 50, 45, 5, 5.0, 2.0),
            ]):
        stats = trading_optimize.swing(client, as_of=AS_OF)

    setups = store.read_setups(client, "swing")
    assert stats["walk_forward"]["trades"] >= sw.MIN_WALK_FORWARD_TRADES
    assert stats["global_setup"] is not None
    assert setups["global"]["holdout"]["method"] == "walk_forward"
    assert len(setups["global"]["holdout"]["folds"]) == 6
    assert set(setups["global"]["params"]) == set(sw.SETUP_KEYS)


def test_swing_optimize_publishes_nothing_without_an_edge():
    client = FakeSupabase()
    store.write_setups(client, "swing", [{"scope": "global", "params": SWING_SETUP, "holdout": {}}])
    rng = np.random.default_rng(7)
    symbols = [f"S{i}.JK" for i in range(4)]
    histories = {}
    for s in symbols:
        close = 1000 * np.cumprod(1 + rng.normal(0, 0.015, 252 * 8))
        histories[s] = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close},
                                    index=pd.bdate_range(end=AS_OF, periods=len(close)))
    with _patch_swing_inputs(pd.DataFrame({"stock": symbols}), histories):
        stats = trading_optimize.swing(client, as_of=AS_OF)

    assert stats["global_setup"] is None and stats["stock_setups"] == 0
    assert store.read_setups(client, "swing") == {}  # last week's setup is withdrawn
    assert store.latest_run(client, "swing", "optimize")["status"] == "ok"


def test_swing_scan_publishes_from_stored_setups():
    client = FakeSupabase()
    store.write_setups(client, "swing", [{"scope": "global", "params": SWING_SETUP,
                                          "holdout": {"level": "global", "trades": 50, "win_rate": 60.0}}])
    falling = _swing_bars(np.r_[np.full(100, 1000.0), np.linspace(1000, 800, 20)])
    flat = _swing_bars(np.full(120, 1000.0) + np.sin(np.arange(120)))
    with patch.object(trading_optimize, "swing_inputs", return_value=(
        pd.DataFrame({"stock": ["FALL.JK", "FLAT.JK"], "companyName": ["PT Fall", "PT Flat"]}),
        {"FALL.JK": falling, "FLAT.JK": flat}, pd.Series("bear", index=falling.index),
        pd.Series(dtype=float), falling["close"],
    )):
        stats = trading_scan.swing(client)

    signals = store.read_signals(client, "swing")
    assert signals["symbol"].tolist() == ["FALL.JK"]
    assert stats["actionable"] == 1 and stats["as_of"] == AS_OF.date().isoformat()


# ── Long Term ───────────────────────────────────────────────────────────── #

import requests  # noqa: E402

import harvest.position as pos  # noqa: E402
from tests.test_position import make_financials, make_prices, make_statements  # noqa: E402


def _lt_profiles(symbols, industry="Conglomerates"):
    return pd.DataFrame(
        {"price": 100.0, "mktCap": 100.0 * 100, "companyName": [f"PT {s} Tbk" for s in symbols],
         "sector": "Industrials", "industry": industry},
        index=pd.Index(symbols, name="symbol"),
    )


# The fixtures' statements fit 100 shares at Rp 100 (P/E about 14), a Rp 10,000 company.
TINY_FLOOR = 1e-8


def _lt_inputs(fundamentals, histories, universe):
    def fetch(symbol):
        value = fundamentals[symbol]
        if isinstance(value, Exception):
            raise value
        return value

    return (
        patch.object(trading_data, "load_long_term_universe", return_value=universe),
        patch.object(trading_data, "load_price_histories", return_value=histories),
        patch.object(trading_data, "load_long_term_fundamentals", side_effect=fetch),
    )


def _lt_prices(**kwargs):
    frame = make_prices(end="2026-10-05", price=100.0, **kwargs)
    return frame.assign(date=pd.to_datetime(frame["date"])).sort_values("date").reset_index(drop=True)


def _growing():
    return make_financials(), pos.prepare_statements(*make_statements(ocf=200.0))


def test_long_term_scan_publishes_ranked_candidates_with_coverage(monkeypatch):
    monkeypatch.setattr(trading_scan, "LONG_TERM_DEFAULTS", {**trading_scan.LONG_TERM_DEFAULTS, "min_market_cap": TINY_FLOOR})
    client = FakeSupabase()
    symbols = ["GOOD.JK", "SHRINK.JK", "THIN.JK", "DOWN.JK", "NOPRICE.JK"]
    universe = _lt_profiles(symbols)
    fundamentals = {
        "GOOD.JK": _growing(),
        "SHRINK.JK": (make_financials(growth=-0.10), pos.prepare_statements(*make_statements(ocf=200.0))),
        "THIN.JK": (make_financials(quarters=6), pos.prepare_statements(*make_statements())),
        "DOWN.JK": requests.HTTPError("403 Forbidden"),
    }
    histories = {s: _lt_prices() for s in symbols if s != "NOPRICE.JK"}

    a, b, c = _lt_inputs(fundamentals, histories, universe)
    with a, b, c:
        stats = trading_scan.long_term(client)

    signals = store.read_signals(client, "long_term")
    assert signals["symbol"].tolist() == ["GOOD.JK"]
    good = signals.iloc[0]
    assert good["status"] == "candidate"
    assert good["as_of"] == datetime.date(2026, 10, 5)
    assert good["entry_price"] == pytest.approx(100.0)
    assert good["target_price"] > good["entry_price"]
    assert good["key_date"] == datetime.date(2026, 6, 30)
    assert good["detail"]["rank"] == 1 and good["detail"]["basis"] == "P/S"
    assert good["detail"]["quality"]["complete"] is True
    assert stats["requested"] == 5 and stats["analyzed"] == 2 and stats["candidates"] == 1
    assert stats["skipped"] == 2
    assert stats["skipped_by_reason"] == {"fewer than 8 quarterly statements": 1, "no prices": 1}
    assert stats["failed"] == 1 and "HTTPError" in stats["failed_symbols"]["DOWN.JK"]
    assert stats["failed_rules"]["Revenue TTM falling"] == 1
    assert store.latest_run(client, "long_term", "scan")["stats"]["analyzed"] == 2


def test_long_term_scan_uses_the_published_screen(monkeypatch):
    client = FakeSupabase()
    store.write_setups(client, "long_term", [{
        "scope": "global", "params": {**pos.DEFAULT_FILTERS, "min_market_cap": TINY_FLOOR, "max_pe": 5.0},
        "holdout": {"years": 8, "cagr": 12.0},
    }])
    a, b, c = _lt_inputs({"GOOD.JK": _growing()}, {"GOOD.JK": _lt_prices()}, _lt_profiles(["GOOD.JK"]))
    with a, b, c:
        stats = trading_scan.long_term(client)

    assert stats["candidates"] == 0 and stats["failed_rules"] == {"P/E > 5": 1}
    assert stats["published_screen"] is True


def test_long_term_scan_checks_tracked_positions_outside_the_screen(monkeypatch):
    monkeypatch.setattr(trading_scan, "LONG_TERM_DEFAULTS", {**trading_scan.LONG_TERM_DEFAULTS, "min_market_cap": TINY_FLOOR})
    client = FakeSupabase()
    store.create_position(client, "fan@example.com", {
        "tool": "long_term", "symbol": "SMALL.JK", "entry_date": "2026-01-05", "entry_price": 900,
        "lots": 5, "exit_rule": {},
    })
    universe = pd.concat([_lt_profiles(["GOOD.JK"]), _lt_profiles(["SMALL.JK"]).assign(mktCap=5e3)])
    received = {}

    def fake_alerts(client, as_of, rows, positions):
        received.update(rows=set(rows), positions=positions["symbol"].tolist())
        return {"exit": 0}

    a, b, c = _lt_inputs({"GOOD.JK": _growing(), "SMALL.JK": _growing()},
                         {"GOOD.JK": _lt_prices(), "SMALL.JK": _lt_prices()}, universe)
    with a, b, c, patch.object(trading_scan.alerts, "send_long_term_exit_alerts", side_effect=fake_alerts):
        stats = trading_scan.long_term(client)

    assert received == {"rows": {"GOOD.JK", "SMALL.JK"}, "positions": ["SMALL.JK"]}
    assert store.read_signals(client, "long_term")["symbol"].tolist() == ["GOOD.JK"]
    assert stats["positions_checked"] == 1 and stats["requested"] == 2


def test_long_term_scan_failure_is_recorded(monkeypatch):
    monkeypatch.setattr(trading_scan, "LONG_TERM_DEFAULTS", {**trading_scan.LONG_TERM_DEFAULTS, "min_market_cap": TINY_FLOOR})
    client = FakeSupabase()
    a, b, c = _lt_inputs({"GOOD.JK": _growing()}, {"GOOD.JK": _lt_prices()}, _lt_profiles(["GOOD.JK"]))
    with a, b, c, patch.object(trading_scan.pos, "score_results", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            trading_scan.long_term(client)

    assert "boom" in store.latest_run(client, "long_term", "scan", status="failed")["stats"]["error"]


def test_with_retries_backs_off_on_rate_limits(monkeypatch):
    monkeypatch.setattr(trading_data.time, "sleep", lambda seconds: None)
    response = requests.Response()
    response.status_code = 429
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise requests.HTTPError(response=response)
        return "ok"

    assert trading_data.with_retries(flaky) == "ok" and len(calls) == 3

    forbidden = requests.Response()
    forbidden.status_code = 403
    calls.clear()

    def denied():
        calls.append(1)
        raise requests.HTTPError(response=forbidden)

    with pytest.raises(requests.HTTPError):
        trading_data.with_retries(denied)
    assert len(calls) == 1
