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
