"""The nightly track record job and the outcome store."""
from unittest.mock import patch

import pandas as pd
import pytest

import harvest.trading_store as store
from pipeline import trading_data, trading_track
from tests.fake_supabase import FakeSupabase


DAYS = pd.bdate_range("2026-09-01", periods=40)


def _bars(closes, lows=None) -> pd.DataFrame:
    days = DAYS[:len(closes)]
    return pd.DataFrame({"open": closes, "high": closes, "low": lows or closes, "close": closes,
                         "volume": 1000}, index=days)


def _publish(client, day, signals, tool="dividend_play"):
    run_id = store.start_run(client, tool, "scan", DAYS[day])
    store.finish_run(client, run_id, "ok", {})
    store.write_signals(client, tool, DAYS[day], signals)


def _card(symbol, limit, exit_day):
    return {"symbol": symbol, "status": "actionable", "entry_price": limit, "exit_by": DAYS[exit_day],
            "key_date": DAYS[exit_day], "detail": {"params": {}, "holdout": {"avg_return": 3.0}}}


def _run(client, bars):
    with patch.object(trading_data, "load_ohlcv_histories", return_value=bars) as load, \
            patch.object(trading_data, "load_dividend_events", return_value={}):
        stats = trading_track.track(client, "dividend_play")
    return stats, load


def test_track_writes_outcomes_and_records_the_run():
    client = FakeSupabase()
    _publish(client, 0, [_card("BBRI.JK", 990.0, 4), _card("TLKM.JK", 990.0, 20)])
    _publish(client, 1, [])
    bars = {"BBRI.JK": _bars([1000.0] * 6, lows=[1000.0, 980.0, 1000.0, 1000.0, 1000.0, 1000.0]),
            "TLKM.JK": _bars([1000.0] * 6)}

    stats, load = _run(client, bars)

    assert stats["closed"] == 1 and stats["not_filled"] == 1 and stats["written"] == 2
    assert load.call_args.args[1] == DAYS[0] - pd.Timedelta(days=trading_track.PRICE_LEAD_DAYS)
    outcomes = store.read_outcomes(client, "dividend_play")
    assert outcomes.set_index("symbol")["status"].to_dict() == {"BBRI.JK": "closed", "TLKM.JK": "not_filled"}
    run = store.latest_run(client, "dividend_play", "track")
    assert run["stats"]["cards"] == 2


def test_settled_outcomes_are_frozen_and_open_ones_refreshed():
    client = FakeSupabase()
    _publish(client, 0, [_card("BBRI.JK", 990.0, 4), _card("ASII.JK", 995.0, 30)])
    dipping = [1000.0, 980.0, 1000.0, 1000.0, 1000.0, 1000.0]
    _run(client, {"BBRI.JK": _bars([1000.0] * 6, lows=dipping), "ASII.JK": _bars([1000.0] * 6, lows=dipping)})
    first = store.read_outcomes(client, "dividend_play").set_index("symbol")

    # Prices revised: the settled BBRI trade keeps its first result; the open ASII one moves.
    stats, _ = _run(client, {"BBRI.JK": _bars([1200.0] * 6, lows=dipping),
                             "ASII.JK": _bars([1100.0] * 6, lows=dipping)})
    second = store.read_outcomes(client, "dividend_play").set_index("symbol")

    assert stats["kept"] == 1 and stats["written"] == 1
    assert second.at["BBRI.JK", "return_pct"] == first.at["BBRI.JK", "return_pct"]
    assert second.at["ASII.JK", "status"] == "open"
    assert second.at["ASII.JK", "return_pct"] > first.at["ASII.JK", "return_pct"]


def test_open_outcomes_the_replay_drops_are_removed():
    client = FakeSupabase()
    store.write_outcomes(client, "swing", [{"symbol": "GONE.JK", "signal_date": "2026-09-01", "status": "open"}])

    assert store.write_outcomes(client, "swing", []) == {"written": 0, "kept": 0, "removed": 1}
    assert store.read_outcomes(client, "swing").empty


def test_signal_history_reads_every_page():
    client = FakeSupabase()
    _publish(client, 0, [_card(f"S{i:03}.JK", 100.0, 4) for i in range(25)])

    with patch.object(store, "PAGE_SIZE", 10):
        history = store.read_signal_history(client, "dividend_play", "actionable", "symbol,as_of,entry_price")

    assert len(history) == 25 and list(history.columns) == ["symbol", "as_of", "entry_price"]
    assert history["symbol"].is_unique


def test_outcome_status_is_checked():
    with pytest.raises(ValueError):
        store.write_outcomes(FakeSupabase(), "swing", [{"symbol": "X.JK", "signal_date": "2026-09-01",
                                                         "status": "pending"}])


def test_main_runs_every_tool_and_reports_failures():
    with patch.object(trading_data, "validate_environment"), \
            patch.object(trading_data, "get_supabase_client", return_value=FakeSupabase()), \
            patch.object(trading_track, "load_benchmark", return_value=None), \
            patch.object(trading_track, "track", side_effect=[{}, RuntimeError("boom"), {}]) as track:
        assert trading_track.main(["--tool", "all"]) == 1
    assert [c.args[1] for c in track.call_args_list] == ["dividend_play", "long_term", "swing"]
