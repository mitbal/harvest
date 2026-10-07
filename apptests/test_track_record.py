"""Track record page: settled results for everyone, open trades for supporters."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

import harvest.premium as premium
import harvest.trading_store as store
from apptests.test_dividend_play_views import _metric, _texts
from tests.fake_supabase import FakeSupabase


REPO = Path(__file__).resolve().parents[1]
PAGE = str(REPO / "apps" / "trading" / "track_record.py")


def _trade(symbol, signal_date, status, result=None, **fields):
    row = {"symbol": symbol, "signal_date": signal_date, "status": status, "return_pct": result,
           "detail": {"expected": 3.0, "cards": 1, "limit": 990.0}}
    if status != "not_filled":
        row.update({"entry_date": signal_date, "entry_price": 1000.0, "benchmark_pct": 1.0})
    if status == "closed":
        row.update({"exit_date": "2026-09-20", "exit_price": 1000.0 * (1 + result / 100),
                    "exit_reason": "Sell-by date"})
    if status == "open":
        row["detail"] = {**row["detail"], "last_close": 1020.0}
    row.update(fields)
    return row


def _client(published=True) -> FakeSupabase:
    fake = FakeSupabase()
    if published:
        for tool in ("dividend_play", "swing", "long_term"):
            run_id = store.start_run(fake, tool, "track", "2026-10-05")
            store.finish_run(fake, run_id, "ok", {"first_card": "2026-08-03"})
        store.write_outcomes(fake, "dividend_play", [
            _trade("BBRI.JK", "2026-08-03", "closed", 4.0),
            _trade("TLKM.JK", "2026-08-10", "closed", -1.5),
            _trade("ASII.JK", "2026-09-28", "open", 2.0),
            _trade("UNVR.JK", "2026-09-01", "not_filled"),
        ])
        store.write_outcomes(fake, "swing", [_trade("BMRI.JK", "2026-09-02", "closed", 1.2,
                                                    exit_reason="Target")])
        store.write_outcomes(fake, "long_term", [
            _trade("CMRY.JK", "2026-08-03", "closed", 20.0, exit_reason="Reached mean valuation", benchmark_pct=3.0),
            _trade("MSIN.JK", "2026-08-03", "open", -5.0, benchmark_pct=2.0),
        ])
    fake.auth = MagicMock()
    return fake


@pytest.fixture
def run_page(monkeypatch):
    import streamlit as st

    monkeypatch.chdir(REPO)
    active = []

    def run(tier="premium", tool=None, client=None, connection_error=False):
        for item in active:
            item.stop()
        active.clear()
        st.cache_data.clear()
        if tier == "premium":
            monkeypatch.delenv("PREMIUM_ENFORCED", raising=False)
        else:
            monkeypatch.setenv("PREMIUM_ENFORCED", "true")
        connection = (patch.object(premium, "get_connection", side_effect=RuntimeError("store down"))
                      if connection_error else patch.object(premium, "get_connection", return_value=client or _client()))
        active.extend([
            patch.object(premium, "current_email", return_value="fan@example.com" if tier == "premium" else None),
            patch.object(premium, "is_premium", side_effect=lambda e: e == "fan@example.com"),
            connection,
        ])
        for item in active:
            item.start()
        app = AppTest.from_file(PAGE, default_timeout=60)
        if tool:
            app.query_params["tool"] = tool
        return app.run()

    yield run
    for item in active:
        item.stop()


def test_dividend_play_record_for_supporters(run_page):
    app = run_page()

    assert not app.exception
    assert _metric(app, "Closed trades") == "2"
    assert _metric(app, "Won") == "1 of 2"
    assert _metric(app, "Average per trade") == "+1.2%"
    assert _metric(app, "Backtest said") == "+3.0%"
    closed, held, missed = (frame.value for frame in app.dataframe)
    assert closed["Stock"].tolist() == ["TLKM", "BBRI"]
    assert held["Stock"].tolist() == ["ASII"] and held["Sell price"].iloc[0] == 1020.0
    assert missed["Stock"].tolist() == ["UNVR"]
    assert app.get("arrow_vega_lite_chart") or app.get("vega_lite_chart")
    assert "Cards since Mon 3 Aug 2026" in _texts(app)


def test_free_visitors_see_settled_results_but_not_open_trades(run_page):
    app = run_page("free")

    assert not app.exception
    assert _metric(app, "Closed trades") == "2"
    tables = [frame.value["Stock"].tolist() for frame in app.dataframe]
    assert ["ASII"] not in tables and ["TLKM", "BBRI"] in tables
    assert "1 trade from recent cards is still open" in _texts(app)


def test_swing_tab_from_the_url(run_page):
    app = run_page(tool="Swing Trading")

    assert not app.exception
    assert _metric(app, "Closed trades") == "1"
    assert app.dataframe[0].value["Status"].tolist() == ["Target"]


def test_long_term_picks(run_page):
    app = run_page(tool="Long Term")

    assert not app.exception
    assert _metric(app, "Companies picked") == "2"
    assert _metric(app, "Ahead of the JKSE") == "1 of 2"
    assert app.dataframe[0].value["Stock"].tolist() == ["MSIN", "CMRY"]

    free = run_page("free", tool="Long Term")
    assert free.dataframe[0].value["Stock"].tolist() == ["CMRY"]
    assert "1 open pick is for supporters." in _texts(free)


def test_before_the_first_update(run_page):
    app = run_page(client=_client(published=False))

    assert not app.exception
    assert "The track record starts after the first nightly update" in _texts(app)


def test_store_outage(run_page):
    app = run_page(connection_error=True)

    assert not app.exception
    assert "The track record is unavailable right now" in app.warning[0].value
