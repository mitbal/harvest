"""My Trades as the hub: signals from every tool, what is coming up and your positions, per access tier."""
import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import harvest.premium as premium
import harvest.trading_store as store
from apptests import test_long_term_views as long_term
from apptests import test_swing_views as swing
from apptests.test_dividend_play_views import AS_OF, _metric, _signal as dividend_signal, _texts
from tests.fake_supabase import FakeSupabase


REPO = Path(__file__).resolve().parents[1]
PAGE = str(REPO / "apps" / "trading" / "my_trades.py")
TODAY = datetime.date.today()


def _publish(fake, tool, signals):
    run_id = store.start_run(fake, tool, "scan", AS_OF)
    store.finish_run(fake, run_id, "ok", {})
    store.write_signals(fake, tool, AS_OF, signals)


def _client(published=True) -> FakeSupabase:
    fake = FakeSupabase()
    if published:
        _publish(fake, "dividend_play", [
            dividend_signal("BBRI.JK", "actionable", 5.0),
            dividend_signal("TLKM.JK", "actionable", 3.0),
            dividend_signal("ASII.JK", "opening_soon", 2.0),
        ])
        _publish(fake, "swing", [
            swing._signal("BMRI.JK", "actionable", 0.9), swing._signal("ADRO.JK", "actionable", 0.5),
            swing._signal("GOTO.JK", "opening_soon", 0.5),
        ])
        _publish(fake, "long_term", long_term._signals())
        store.create_position(fake, "fan@example.com", {
            "tool": "dividend_play", "symbol": "UNVR.JK", "entry_date": TODAY - datetime.timedelta(days=20),
            "entry_price": 2000.0, "lots": 3, "exit_by": TODAY + datetime.timedelta(days=14),
            "exit_rule": {"setup_text": "Buy 90-30d before ex", "key_date": TODAY + datetime.timedelta(days=10)},
        })
        store.create_position(fake, "other@example.com", {
            "tool": "swing", "symbol": "ICBP.JK", "entry_date": TODAY, "entry_price": 1000.0, "lots": 1,
            "exit_by": TODAY + datetime.timedelta(days=5), "exit_rule": {},
        })
    fake.auth = MagicMock()
    return fake


@pytest.fixture
def run_page(monkeypatch):
    import streamlit as st

    monkeypatch.chdir(REPO)
    active = []

    def run(tier="premium", client=None, connection_error=False):
        for item in active:
            item.stop()
        active.clear()
        st.cache_data.clear()
        if tier == "premium":
            monkeypatch.delenv("PREMIUM_ENFORCED", raising=False)
        else:
            monkeypatch.setenv("PREMIUM_ENFORCED", "true")
        email = {"premium": "fan@example.com", "free": "free@example.com", "logged_out": None}[tier]
        connection = (patch.object(premium, "get_connection", side_effect=RuntimeError("store down"))
                      if connection_error else patch.object(premium, "get_connection", return_value=client or _client()))
        active.extend([
            patch.object(premium, "current_email", return_value=email),
            patch.object(premium, "is_premium", side_effect=lambda e: e == "fan@example.com"),
            connection,
        ])
        for item in active:
            item.start()
        return AppTest.from_file(PAGE, default_timeout=60).run()

    yield run
    for item in active:
        item.stop()


def _cards(app) -> list[str]:
    return [b.key.removeprefix("open_") for b in app.button if (b.key or "").startswith("open_")]


def test_supporters_see_signals_from_every_tool(run_page):
    app = run_page()

    assert not app.exception
    cards = _cards(app)
    assert cards[:4] == ["dividend_play_BBRI.JK", "dividend_play_TLKM.JK", "swing_BMRI.JK", "swing_ADRO.JK"]
    assert any(card.startswith("long_term_") for card in cards)
    assert _metric(app, "Signals today") == "4"
    text = _texts(app)
    assert "Dividend Play 2 Entry window open now" in text and "Swing Trading 2 Plans for the next session" in text
    assert "Holdout 30 of 40 won, +5.0% average after costs" in text
    assert "Stop −2% about Rp 980" in text
    assert not any(b.label == "Unlock the trading lab" for b in app.button)


def test_free_visitors_get_one_card_per_tool_and_an_unlock_prompt(run_page):
    app = run_page("free")

    assert not app.exception
    cards = _cards(app)
    assert cards[:2] == ["dividend_play_BBRI.JK", "swing_BMRI.JK"]
    assert len(cards) == 3
    assert any(b.label == "Unlock the trading lab" for b in app.button)
    assert "TLKM" in _texts(app) and "ADRO" in _texts(app)


def test_positions_and_coming_up(run_page):
    app = run_page()

    assert "UNVR Dividend Play · bought" in _texts(app)
    assert _metric(app, "Your positions") == "1"
    events = app.dataframe[0].value
    assert set(events["Stock"]) == {"ASII", "GOTO", "UNVR"}
    assert "Sell by, your position" in events["What"].tolist()
    assert "Cum date, your position" in events["What"].tolist()
    assert "ICBP" not in _texts(app)
    assert events["Date"].is_monotonic_increasing


def test_logged_out_visitors_are_asked_to_log_in_for_positions(run_page):
    app = run_page("logged_out")

    assert not app.exception
    assert "Log in to track your trades" in _texts(app)
    assert any(b.label == "Log in with Google" for b in app.button)
    events = app.dataframe[0].value
    assert set(events["Stock"]) == {"ASII", "GOTO"}


def test_nothing_published_yet(run_page):
    app = run_page(client=_client(published=False))

    assert not app.exception
    assert _texts(app).count("Not published yet") == 3
    assert "No open positions" in _texts(app) and "Nothing scheduled" in _texts(app)


def test_store_outage_shows_a_warning(run_page):
    app = run_page(connection_error=True)

    assert not app.exception
    assert "Signals are unavailable right now" in app.warning[0].value


def test_see_evidence_opens_the_tool_on_that_stock(run_page):
    app = run_page()

    with patch("streamlit.switch_page") as switch:
        next(b for b in app.button if b.key == "evidence_swing_BMRI.JK").click().run()

    switch.assert_called_once_with("apps/trading/swing_trading.py",
                                   query_params={"view": "Evidence", "stock": "BMRI.JK"})
