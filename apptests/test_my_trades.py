"""My Trades page and the Today view's Track this trade / Alert me controls."""
import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

import harvest.premium as premium
import harvest.trading_store as store
from tests.fake_supabase import FakeSupabase

# Shared fixtures; pytest finds them by name in this module.
from apptests.test_dividend_play_views import client, run_page, _texts  # noqa: F401,F811


REPO = Path(__file__).resolve().parents[1]
MY_TRADES = str(REPO / "apps" / "trading" / "my_trades.py")
TODAY = datetime.date.today()


def _position(symbol, **overrides):
    position = {
        "tool": "dividend_play", "symbol": symbol, "entry_date": TODAY - datetime.timedelta(days=20),
        "entry_price": 1000.0, "lots": 5, "exit_by": TODAY + datetime.timedelta(days=14),
        "exit_rule": {"setup_text": "Buy 90-30d before ex", "key_date": TODAY + datetime.timedelta(days=10)},
    }
    position.update(overrides)
    return position


@pytest.fixture
def trades_client():
    fake = FakeSupabase()
    store.create_position(fake, "fan@example.com", _position("BBRI.JK"))
    closed = store.create_position(fake, "fan@example.com", _position("TLKM.JK"))
    store.close_position(fake, "fan@example.com", closed["id"], TODAY - datetime.timedelta(days=2), 1100)
    store.create_position(fake, "other@example.com", _position("ASII.JK"))
    return fake


def _run_trades(monkeypatch, client, email, enforced=False, bot=None):
    import streamlit as st

    st.cache_data.clear()
    if enforced:
        monkeypatch.setenv("PREMIUM_ENFORCED", "true")
    else:
        monkeypatch.delenv("PREMIUM_ENFORCED", raising=False)
    if bot:
        monkeypatch.setenv("TELEGRAM_BOT_USERNAME", bot)
    else:
        monkeypatch.delenv("TELEGRAM_BOT_USERNAME", raising=False)
    patches = [
        patch.object(premium, "current_email", return_value=email),
        patch.object(premium, "is_premium", side_effect=lambda e: e == "fan@example.com"),
        patch.object(premium, "get_connection", return_value=client),
    ]
    for item in patches:
        item.start()
    monkeypatch.chdir(REPO)
    app = AppTest.from_file(MY_TRADES, default_timeout=60).run()
    return app, patches


@pytest.fixture
def trades(monkeypatch, trades_client):
    started = []

    def run(email="fan@example.com", **kwargs):
        for item in started:
            item.stop()
        started.clear()
        app, patches = _run_trades(monkeypatch, trades_client, email, **kwargs)
        started.extend(patches)
        return app

    yield run
    for item in started:
        item.stop()


def test_my_trades_asks_logged_out_visitors_to_log_in(trades):
    app = trades(email=None)

    assert not app.exception
    assert [b.label for b in app.button] == ["Log in with Google"]
    assert not app.subheader


def test_my_trades_is_gated_for_free_users(trades):
    app = trades(email="free@example.com", enforced=True)

    assert not app.exception
    assert app.title[-1].value == "Unlock the trading lab"


def test_my_trades_lists_only_the_users_positions(trades):
    app = trades()

    assert not app.exception
    text = _texts(app)
    assert "**BBRI** · Dividend Play · 5 lots at Rp 1,000" in text
    assert "sessions left, sell by" in text
    assert "ASII" not in text
    closed = app.dataframe[0].value
    assert closed["Stock"].tolist() == ["TLKM"]
    expected = (1100 * (1 - 0.0025) / (1000 * 1.0015) - 1) * 100
    assert closed["Return"].iloc[0] == pytest.approx(expected)
    assert next(m.value for m in app.metric if m.label == "Won") == "1 of 1"


def test_settings_form_saves_fees_and_alert_choice(trades, trades_client):
    app = trades()

    app.number_input[0].set_value(0.2)
    app.toggle[0].set_value(True)
    app = next(b for b in app.button if b.label == "Save settings").click().run()

    saved = store.get_settings(trades_client, "fan@example.com")
    assert saved["buy_fee_pct"] == 0.2
    assert saved["alert_prefs"]["new_signals"] is True


def test_link_telegram_creates_a_start_link(trades, trades_client):
    app = trades(bot="panen_bot")

    app = next(b for b in app.button if b.label == "Link Telegram").click().run()

    token = store.get_settings(trades_client, "fan@example.com")["telegram_link_token"]
    assert token
    assert "Open the bot and press **Start**" in _texts(app)


def test_telegram_section_without_a_bot(trades):
    assert "Telegram alerts are not available yet." in _texts(trades())


# ── Today: Track this trade and Alert me ────────────────────────────────── #

def test_track_this_trade_opens_a_prefilled_dialog(run_page):
    # AppTest can't replay clicks inside a dialog (they rerun only the dialog fragment),
    # so saving is covered by the store tests and checked in the browser.
    app = run_page("premium")
    app = next(b for b in app.button if b.key == "track_BBRI.JK").click().run()

    assert not app.exception
    entry = next(n for n in app.number_input if n.label == "Entry price (Rp)")
    assert entry.value == 4010.0  # the card's tick-rounded entry level
    assert next(n for n in app.number_input if n.label == "Lots").value == 1
    assert "Track" in [b.label for b in app.button]


def test_alert_me_pills_save_the_watchlist(run_page, client):
    store.save_settings(client, "fan@example.com", {"alert_prefs": {"watch": ["UNVR.JK"]}})
    with patch.object(premium, "current_email", return_value="fan@example.com"):
        app = run_page("premium")
        pills = app.pills(key="alert_watch")
        assert pills.options == ["ASII"]
        app = pills.set_value(["ASII.JK"]).run()

    watch = store.get_settings(client, "fan@example.com")["alert_prefs"]["watch"]
    assert sorted(watch) == ["ASII.JK", "UNVR.JK"]  # stocks outside today's list are kept
    assert "Link it under Settings in My Trades" in _texts(app)


def test_logged_out_visitors_get_login_buttons(run_page):
    app = run_page("open")

    labels = [b.label for b in app.button]
    assert "Log in to get alerts" in labels
    # Track this trade is still offered; it starts the Google login.
    assert "Track this trade" in labels
