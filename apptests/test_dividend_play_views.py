"""Today, Evidence and Lab views of the Dividend Play page, per access tier.

The page runs for real against an in-memory trading store, synthetic prices
and a synthetic dividend history; nothing touches the network.
"""
import pickle
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import harvest.data as hd
import harvest.premium as premium
import harvest.prices as hprices
import harvest.trading_store as store
from tests.fake_supabase import FakeSupabase


REPO = Path(__file__).resolve().parents[1]
PAGE = str(REPO / "apps" / "trading" / "dividend_play.py")
AS_OF = pd.Timestamp.today().normalize()
SYMBOLS = ["BBRI.JK", "TLKM.JK", "ASII.JK", "UNVR.JK"]
SETUP = {
    "entry_start_days": 90, "entry_end_days": 30, "entry_rule": "dip", "dip_ma_days": 20,
    "dip_pct": 0.0, "exit_rule": "after", "hold_after_days": 10,
    "take_profit_pct": 0.0, "stop_loss_pct": 0.0,
}


def _signal(symbol, status, avg_return, **detail):
    window_start = AS_OF - pd.Timedelta(days=10) if status == "actionable" else AS_OF + pd.Timedelta(days=12)
    return {
        "symbol": symbol, "status": status,
        "entry_price": 4012.0 if status == "actionable" else None,
        "exit_by": AS_OF + pd.Timedelta(days=70), "key_date": AS_OF + pd.Timedelta(days=60),
        "key_date_estimated": symbol != "BBRI.JK", "setup_scope": "global",
        "detail": {
            "company": f"PT {symbol[:4]} Tbk", "yield": 6.2, "next_dividend": 150.0,
            "setup": "Buy 90-30d before ex, dip 0% < MA20; sell 10d after ex",
            "window_start": window_start.date().isoformat(), "days_to_window": 12,
            "window": {"day": 8, "of": 43, "sessions_left": 35},
            "dip": {"rule": True, "ma_days": 20, "pct": 0.0, "active": True, "vs_ma_pct": -2.5},
            "holdout": {"level": "global", "trades": 40, "wins": 30, "win_rate": 75.0, "avg_return": avg_return},
            "history": {"trades": 20, "win_rate": 70.0, "avg_return": 3.4, "avg_days": 60.0},
            **detail,
        },
    }


@pytest.fixture
def client():
    fake = FakeSupabase()
    run_id = store.start_run(fake, "dividend_play", "scan", AS_OF)
    store.finish_run(fake, run_id, "ok", {})
    store.write_setups(fake, "dividend_play", [
        {"scope": "global", "params": SETUP, "holdout": {"level": "global", "trades": 40, "win_rate": 75.0, "avg_trade": 4.1}},
    ])
    store.write_signals(fake, "dividend_play", AS_OF, [
        _signal("BBRI.JK", "actionable", 5.0),
        _signal("TLKM.JK", "actionable", 3.0),
        _signal("ASII.JK", "opening_soon", 2.0),
    ])
    dividends = {
        symbol: pd.DataFrame({
            "date": [f"{year}-06-15" for year in range(2015, AS_OF.year)],
            "adjDividend": 100.0,
        })
        for symbol in SYMBOLS
    }
    fake.auth = MagicMock()
    fake.client = types.SimpleNamespace(storage=types.SimpleNamespace(
        from_=lambda bucket: types.SimpleNamespace(download=lambda path: pickle.dumps(dividends)),
    ))
    return fake


def _fake_prices(exchange, symbols=None, start=None, end=None, columns=("close",)):
    dates = pd.bdate_range("2014-01-01", AS_OF)
    wave = 1000 + 100 * np.sin(np.arange(len(dates)) / 40)
    return pd.concat(
        [pd.DataFrame({"symbol": symbol, "date": dates, "close": wave}) for symbol in (symbols or SYMBOLS)],
        ignore_index=True,
    )


@pytest.fixture
def run_page(monkeypatch, client):
    """Run the page for a tier; patches stay active until the test ends so clicks can rerun it."""
    import streamlit as st

    monkeypatch.chdir(REPO)
    monkeypatch.delenv("REDIS_URL", raising=False)
    active = []

    def run(tier="premium", view=None, stock=None, connection=None):
        for item in active:
            item.stop()
        active.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
        # "open": the gate is off and nobody is logged in.
        if tier in ("premium", "open"):
            monkeypatch.delenv("PREMIUM_ENFORCED", raising=False)
        else:
            monkeypatch.setenv("PREMIUM_ENFORCED", "true")
        email = {"premium": "fan@example.com", "free": "free@example.com", "logged_out": None, "open": None}[tier]
        active.extend([
            patch.object(premium, "current_email", return_value=email),
            patch.object(premium, "is_premium", side_effect=lambda e: e == "fan@example.com"),
            patch.object(premium, "get_connection", return_value=connection or client),
            patch("streamlit.connection", return_value=client),
            patch.object(hprices, "load_prices", side_effect=_fake_prices),
            patch.object(hd, "get_daily_stock_price", side_effect=RuntimeError("no network")),
        ])
        for item in active:
            item.start()
        app = AppTest.from_file(PAGE, default_timeout=120)
        if view:
            app.query_params["view"] = view
        if stock:
            app.query_params["stock"] = stock
        return app.run()

    yield run
    for item in active:
        item.stop()


def _texts(app) -> str:
    parts = [e.value for e in app.markdown] + [e.value for e in app.caption] + [e.value for e in app.subheader]
    parts += [e.value for e in app.title] + [e.value for e in app.info] + [e.value for e in app.warning]
    return "\n".join(str(p) for p in parts)


def _metric(app, label):
    return next(m.value for m in app.metric if m.label == label)


def _evidence_buttons(app):
    return [b for b in app.button if b.label == "See evidence"]


# ── Today ───────────────────────────────────────────────────────────────── #

def test_today_premium_shows_every_card_and_the_opening_list(run_page):
    app = run_page("premium")

    assert not app.exception
    assert app.title[0].value == "Dividend Play"
    assert _metric(app, "Entry windows open") == "2"
    assert _metric(app, "Opening in 14 days") == "1"
    assert len(_evidence_buttons(app)) == 2
    text = _texts(app)
    assert "**BBRI** · PT BBRI Tbk" in text and "**TLKM** · PT TLKM Tbk" in text
    assert "Yield 6.2% · next dividend Rp 150" in text
    assert "Entry at or below **Rp 4,010**" in text  # rounded down to the IDX tick
    assert "Dip rule active, 2.5% below its 20-day average" in text
    assert "holdout 30 of 40 won on all shortlist stocks, +5.0% average after costs" in text
    assert "Not financial advice." in text
    assert app.dataframe[0].value["Stock"].tolist() == ["ASII"]
    assert "Unlock the trading lab" not in [b.label for b in app.button]


@pytest.mark.parametrize("tier", ["free", "logged_out"])
def test_today_free_shows_one_card_and_names_only(run_page, tier):
    app = run_page(tier)

    assert not app.exception
    buttons = _evidence_buttons(app)
    assert len(buttons) == 1 and buttons[0].key == "evidence_BBRI.JK"  # best holdout first
    text = _texts(app)
    assert "Also open: TLKM" in text
    assert "ASII" in text
    assert not app.dataframe
    assert "Unlock the trading lab" in [b.label for b in app.button]
    if tier == "logged_out":
        assert _metric(app, "Your positions") == "–"


def test_today_marks_estimated_cum_dates(run_page):
    app = run_page("premium")

    text = _texts(app)
    assert "Cum date **~" in text  # TLKM is estimated
    assert text.count("Cum date **~") == 1


def test_today_without_store_shows_a_warning(run_page):
    broken = MagicMock()
    broken.table.side_effect = RuntimeError("store down")
    app = run_page("premium", connection=broken)

    assert not app.exception
    assert "Signals are unavailable right now." in _texts(app)


def test_see_evidence_opens_that_stock(run_page):
    app = run_page("premium")
    button = next(b for b in _evidence_buttons(app) if b.key == "evidence_TLKM.JK")

    app = button.click().run()

    assert not app.exception
    assert app.segmented_control(key="view").value == "Evidence"
    assert app.selectbox(key="stock").value == "TLKM.JK"
    assert "Performance Outline" in _texts(app)


def test_unlock_prompt_opens_the_paywall(run_page):
    app = run_page("free")

    app = next(b for b in app.button if b.label == "Unlock the trading lab").click().run()

    assert app.segmented_control(key="view").value == "Lab"
    assert "Dividend Stock Shortlist" not in _texts(app)


# ── Evidence ────────────────────────────────────────────────────────────── #

def test_evidence_premium_runs_the_backtest_for_any_stock(run_page):
    app = run_page("premium", view="Evidence", stock="TLKM.JK")

    assert not app.exception
    assert app.selectbox(key="stock").value == "TLKM.JK"
    text = _texts(app)
    assert "Performance Outline" in text
    assert "Shortlist-wide setup. Holdout 30 of 40 trades won" in text
    assert "Unlock the trading lab" not in text


def test_stock_link_opens_evidence(run_page):
    app = run_page("premium", stock="BBRI.JK")

    assert app.segmented_control(key="view").value == "Evidence"
    assert "Performance Outline" in _texts(app)


@pytest.mark.parametrize("tier", ["free", "logged_out"])
def test_evidence_free_allows_only_the_sample_stock(run_page, tier):
    sample = run_page(tier, view="Evidence")
    assert not sample.exception
    assert sample.selectbox(key="stock").value == "BBRI.JK"
    assert "Performance Outline" in _texts(sample)

    other = run_page(tier, view="Evidence", stock="TLKM.JK")
    assert not other.exception
    text = _texts(other)
    assert "Unlock the trading lab" in text
    assert "Performance Outline" not in text


# ── Lab ─────────────────────────────────────────────────────────────────── #

def test_lab_premium_shows_the_research_tools(run_page):
    app = run_page("premium", view="Lab")

    assert not app.exception
    text = _texts(app)
    assert "Results here do not change the published signals" in text
    assert "Dividend Stock Shortlist" in text


@pytest.mark.parametrize("tier", ["free", "logged_out"])
def test_lab_is_gated(run_page, tier):
    app = run_page(tier, view="Lab")

    assert not app.exception
    text = _texts(app)
    assert "Unlock the trading lab" in text
    assert "Dividend Stock Shortlist" not in text
    labels = [b.label for b in app.button]
    assert ("Log in with Google" in labels) == (tier == "logged_out")
