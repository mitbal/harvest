"""Today, Evidence and Lab views of the Swing Trading page, per access tier."""
import html
import pickle
import re
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
import harvest.universe as universe
from tests.fake_supabase import FakeSupabase


REPO = Path(__file__).resolve().parents[1]
PAGE = str(REPO / "apps" / "trading" / "swing_trading.py")
AS_OF = pd.Timestamp.today().normalize()
SYMBOLS = ["BBRI.JK", "TLKM.JK", "ASII.JK", "GOTO.JK"]
SETUP = {"strategy_type": "Regime RSI", "param_a": 14.0, "param_b": 30.0, "param_c": 50.0, "param_d": 45.0,
         "max_hold": 5, "profit_target": 5.0, "stop_loss": 2.0}
WALK_FORWARD = {"level": "global", "method": "walk_forward", "trades": 80, "win_rate": 55.0, "avg_trade": 0.6,
                "profit_factor": 1.4, "t_stat": 2.3, "folds": []}


def _signal(symbol, status, avg_return):
    actionable = status == "actionable"
    return {
        "symbol": symbol, "status": status,
        "entry_price": 1000.0 if actionable else None,
        "stop_price": 980.0 if actionable else None, "target_price": 1050.0 if actionable else None,
        "exit_by": AS_OF + pd.Timedelta(days=9) if actionable else None,
        "key_date": AS_OF + pd.Timedelta(days=1) if actionable else None,
        "key_date_estimated": False, "setup_scope": "global",
        "detail": {
            "company": f"PT {symbol[:4]} Tbk", "setup": "Regime RSI: RSI Period: 14", "params": SETUP,
            "stop_pct": 2.0, "target_pct": 5.0, "max_hold": 5,
            "price_limits": {"floor": 850.0, "ceiling": 1250.0}, "stop_below_floor": False,
            "holdout": {"level": "global", "method": "walk_forward", "trades": 80, "wins": 44,
                        "win_rate": 55.0, "avg_return": avg_return},
        },
    }


def _client(with_setup=True):
    fake = FakeSupabase()
    for kind in ("optimize", "scan"):
        run_id = store.start_run(fake, "swing", kind, AS_OF)
        store.finish_run(fake, run_id, "ok", {
            "combinations": 22, "shortlist": 167,
            "walk_forward": {"trades": 0} if not with_setup else WALK_FORWARD,
        })
    if with_setup:
        store.write_setups(fake, "swing", [{"scope": "global", "params": SETUP, "holdout": WALK_FORWARD}])
        store.write_signals(fake, "swing", AS_OF, [
            _signal("BBRI.JK", "actionable", 0.9), _signal("TLKM.JK", "actionable", 0.5),
            _signal("GOTO.JK", "opening_soon", 0.5),
        ])
    else:
        store.write_signals(fake, "swing", AS_OF, [])
    fake.auth = MagicMock()
    fake.client = types.SimpleNamespace(storage=types.SimpleNamespace(
        from_=lambda bucket: types.SimpleNamespace(download=lambda path: pickle.dumps({})),
    ))
    return fake


def _fake_prices(exchange, symbols=None, start=None, end=None, columns=("close",)):
    dates = pd.bdate_range(end=AS_OF, periods=600)
    wave = 1000 + 120 * np.sin(np.arange(len(dates)) / 9)
    frames = []
    for symbol in symbols or SYMBOLS:
        frame = pd.DataFrame({"symbol": symbol, "date": dates, "close": wave})
        frame["open"], frame["high"], frame["low"], frame["volume"] = wave, wave * 1.01, wave * 0.99, 10_000_000
        frames.append(frame[["symbol", "date", *columns]])
    return pd.concat(frames, ignore_index=True)


def _fake_universe(exchange="jkse"):
    return pd.DataFrame({"stock": SYMBOLS, "companyName": [f"PT {s[:4]} Tbk" for s in SYMBOLS],
                         "mktCap": 50e12, "price": 1000.0, "volAvg": 1e8})


@pytest.fixture
def run_page(monkeypatch):
    import streamlit as st

    monkeypatch.chdir(REPO)
    monkeypatch.delenv("REDIS_URL", raising=False)
    active = []

    def run(tier="premium", view=None, stock=None, client=None, settings=None):
        for item in active:
            item.stop()
        active.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
        client = client or _client()
        email = {"premium": "fan@example.com", "free": "free@example.com", "logged_out": None}[tier]
        if tier == "premium":
            monkeypatch.delenv("PREMIUM_ENFORCED", raising=False)
        else:
            monkeypatch.setenv("PREMIUM_ENFORCED", "true")
        if settings and email:
            store.save_settings(client, email, settings)
        active.extend([
            patch.object(premium, "current_email", return_value=email),
            patch.object(premium, "is_premium", side_effect=lambda e: e == "fan@example.com"),
            patch.object(premium, "get_connection", return_value=client),
            patch("streamlit.connection", return_value=client),
            patch.object(hprices, "load_prices", side_effect=_fake_prices),
            patch.object(universe, "load_universe", side_effect=_fake_universe),
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


def _hero(app) -> str:
    return next(e.proto.body for e in app.get("html") if "hv-hero" in e.proto.body)


def _texts(app) -> str:
    parts = [e.value for e in app.markdown] + [e.value for e in app.caption] + [e.value for e in app.subheader]
    parts += [e.value for e in app.title] + [e.value for e in app.info] + [e.value for e in app.warning]
    # The trading cards are st.html; compare their visible text with tags and spacing collapsed.
    parts += [" ".join(html.unescape(re.sub(r"<style.*?</style>|<span class=\"icon\".*?</span>|<[^>]+>", " ", e.proto.body, flags=re.S)).split())
              for e in app.get("html")]
    return "\n".join(str(p) for p in parts)


def _cards(app):
    return [b for b in app.button if b.label == "See evidence"]


def test_today_premium_shows_cards_with_levels(run_page):
    app = run_page("premium")

    assert not app.exception
    assert "<h1>Swing Trading</h1>" in _hero(app)
    assert [b.key for b in _cards(app)] == ["evidence_BBRI.JK", "evidence_TLKM.JK"]
    text = _texts(app)
    assert "Stop −2% about Rp 980 Target +5% about Rp 1,050" in text
    assert "Next session limits Rp 850 to Rp 1,250" in text
    assert "walk-forward: 44 of 80 won, +0.90% average after costs" in text
    assert "Set your capital and risk per trade in My Trades" in text
    assert "GOTO" in text
    assert "No swing setup currently passes" not in text


def test_today_sizes_positions_from_settings(run_page):
    app = run_page("premium", settings={"capital_idr": 100_000_000, "risk_pct": 1.0})

    # Rp 1,000,000 at risk / (Rp 20 x 100 shares) = 500 lots.
    assert "Size 500 lots (about Rp 50,000,000), Rp 1,000,000 at risk" in _texts(app)


def test_today_explains_an_empty_day(run_page):
    app = run_page("premium", client=_client(with_setup=False))

    assert not app.exception
    text = _texts(app)
    assert "No swing setup currently passes the walk-forward test" in text
    assert "none of 22 setups on 167 stocks made money after costs" in text
    assert not _cards(app)


@pytest.mark.parametrize("tier", ["free", "logged_out"])
def test_today_free_shows_one_card(run_page, tier):
    app = run_page(tier)

    assert not app.exception
    assert [b.key for b in _cards(app)] == ["evidence_BBRI.JK"]
    assert "Also today: TLKM" in _texts(app)
    assert "Track this trade" not in [b.label for b in app.button]


def test_evidence_runs_the_backtest(run_page):
    app = run_page("premium", view="Evidence", stock="TLKM.JK")

    assert not app.exception
    assert app.selectbox(key="stock").value == "TLKM.JK"
    text = _texts(app)
    assert "Performance outline" in text
    assert "Published shortlist-wide setup; walk-forward: 80 trades, +0.60% average after costs" in text


@pytest.mark.parametrize("tier", ["free", "logged_out"])
def test_evidence_free_only_for_the_sample(run_page, tier):
    assert "Performance outline" in _texts(run_page(tier, view="Evidence"))
    other = _texts(run_page(tier, view="Evidence", stock="TLKM.JK"))
    assert "Unlock the trading lab" in other and "Performance outline" not in other


def test_lab_is_gated_and_has_no_pe_filter(run_page):
    lab = run_page("premium", view="Lab")
    assert not lab.exception
    text = _texts(lab)
    assert "Results here do not change the published signals" in text and "Shortlist" in text
    labels = [n.label for n in lab.number_input]
    assert "Minimum PE" not in labels and "Maximum PE" not in labels

    assert "Unlock the trading lab" in _texts(run_page("free", view="Lab"))
