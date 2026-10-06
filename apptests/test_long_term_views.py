"""Today, Evidence and Lab views of the Long Term page, per access tier."""
import html
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import harvest.data as hd
import harvest.position as pos
import harvest.premium as premium
import harvest.trading_store as store
from pipeline.trading_scan import long_term_signal
from tests.fake_supabase import FakeSupabase
from tests.test_position import make_financials, make_prices, make_statements


REPO = Path(__file__).resolve().parents[1]
PAGE = str(REPO / "apps" / "trading" / "backtester.py")
AS_OF = pd.Timestamp.today().normalize()
INDUSTRIES = {"BBRI.JK": "Banks - Regional", "TLKM.JK": "Telecommunications Services", "ASII.JK": "Conglomerates"}
SECTORS = {"BBRI.JK": "Financial Services", "TLKM.JK": "Communication Services", "ASII.JK": "Industrials"}
RECORD = {
    "method": "yearly_rebalance", "years": 2, "top_n": 10, "start": "2024-04-01", "end": "2026-04-01",
    "cagr": 12.3, "benchmark_cagr": 4.1, "beat_years": 2, "costs_pct": 0.4,
    "per_year": [
        {"start": "2024-04-01", "end": "2025-04-01", "complete": True, "analyzed": 150, "passed": 12,
         "holdings": {"ASII.JK": 15.0, "TLKM.JK": 9.0}, "portfolio": 12.0, "benchmark": 5.0},
        {"start": "2025-04-01", "end": "2026-04-01", "complete": True, "analyzed": 160, "passed": 0,
         "holdings": {}, "portfolio": 12.6, "benchmark": 3.2},
    ],
}


def _profiles() -> pd.DataFrame:
    symbols = list(INDUSTRIES)
    return pd.DataFrame({
        "price": 100.0, "mktCap": 100.0 * 100, "companyName": [f"PT {s[:4]} Tbk" for s in symbols],
        "sector": [SECTORS[s] for s in symbols], "industry": [INDUSTRIES[s] for s in symbols],
        "isActivelyTrading": True, "isEtf": False, "isFund": False,
    }, index=pd.Index(symbols, name="symbol"))


def _prices(*args, **kwargs) -> pd.DataFrame:
    return make_prices(end=AS_OF.strftime("%Y-%m-%d"), price=100.0)


def _statements(symbol=None) -> pd.DataFrame:
    return pos.prepare_statements(*make_statements(ocf=200.0))


def _signals() -> list[dict]:
    rows = [
        pos.analyze_frames(symbol, dict(profile), _prices(), make_financials(), 5, statements=_statements())
        for symbol, profile in _profiles().iterrows()
    ]
    screened = pos.ranked(pos.apply_screen(pos.score_results(rows), pos.PUBLISHED_SCREEN))
    return [long_term_signal(row, int(rank), 100.0, AS_OF, RECORD) for rank, row in screened.iterrows()]


def _client(published=True) -> FakeSupabase:
    fake = FakeSupabase()
    if published:
        run_id = store.start_run(fake, "long_term", "scan", AS_OF)
        store.finish_run(fake, run_id, "ok", {"requested": 200, "analyzed": 187})
        store.write_setups(fake, "long_term", [{"scope": "global", "params": pos.PUBLISHED_SCREEN, "holdout": RECORD}])
        store.write_signals(fake, "long_term", AS_OF, _signals())
    fake.auth = MagicMock()
    return fake


@pytest.fixture
def run_page(monkeypatch):
    import streamlit as st

    monkeypatch.chdir(REPO)
    active = []

    def run(tier="premium", view=None, stock=None, client=None):
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
        active.extend([
            patch.object(premium, "current_email", return_value=email),
            patch.object(premium, "is_premium", side_effect=lambda e: e == "fan@example.com"),
            patch.object(premium, "get_connection", return_value=client),
            patch("streamlit.connection", return_value=client),
            patch.object(pos, "load_idx_universe", side_effect=_profiles),
            patch.object(pos, "load_statements", side_effect=_statements),
            patch.object(hd, "get_daily_stock_price", side_effect=_prices),
            patch.object(hd, "get_financial_data", side_effect=lambda *a, **k: make_financials()),
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


def _evidence_buttons(app):
    return [b.key for b in app.button if b.label == "See evidence"]


def test_signals_fixture_has_three_ranked_candidates():
    signals = _signals()

    assert [s["detail"]["rank"] for s in signals] == [1, 2, 3]
    assert {s["detail"]["basis"] for s in signals} == {"P/S", "P/B"}


def test_today_premium_lists_candidates_with_a_card(run_page):
    app = run_page("premium")

    assert not app.exception
    assert "<h1>Growth at a Discount</h1>" in _hero(app)
    text = _texts(app)
    assert "the top 10 each April returned +12.3% a year against +4.1% for the JKSE, ahead in 2 of 2 years" in text
    assert "3 of 3 candidates" in text
    assert [m.label for m in app.metric][:2] == ["Candidates", "Companies analyzed"]
    assert app.metric[1].value == "187 / 200"
    assert len(_evidence_buttons(app)) == 1
    assert "Track this position" in [b.label for b in app.button]
    assert "Back at its mean valuation" in text
    assert "Companies that have since delisted are missing" in text


def test_today_filters_narrow_the_table(run_page):
    app = run_page("premium")
    app.multiselect[0].select("Industrials").run()

    assert not app.exception
    assert "1 of 3 candidates" in _texts(app)
    assert _evidence_buttons(app) == ["evidence_ASII.JK"]


@pytest.mark.parametrize("tier", ["free", "logged_out"])
def test_today_free_shows_the_top_candidate(run_page, tier):
    app = run_page(tier)

    assert not app.exception
    assert len(_evidence_buttons(app)) == 1
    assert "Also today:" in _texts(app)
    assert "Unlock the trading lab" in [b.label for b in app.button]
    assert "Track this position" not in [b.label for b in app.button]


def test_today_before_the_first_scan(run_page):
    app = run_page("premium", client=_client(published=False))

    assert not app.exception
    assert "Candidates have not been published yet" in _texts(app)


def test_see_evidence_opens_the_company(run_page):
    app = run_page("premium")
    key = _evidence_buttons(app)[0]
    app.button(key=key).click().run()

    assert not app.exception
    assert app.selectbox(key="stock").value == key.removeprefix("evidence_")
    assert "P/E vs median" in [m.label for m in app.metric]


def test_evidence_values_banks_on_book_value(run_page):
    app = run_page("premium", view="Evidence", stock="BBRI.JK")

    assert not app.exception
    labels = [m.label for m in app.metric]
    assert "P/B vs median" in labels and "P/S vs median" not in labels
    text = _texts(app)
    assert "valued on book value as a bank" in text
    assert "in the latest screen" in text


@pytest.mark.parametrize("tier", ["free", "logged_out"])
def test_evidence_free_only_for_the_sample(run_page, tier):
    sample = run_page(tier, view="Evidence")
    assert not sample.exception
    assert "P/E vs median" in [m.label for m in sample.metric]

    other = run_page(tier, view="Evidence", stock="TLKM.JK")
    assert "Unlock the trading lab" in _texts(other) and "P/E vs median" not in [m.label for m in other.metric]


def test_lab_is_gated_and_offers_quality_checks(run_page):
    lab = run_page("premium", view="Lab")
    assert not lab.exception
    text = _texts(lab)
    assert "Results here do not change the published candidates" in text
    assert "Require quality checks" in [c.label for c in lab.checkbox]

    assert "Unlock the trading lab" in _texts(run_page("free", view="Lab"))
