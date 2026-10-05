"""Tests for the Long Term engine in harvest/position.py."""
import numpy as np
import pandas as pd
import pytest

import harvest.data as hd
import harvest.position as pos


def make_prices(start="2019-01-01", end="2026-09-30", price=1000.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    # Newest first, like the FMP endpoint the page reads.
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": price}).iloc[::-1].reset_index(drop=True)


def make_financials(quarters=28, revenue=1000.0, margin=0.1, growth=0.10, end="2026-06-30") -> pd.DataFrame:
    dates = pd.date_range(end=end, periods=quarters, freq="QE")
    quarterly_growth = (1 + growth) ** 0.25
    revenues = revenue * quarterly_growth ** np.arange(quarters)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "revenue": revenues,
        "netIncome": revenues * margin,
        "reportedCurrency": "IDR",
    }).iloc[::-1].reset_index(drop=True)


PROFILE = {"price": 1000.0, "mktCap": 1000.0 * 100, "companyName": "PT Contoh Tbk", "sector": "Industrials"}


def test_analyze_frames_measures_multiples_and_growth():
    row = pos.analyze_frames("TEST.JK", PROFILE, make_prices(), make_financials(), 5)

    shares = 100
    ttm = make_financials().head(4)
    eps = ttm["netIncome"].sum() / shares
    sales = ttm["revenue"].sum() / shares
    assert row["Current PE"] == pytest.approx(1000 / eps)
    assert row["Current PS"] == pytest.approx(1000 / sales)
    assert row["Revenue Growth TTM"] == pytest.approx(10.0)
    assert row["Earnings Growth 5Y"] == pytest.approx(10.0)
    assert row["Margin TTM"] == pytest.approx(10.0)
    # Earnings grew while the price stayed flat, so today's P/E is the cheapest in the window.
    assert row["PE Discount"] > 0
    assert row["Statement Date"] == pd.Timestamp("2026-06-30")
    assert row["Company"] == "PT Contoh Tbk"


def test_analyze_frames_returns_histories_on_request():
    detail = pos.analyze_frames("TEST.JK", PROFILE, make_prices(), make_financials(), 5, include_history=True)

    assert set(detail) == {"row", "pe_history", "ps_history", "financial_history"}
    assert detail["pe_history"]["date"].is_monotonic_increasing
    assert "Revenue TTM" in detail["financial_history"]


@pytest.mark.parametrize("profile, prices, financials, reason", [
    ({**PROFILE, "price": 0}, make_prices(), make_financials(), "no price"),
    (PROFILE, pd.DataFrame(), make_financials(), "no prices"),
    (PROFILE, make_prices(), make_financials(quarters=7), "fewer than 8"),
    (PROFILE, make_prices(), None, "fewer than 8"),
    (PROFILE, make_prices(start="2026-08-01"), make_financials(), "valuation history"),
])
def test_analyze_frames_skips_stocks_without_enough_data(profile, prices, financials, reason):
    with pytest.raises(pos.SkipStock, match=reason):
        pos.analyze_frames("TEST.JK", profile, prices, financials, 5)


def test_analyze_stock_returns_none_on_any_failure(monkeypatch):
    def broken(*args, **kwargs):
        raise ConnectionError("FMP down")

    monkeypatch.setattr(hd, "get_daily_stock_price", broken)
    monkeypatch.setattr(hd, "get_financial_data", lambda *a, **k: make_financials())
    assert pos.analyze_stock("TEST.JK", PROFILE, 5) is None


def test_analyze_stock_fetches_prices_from_the_window_start(monkeypatch):
    calls = {}

    def prices(symbol, start_from=None):
        calls["start"] = start_from
        return make_prices()

    monkeypatch.setattr(hd, "get_daily_stock_price", prices)
    monkeypatch.setattr(hd, "get_financial_data", lambda *a, **k: make_financials())
    assert pos.analyze_stock("TEST.JK", PROFILE, 5)["Symbol"] == "TEST.JK"
    assert calls["start"] == pos.price_start(5)


def test_project_upside_caps_growth():
    result = pos.project_upside(10, 12, 2, 2, earnings_growth=80, revenue_growth=-50, years=1)

    assert result["earnings_growth"] == pytest.approx(pos.MAX_PROJECTION_GROWTH)
    assert result["revenue_growth"] == pytest.approx(pos.MIN_PROJECTION_GROWTH)
    assert result["earnings"] == pytest.approx((1.2 * 1.3 - 1) * 100)
    assert result["revenue"] == pytest.approx((0.8 - 1) * 100)
    assert result["blended"] == pytest.approx((result["earnings"] + result["revenue"]) / 2)


def _row(symbol, pe_pct, ps_pct, growth, margin):
    return {
        "Symbol": symbol, "PE Percentile": pe_pct, "PS Percentile": ps_pct,
        "Revenue Growth 5Y": growth, "Revenue Growth TTM": growth,
        "Earnings Growth 5Y": growth, "Earnings Growth TTM": growth,
        "Margin TTM": margin, "Statement Date": pd.Timestamp("2026-06-30"),
    }


def test_score_results_weights_valuation_growth_and_quality():
    result = pos.score_results([
        _row("CHEAP.JK", 10, 10, 1, 5),
        _row("GROWTH.JK", 90, 90, 20, 30),
    ])

    cheap = result.set_index("Symbol").loc["CHEAP.JK"]
    assert cheap["Valuation Score"] == pytest.approx(90)
    assert cheap["Growth Score"] == pytest.approx(50)
    assert cheap["Quality Score"] == pytest.approx(50)
    assert cheap["Score"] == pytest.approx(90 * 0.55 + 50 * 0.35 + 50 * 0.10)
    assert result["Symbol"].tolist() == ["CHEAP.JK", "GROWTH.JK"]
    assert result["Statement Date"].iloc[0] == pd.Timestamp("2026-06-30").date()


def test_score_results_without_rows_has_the_result_columns():
    assert list(pos.score_results([]).columns) == pos.RESULT_COLUMNS


def _screen_frame(**overrides):
    row = {
        "Earnings TTM": 10.0, "Revenue TTM": 100.0, "Current PE": 8.0, "Current PS": 1.0,
        "Revenue Growth 5Y": 5.0, "Earnings Growth 5Y": 5.0, "Revenue Growth TTM": 4.0,
        "Earnings Growth TTM": 4.0, "PE Discount": 10.0, "PS Discount": 10.0, "Score": 70.0,
    }
    return pd.DataFrame([{**row, **overrides}])


def test_apply_screen_passes_a_cheap_growing_company():
    screened = pos.apply_screen(_screen_frame(), pos.DEFAULT_FILTERS)

    assert bool(screened["Passes All"].iloc[0])
    assert screened["Failed Rules"].iloc[0] == ""


def test_apply_screen_lists_every_failed_rule():
    screened = pos.apply_screen(
        _screen_frame(**{"Current PE": 25.0, "Earnings Growth TTM": -3.0, "PS Discount": -5.0}),
        pos.DEFAULT_FILTERS,
    )

    assert not screened["Passes All"].iloc[0]
    assert screened["Failed Rules"].iloc[0] == "P/E > 20, Earnings TTM falling, P/S above median"


def test_apply_screen_drops_optional_rules_when_turned_off():
    filters = {**pos.DEFAULT_FILTERS, "require_recent_growth": False, "require_both_discounts": False}
    screened = pos.apply_screen(_screen_frame(**{"Earnings Growth TTM": -3.0, "PS Discount": -5.0}), filters)

    assert bool(screened["Passes All"].iloc[0])


def test_ranked_numbers_rows_from_one_by_score():
    table = pos.ranked(pd.DataFrame({"Symbol": ["A", "B"], "Score": [40.0, 80.0]}))

    assert table.index.tolist() == [1, 2]
    assert table.index.name == "Rank"
    assert table["Symbol"].tolist() == ["B", "A"]


def test_filter_universe_keeps_active_operating_companies_by_size():
    profiles = pd.DataFrame(
        {
            "mktCap": [5e12, 9e12, 7e12, 3e12, 0],
            "price": [100, 200, 300, 400, 500],
            "isActivelyTrading": [True, True, True, False, True],
            "isEtf": [False, False, True, False, False],
            "isFund": [False, False, False, False, False],
        },
        index=["A.JK", "B.JK", "ETF.JK", "DELISTED.JK", "ZERO.JK"],
    )

    universe = pos.filter_universe(profiles)

    assert universe.index.tolist() == ["B.JK", "A.JK"]
    assert pos.select_universe(universe, 6.0, 10).index.tolist() == ["B.JK"]
    assert pos.select_universe(universe, 1.0, 1).index.tolist() == ["B.JK"]


def test_profile_snapshot_is_stable_for_the_same_profiles():
    profiles = pd.DataFrame([PROFILE], index=["TEST.JK"])

    version, records = pos.profile_snapshot(profiles)

    assert version == pos.profile_snapshot(profiles.copy())[0]
    assert records[0][0] == "TEST.JK"
    assert dict(records[0][1])["companyName"] == "PT Contoh Tbk"


def test_checklist_marks_failures_for_review():
    row = pos.analyze_frames("TEST.JK", PROFILE, make_prices(), make_financials(), 5)
    checks = pos.checklist({**row, "Earnings Growth TTM": -1.0})

    status = checks.set_index("Test")["Status"]
    assert status["Recent earnings growth"] == "Review"
    assert status["Profitable"] == "Pass"

