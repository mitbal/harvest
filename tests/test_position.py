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

    assert set(detail) == {"row", "pe_history", "ps_history", "pb_history", "financial_history"}
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
        "Value Basis": "P/S", "PB Discount": -5.0, "Debt to Equity": 0.5, "Interest Coverage": 8.0,
        "Cash Conversion": 110.0, "ROE": 15.0, "Share Change 1Y": 0.0, "Quality Data": True,
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
    filters = {
        **pos.DEFAULT_FILTERS, "require_recent_growth": False, "require_both_discounts": False,
        "require_quality": False,
    }
    screened = pos.apply_screen(
        _screen_frame(**{"Earnings Growth TTM": -3.0, "PS Discount": -5.0, "ROE": 2.0, "Quality Data": False}),
        filters,
    )

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



def make_statements(quarters=28, equity=1000.0, debt=500.0, interest=10.0, pretax=50.0,
                    ocf=40.0, net_income=40.0, shares=100.0, share_growth=0.0, end="2026-06-30"):
    """FMP-shaped income, balance sheet and cash flow statements, newest first."""
    dates = pd.date_range(end=end, periods=quarters, freq="QE").strftime("%Y-%m-%d")
    steps = np.arange(quarters)
    income = pd.DataFrame({
        "date": dates, "interestExpense": interest, "incomeBeforeTax": pretax,
        "weightedAverageShsOut": shares * (1 + share_growth) ** (steps / 4),
    })
    balance = pd.DataFrame({"date": dates, "totalDebt": debt, "totalStockholdersEquity": equity * 1.02 ** steps})
    cash = pd.DataFrame({"date": dates, "operatingCashFlow": ocf, "netIncome": net_income})
    return tuple(frame.iloc[::-1].reset_index(drop=True) for frame in (income, balance, cash))


def test_prepare_statements_merges_quarters_oldest_first():
    income, balance, cash = make_statements(quarters=6)
    frame = pos.prepare_statements(income, balance.iloc[1:], cash)

    assert frame["date"].is_monotonic_increasing
    assert len(frame) == 6
    assert "cashFlowNetIncome" in frame and "netIncome" not in frame
    assert pd.isna(frame["totalDebt"].iloc[-1])


def test_prepare_statements_tolerates_missing_statements():
    frame = pos.prepare_statements(None, pd.DataFrame(), None)

    assert frame.empty
    assert {"totalDebt", "operatingCashFlow", "weightedAverageShsOut"} <= set(frame.columns)


def test_load_statements_reads_the_three_fmp_statements(monkeypatch):
    income, balance, cash = make_statements(quarters=6)
    frames = {"income-statement": income, "balance-sheet-statement": balance, "cash-flow-statement": cash}
    monkeypatch.setattr(hd, "get_fmp_statement", lambda symbol, statement: frames[statement])

    assert len(pos.load_statements("TEST.JK")) == 6


def test_quality_metrics_from_statements():
    statements = pos.prepare_statements(*make_statements(debt=500.0, share_growth=0.08))
    equity = statements["totalStockholdersEquity"]

    metrics = pos.quality_metrics(statements, earnings_ttm=160.0)

    assert metrics["Debt to Equity"] == pytest.approx(500.0 / equity.iloc[-1])
    assert metrics["Interest Coverage"] == pytest.approx((200 + 40) / 40)
    assert metrics["Cash Conversion"] == pytest.approx(100.0)
    assert metrics["ROE"] == pytest.approx(160.0 / ((equity.iloc[-1] + equity.iloc[-5]) / 2) * 100)
    assert metrics["Share Change 1Y"] == pytest.approx(8.0)


def test_quality_metrics_edge_cases():
    no_debt = pos.prepare_statements(*make_statements(interest=0.0))
    assert pos.quality_metrics(no_debt, 160.0)["Interest Coverage"] == np.inf

    short = pos.prepare_statements(*make_statements(quarters=4))
    metrics = pos.quality_metrics(short, 160.0)
    assert np.isnan(metrics["Cash Conversion"]) and np.isnan(metrics["ROE"]) and np.isnan(metrics["Share Change 1Y"])

    assert all(np.isnan(value) for value in pos.quality_metrics(None, 160.0).values())


def test_has_quality_data_asks_banks_for_fewer_metrics():
    metrics = {"Debt to Equity": np.nan, "Interest Coverage": np.inf, "Cash Conversion": np.nan,
               "ROE": 12.0, "Share Change 1Y": 0.0}

    assert pos.has_quality_data(metrics, bank=True)
    assert not pos.has_quality_data(metrics, bank=False)


def test_analyze_frames_adds_quality_and_book_value():
    statements = pos.prepare_statements(*make_statements())
    row = pos.analyze_frames("TEST.JK", PROFILE, make_prices(), make_financials(), 5, statements=statements)

    assert row["Value Basis"] == "P/S"
    assert row["Quality Data"]
    assert row["Current PB"] == pytest.approx(1000 / (statements["totalStockholdersEquity"].iloc[-1] / 100))
    assert row["ROE"] > 0
    assert np.isnan(row["Book Upside 1Y"])


def test_analyze_frames_without_statements_has_no_quality_data():
    row = pos.analyze_frames("TEST.JK", PROFILE, make_prices(), make_financials(), 5)

    assert not row["Quality Data"]
    assert np.isnan(row["Current PB"])


BANK = {**PROFILE, "sector": "Financial Services", "industry": "Banks - Regional"}


def test_banks_are_valued_on_book_value():
    statements = pos.prepare_statements(*make_statements(share_growth=0.0))
    row = pos.analyze_frames("BANK.JK", BANK, make_prices(), make_financials(), 5, statements=statements)

    assert pos.is_bank(BANK) and not pos.is_bank(PROFILE)
    assert row["Value Basis"] == "P/B"
    # Equity per share grows 2% a quarter.
    assert row["Book Growth TTM"] == pytest.approx((1.02 ** 4 - 1) * 100)
    assert row["Book Growth Assumption 1Y"] == pytest.approx(row["Book Growth TTM"])
    pe_leg = row["PE Mean"] / row["Current PE"] * (1 + row["Earnings Growth Assumption 1Y"] / 100)
    pb_leg = row["PB Mean"] / row["Current PB"] * (1 + row["Book Growth TTM"] / 100)
    assert row["Potential Upside 1Y"] == pytest.approx(((pe_leg + pb_leg) / 2 - 1) * 100)
    assert np.isnan(row["Revenue Upside 1Y"])
    assert row["Quality Data"]


def test_banks_without_a_balance_sheet_are_skipped():
    with pytest.raises(pos.SkipStock, match="valuation history"):
        pos.analyze_frames("BANK.JK", BANK, make_prices(), make_financials(), 5)


def test_score_uses_p_b_percentile_for_banks():
    rows = [
        {**_row("BANK.JK", 50, 99, 5, 10), "Value Basis": "P/B", "PB Percentile": 10, "ROE": 15},
        {**_row("CORP.JK", 50, 10, 5, 10), "Value Basis": "P/S", "PB Percentile": 99, "ROE": 15},
    ]
    scores = pos.score_results(rows).set_index("Symbol")["Valuation Score"]

    assert scores["BANK.JK"] == pytest.approx(70)
    assert scores["CORP.JK"] == pytest.approx(70)


def test_quality_score_averages_margin_and_roe_ranks():
    rows = [
        {**_row("A.JK", 50, 50, 5, 30), "ROE": 5},
        {**_row("B.JK", 50, 50, 5, 10), "ROE": 25},
    ]
    scores = pos.score_results(rows).set_index("Symbol")["Quality Score"]

    assert scores["A.JK"] == pytest.approx(75)
    assert scores["B.JK"] == pytest.approx(75)


def test_quality_rules_name_each_weakness():
    screened = pos.apply_screen(
        _screen_frame(**{"Debt to Equity": 2.4, "Interest Coverage": 1.5, "Cash Conversion": 40.0,
                         "ROE": 4.0, "Share Change 1Y": 12.0}),
        pos.DEFAULT_FILTERS,
    )

    assert screened["Failed Rules"].iloc[0] == (
        "Debt/equity > 1.5, Interest cover < 3x, Cash flow < 80% of profit, ROE < 8%, Shares up > 5%"
    )


def test_missing_quality_data_is_reported_once():
    screened = pos.apply_screen(
        _screen_frame(**{"Quality Data": False, "Cash Conversion": np.nan}), pos.DEFAULT_FILTERS,
    )

    assert screened["Failed Rules"].iloc[0] == "Quality data missing"


def test_banks_skip_p_s_and_corporate_quality_rules():
    screened = pos.apply_screen(
        _screen_frame(**{"Value Basis": "P/B", "Current PS": np.nan, "PS Discount": np.nan, "PB Discount": 8.0,
                         "Debt to Equity": 6.0, "Interest Coverage": 1.1, "Cash Conversion": -50.0}),
        pos.DEFAULT_FILTERS,
    )
    assert bool(screened["Passes All"].iloc[0])

    expensive = pos.apply_screen(_screen_frame(**{"Value Basis": "P/B", "PB Discount": -3.0}), pos.DEFAULT_FILTERS)
    assert expensive["Failed Rules"].iloc[0] == "P/B above median"


def test_checklist_includes_quality_rows():
    statements = pos.prepare_statements(*make_statements(debt=3000.0, ocf=500.0))
    row = pos.analyze_frames("TEST.JK", PROFILE, make_prices(), make_financials(), 5, statements=statements)
    status = pos.checklist(row).set_index("Test")["Status"]

    assert status["Moderate debt"] == "Review"
    assert status["Profit backed by cash"] == "Pass"

    bank = pos.analyze_frames("BANK.JK", BANK, make_prices(), make_financials(), 5, statements=statements)
    tests = pos.checklist(bank)["Test"].tolist()
    assert "Book valuation" in tests and "Revenue valuation" not in tests and "Moderate debt" not in tests


def test_cash_conversion_can_use_the_income_statement_net_income():
    statements = pos.prepare_statements(*make_statements(ocf=40.0, net_income=1e9))
    net_income = statements[["date"]].assign(netIncome=50.0)

    metrics = pos.quality_metrics(statements, 160.0, net_income)

    assert metrics["Cash Conversion"] == pytest.approx(80.0)


def test_usd_statements_are_converted_to_idr(monkeypatch):
    monkeypatch.setattr(hd, "get_usd_idr_rate", lambda *a, **k: 16000.0)
    income, balance, cash = make_statements()
    balance = balance.assign(reportedCurrency="USD")
    statements = pos.prepare_statements(income, balance, cash)

    converted = pos.statements_in_idr(statements)

    assert statements["reportedCurrency"].iloc[0] == "USD"
    assert converted["totalStockholdersEquity"].iloc[-1] == pytest.approx(statements["totalStockholdersEquity"].iloc[-1] * 16000)
    assert converted["weightedAverageShsOut"].iloc[-1] == statements["weightedAverageShsOut"].iloc[-1]
    assert pos.statements_in_idr(pos.prepare_statements(*make_statements())) is not None


# ── Backtest ────────────────────────────────────────────────────────────── #

def test_statements_become_public_after_the_reporting_lag():
    dates = pd.Series(pd.to_datetime(["2025-12-31", "2026-03-31", "2026-06-30"]))

    assert pos.available_from(dates).dt.strftime("%Y-%m-%d").tolist() == ["2026-03-31", "2026-05-30", "2026-08-29"]


def test_point_in_time_cuts_unpublished_quarters():
    income, balance, cash = make_statements(quarters=8, end="2026-06-30")
    statements = pos.prepare_statements(income, balance, cash)

    visible = pos.point_in_time(statements, "2026-05-15")

    # Q1 2026 is public from 30 May, so the latest is the annual report, public from 31 Mar.
    assert visible["date"].max() == pd.Timestamp("2026-03-31")
    assert pos.point_in_time(None, "2026-05-15") is None


def test_rebalance_dates_are_the_first_april_session():
    sessions = pd.bdate_range("2016-01-01", "2018-06-30")

    dates = pos.rebalance_dates(sessions, 2016)

    # 1 April 2017 and 2018 fall on a weekend.
    assert [d.date().isoformat() for d in dates] == ["2016-04-01", "2017-04-03", "2018-04-02"]


def _inputs(symbols, end="2026-09-30"):
    income, balance, cash = make_statements(end="2026-06-30", ocf=200.0)
    statements = pos.prepare_statements(income, balance, cash)
    prices = make_prices(start="2015-01-01", end=end, price=100.0)
    prices = prices.assign(date=pd.to_datetime(prices["date"])).sort_values("date").reset_index(drop=True)
    return {
        symbol: {
            "prices": prices, "financials": make_financials(), "statements": statements,
            "profile": {"companyName": symbol, "sector": "Industrials", "industry": "Conglomerates"},
        }
        for symbol in symbols
    }


def test_screen_at_uses_only_public_statements_and_period_market_cap():
    screen = {**pos.DEFAULT_FILTERS, "min_market_cap": 1e-8}

    screened = pos.screen_at(_inputs(["A.JK"]), "2026-05-15", screen)

    row = screened.iloc[0]
    assert row["Statement Date"] == pd.Timestamp("2026-03-31").date()
    assert row["Market Cap"] == pytest.approx(100.0 * 100)
    assert bool(row["Passes All"])
    assert pos.screen_at(_inputs(["A.JK"]), "2026-05-15", {**screen, "min_market_cap": 1.0}).empty


def test_backtest_compounds_yearly_picks_against_the_index(monkeypatch):
    inputs = _inputs(["UP.JK", "FLAT.JK"])
    sessions = pd.bdate_range("2023-01-02", "2025-12-31")
    inputs["UP.JK"]["prices"] = pd.DataFrame({"date": sessions, "close": 100.0 * 1.2 ** ((sessions - sessions[0]).days / 365.25)})
    inputs["FLAT.JK"]["prices"] = pd.DataFrame({"date": sessions, "close": 100.0})
    benchmark = pd.Series(1000.0 * 1.05 ** ((sessions - sessions[0]).days / 365.25), index=sessions)
    picks = {pd.Timestamp("2023-04-03"): ["UP.JK", "FLAT.JK"], pd.Timestamp("2024-04-01"): [], pd.Timestamp("2025-04-01"): ["UP.JK"]}

    def fake_screen(inputs, as_of, screen):
        symbols = picks[pd.Timestamp(as_of)]
        return pd.DataFrame({"Symbol": symbols, "Passes All": True, "Score": 80.0})

    monkeypatch.setattr(pos, "screen_at", fake_screen)
    result = pos.backtest_screen(inputs, benchmark, pos.DEFAULT_FILTERS, start_year=2023)

    years = result["years"]
    assert years["start"].astype(str).tolist() == ["2023-04-03", "2024-04-01", "2025-04-01"]
    assert years["complete"].tolist() == [True, True, False]
    first = years.iloc[0]
    up = (1.2 ** ((pd.Timestamp("2024-04-01") - pd.Timestamp("2023-04-03")).days / 365.25) - 1) * 100
    assert first["portfolio"] == pytest.approx(((up - 0.4) + (0 - 0.4)) / 2)
    assert years.iloc[1]["portfolio"] == 0.0 and years.iloc[1]["holdings"] == {}
    summary = result["summary"]
    assert summary["years"] == 2 and summary["cash_years"] == 1
    assert summary["total_return"] == pytest.approx(first["portfolio"])
    assert summary["benchmark_cagr"] == pytest.approx(5.0, abs=0.1)
    assert summary["beat_years"] == int(first["excess"] > 0)


def test_backtest_on_synthetic_inputs_runs_end_to_end():
    screen = {**pos.DEFAULT_FILTERS, "min_market_cap": 1e-8}
    inputs = _inputs(["A.JK"])
    sessions = pd.DatetimeIndex(inputs["A.JK"]["prices"]["date"])
    benchmark = pd.Series(1000.0, index=sessions)

    result = pos.backtest_screen(inputs, benchmark, screen, start_year=2025)

    years = result["years"]
    assert years["start"].astype(str).tolist() == ["2025-04-01", "2026-04-01"]
    assert list(years.iloc[0]["holdings"]) == ["A.JK"]
    assert years.iloc[0]["portfolio"] == pytest.approx(-0.4)
    assert result["summary"]["years"] == 1
