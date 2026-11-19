"""
tests/apps/test_stock_comparison.py
Tests for get_processed_df, METRIC_OPTIONS, TABLE_DEFAULT_METRICS.
Stubs are installed by tests/apps/conftest.py.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# conftest.py already installed all stubs. Just import.
with patch.dict("os.environ", {"REDIS_URL": "redis://stub", "FMP_API_KEY": "stub"}):
    with patch("harvest.data.get_company_profile", return_value=pd.DataFrame(
        {"price": [1.0], "changes": [0.0], "beta": [1.0]},
        index=pd.Index(["STUB.JK"], name="symbol"),
    )):
        import apps.screener.stock_comparison as sc

# ─────────────────────────────────────────────────────────────────────────────

def _make_raw_df(n: int = 5):
    symbols = ["BBCA.JK", "TLKM.JK", "BMRI.JK", "ASII.JK", "EXCL.JK"][:n]
    return pd.DataFrame({
        "price":                   [5000, 3800, 9500, 6000, 4500][:n],
        "changes":                 [100, -50, 200, 0, -100][:n],
        "sector":                  ["Fin", "Tel", "Fin", "Auto", "Tel"][:n],
        "industry":                ["Banks", "Tel", "Banks", "Auto", "Tel"][:n],
        "mktCap":                  [5e14, 2e14, 8e14, 1e14, 3e14][:n],
        "ipoDate":                 ["2000-01-01"] * n,
        "yield":                   [2.5, 4.0, 3.0, 2.0, 3.5][:n],
        "lastDiv":                 [250, 200, 300, 150, 175][:n],
        "avgFlatAnnualDivIncrease":[10, 5, 15, 8, 6][:n],
        "numDividendYear":         [15, 10, 20, 8, 12][:n],
        "positiveYear":            [12, 8, 18, 6, 9][:n],
        "numOfYear":               [20, 15, 25, 12, 18][:n],
        "maximumCutPct":           [-20, -15, -10, -30, -25][:n],
        "max10CutPct":             [-10, -5, -5, -15, -12][:n],
        "peRatio":                 [15.0, 12.0, 18.0, 10.0, 14.0][:n],
        "psRatio":                 [3.0, 2.5, 4.0, 1.5, 2.0][:n],
        "revenueGrowth":           [10.0, 5.0, 12.0, 8.0, 6.0][:n],
        "netIncomeGrowth":         [12.0, 4.0, 14.0, 7.0, 5.0][:n],
        "medianProfitMargin":      [25.0, 20.0, 30.0, 15.0, 18.0][:n],
        "earningTTM":              [1e13, 8e12, 2e13, 5e12, 7e12][:n],
        "revenueTTM":              [4e13, 4e13, 6e13, 3e13, 4e13][:n],
        "revenueGrowthTTM":        [8.0, 3.0, 10.0, 5.0, 4.0][:n],
        "netIncomeGrowthTTM":      [10.0, 2.0, 12.0, 4.0, 3.0][:n],
        "beta":                    [0.8, 0.9, 0.7, 1.1, 1.0][:n],
        "return_7d":               [0.01, -0.02, 0.03, 0.005, -0.01][:n],
        "return_1m":               [0.03, -0.01, 0.05, 0.02, -0.02][:n],
        "return_1y":               [0.15, 0.08, 0.20, 0.10, 0.12][:n],
        "return_10y":              [2.0, 1.5, 3.0, 1.0, 1.8][:n],
        "total_return_1y":         [0.18, 0.10, 0.22, 0.12, 0.14][:n],
        "total_return_10y":        [2.5, 2.0, 3.5, 1.5, 2.2][:n],
        "is_syariah":              [False, False, False, True, False][:n],
    }, index=pd.Index(symbols, name="stock"))


# ══════════════════════════════════════════════════════════════════════════════

class TestGetProcessedDf:

    @pytest.fixture
    def raw(self):
        return _make_raw_df()

    def test_margin_ttm_column(self, raw):
        result = sc.get_processed_df(raw)
        assert "marginTTM" in result.columns
        for idx in result.index:
            expected = raw.loc[idx, "earningTTM"] / raw.loc[idx, "revenueTTM"] * 100
            assert abs(result.loc[idx, "marginTTM"] - expected) < 1e-6

    def test_dividend_payout_ratio_column(self, raw):
        result = sc.get_processed_df(raw)
        assert "dividendPayoutRatio" in result.columns
        pos_rows = result[result["earningTTM"] > 0]
        assert (pos_rows["dividendPayoutRatio"] > 0).all()

    def test_rank_column_ascending(self, raw):
        result = sc.get_processed_df(raw)
        assert "Rank" in result.columns
        assert list(result["Rank"]) == list(range(1, len(result) + 1))

    def test_returns_converted_to_percent(self, raw):
        result = sc.get_processed_df(raw)
        assert result.loc["BBCA.JK", "return_1y"] == pytest.approx(15.0, abs=0.01)

    def test_max_cut_pct_made_positive(self, raw):
        result = sc.get_processed_df(raw)
        assert result.loc["BBCA.JK", "maximumCutPct"] == pytest.approx(20.0, abs=0.01)

    def test_payout_ratio_nan_when_negative_earnings(self):
        df = _make_raw_df(n=1).copy()
        df["earningTTM"] = -1e12
        result = sc.get_processed_df(df)
        assert np.isnan(result["dividendPayoutRatio"].iloc[0])

    def test_missing_analytical_values_remain_missing(self, raw):
        raw.loc["BBCA.JK", "revenueGrowth"] = np.nan
        raw.loc["TLKM.JK", "beta"] = np.nan
        result = sc.get_processed_df(raw)
        assert np.isnan(result.loc["BBCA.JK", "revenueGrowth"])
        assert np.isnan(result.loc["TLKM.JK", "beta"])

    def test_payout_ratio_uses_snapshot_price_for_share_count(self):
        raw = _make_raw_df(n=1)
        # Market cap 5e14 at a snapshot price of 5000 -> 1e11 shares -> EPS 100.
        raw["snapshotPrice"] = 5000
        raw["price"] = 10000  # live price doubled since the snapshot
        result = sc.get_processed_df(raw)
        assert result.loc["BBCA.JK", "dividendPayoutRatio"] == pytest.approx(250.0)

    def test_payout_ratio_falls_back_to_live_price(self):
        raw = _make_raw_df(n=1)
        raw["snapshotPrice"] = np.nan
        result = sc.get_processed_df(raw)
        assert result.loc["BBCA.JK", "dividendPayoutRatio"] == pytest.approx(250.0)

    def test_zero_revenue_produces_missing_margin(self):
        raw = _make_raw_df(n=1)
        raw.loc["BBCA.JK", "revenueTTM"] = 0
        result = sc.get_processed_df(raw)
        assert np.isnan(result.loc["BBCA.JK", "marginTTM"])


class TestMetricOptions:

    def test_all_options_have_semantic_metadata(self):
        for spec in sc.METRIC_OPTIONS.values():
            assert spec.source
            assert spec.group
            assert spec.format
            assert spec.unit
            assert spec.description

    def test_direction_values_valid(self):
        valid = {"higher_better", "lower_better", "neutral"}
        for spec in sc.METRIC_OPTIONS.values():
            assert spec.direction in valid

    def test_format_strings_parseable(self):
        for label, spec in sc.METRIC_OPTIONS.items():
            try:
                spec.format.format(42.5)
            except (ValueError, KeyError) as e:
                pytest.fail(f"METRIC_OPTIONS['{label}'] fmt='{spec.format}' raised: {e}")

    def test_dividend_payout_ratio_present(self):
        assert "Dividend Payout Ratio (%)" in sc.METRIC_OPTIONS

    def test_dividend_payout_ratio_is_contextual(self):
        spec = sc.METRIC_OPTIONS["Dividend Payout Ratio (%)"]
        assert spec.direction == "neutral"

    def test_last_dividend_is_not_ranked(self):
        spec = sc.METRIC_OPTIONS["Last Dividend"]
        assert spec.direction == "neutral"
        assert sc.comparison_states({"A": 100, "B": 200}, spec) == {"A": "", "B": ""}

    @pytest.mark.parametrize("label", ["Div Growth (Annual)", "Revenue (TTM)", "Net Income (TTM)"])
    def test_size_driven_metrics_are_not_ranked(self, label):
        spec = sc.METRIC_OPTIONS[label]
        assert spec.direction == "neutral"
        assert sc.comparison_states({"A": 1e12, "B": 5e12}, spec) == {"A": "", "B": ""}
        assert label in sc.CHART_METRIC_OPTIONS

    def test_non_positive_valuation_is_invalid(self):
        pe = sc.METRIC_OPTIONS["PE Ratio"]
        ps = sc.METRIC_OPTIONS["PS Ratio"]
        assert not sc.is_valid_metric_value(0, pe)
        assert not sc.is_valid_metric_value(-4, pe)
        assert not sc.is_valid_metric_value(0, ps)
        assert sc.is_valid_metric_value(12.5, pe)

    def test_formatting_handles_missing_and_units(self):
        assert sc.format_metric_value("Dividend Yield (%)", np.nan) == "N/A"
        assert sc.format_metric_value("Dividend Yield (%)", 4.25) == "4.25%"
        assert sc.format_metric_value("Revenue (TTM)", 2e12, 1e12, "T IDR") == "2.00 T IDR"


class TestTableDefaultMetrics:

    def test_all_default_metrics_in_metric_options(self):
        for m in sc.TABLE_DEFAULT_METRICS:
            assert m in sc.METRIC_OPTIONS

    def test_minimum_count(self):
        assert len(sc.TABLE_DEFAULT_METRICS) >= 5


class TestComparisonHelpers:

    def test_clear_selection_resets_widget_and_query_state(self):
        sc.st.session_state['comp_stocks'] = ['BBCA.JK', 'TLKM.JK']
        sc.st.query_params['stocks'] = 'BBCA.JK,TLKM.JK'
        sc._clear_selection()
        assert sc.st.session_state['comp_stocks'] == []
        assert sc.st.session_state['_comp_last_qp_stocks'] == ''
        assert 'stocks' not in sc.st.query_params

    def test_higher_better_percentile_and_tie(self):
        result = sc.direction_aware_percentile([1, 2, 4], 2, "higher_better")
        assert result["percent"] == pytest.approx(100 / 3)
        assert result["tie_count"] == 1
        assert "tied with 1" in result["text"]
        untied = sc.direction_aware_percentile([1, 3, 4], 2, "higher_better")
        assert "tied" not in untied["text"]

    def test_percentile_excludes_the_stock_itself(self):
        peers = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "TOP": 9.0})
        result = sc.direction_aware_percentile(peers, 9.0, "higher_better", exclude="TOP")
        assert result["percent"] == pytest.approx(100)
        assert result["peer_count"] == 3
        assert result["tie_count"] == 0
        assert result["text"] == "Better than 100% of 3 valid peers"

    def test_percentile_exclude_missing_label_is_harmless(self):
        peers = pd.Series({"A": 1.0, "B": 2.0})
        result = sc.direction_aware_percentile(peers, 3.0, "higher_better", exclude="OUTSIDE")
        assert result["peer_count"] == 2

    def test_percentile_with_only_self_has_no_peers(self):
        peers = pd.Series({"ONLY": 1.0})
        result = sc.direction_aware_percentile(peers, 1.0, "higher_better", exclude="ONLY")
        assert result["percent"] is None
        assert result["text"] == "No valid peer comparison"

    def test_lower_better_percentile(self):
        result = sc.direction_aware_percentile([5, 10, 15, 20], 10, "lower_better")
        assert result["percent"] == pytest.approx(50)
        assert "Better than 50%" in result["text"]

    def test_neutral_percentile_avoids_directional_claim(self):
        result = sc.direction_aware_percentile([0.8, 1.2], 1.0, "neutral")
        assert "Better" not in result["text"]
        assert result["text"] == "50th percentile of 2 valid peers"

    @pytest.mark.parametrize(
        ("number", "expected"),
        [(0, "0th"), (1, "1st"), (2, "2nd"), (3, "3rd"), (11, "11th"), (12, "12th"), (13, "13th"), (21, "21st"), (100, "100th")],
    )
    def test_ordinal(self, number, expected):
        assert sc._ordinal(number) == expected

    def test_split_valid_size_drops_missing_and_invalid(self):
        df = pd.DataFrame({"stock": ["A", "B", "C", "D"], "mktCap": [1e12, np.nan, -5.0, np.inf]})
        kept, dropped = sc.split_valid_size(df, sc.METRIC_OPTIONS["Market Cap"])
        assert kept["stock"].tolist() == ["A"]
        assert dropped == ["B", "C", "D"]

    def test_split_valid_size_keeps_negative_values_unclipped(self):
        df = pd.DataFrame({"stock": ["A", "B"], "revenueGrowthTTM": [-12.5, 8.0]})
        kept, dropped = sc.split_valid_size(df, sc.METRIC_OPTIONS["Revenue Growth (TTM)"])
        assert dropped == []
        assert kept["revenueGrowthTTM"].tolist() == [-12.5, 8.0]

    def test_comparison_states_exclude_invalid_ratios(self):
        spec = sc.METRIC_OPTIONS["PE Ratio"]
        states = sc.comparison_states({"A": -5, "B": 10, "C": 20}, spec)
        assert states == {"A": "", "B": "Best", "C": "Weakest"}

    def test_comparison_states_handle_all_ties(self):
        spec = sc.METRIC_OPTIONS["Dividend Yield (%)"]
        states = sc.comparison_states({"A": 3.0, "B": 3.0}, spec)
        assert states == {"A": "Tied", "B": "Tied"}

    def test_comparison_cell_styles_are_color_coded(self):
        assert "22, 163, 74" in sc.comparison_cell_style("4.00% - Best")
        assert "220, 38, 38" in sc.comparison_cell_style("2.00% - Weakest")
        assert "100, 116, 139" in sc.comparison_cell_style("3.00% - Tied")
        assert sc.comparison_cell_style("N/A") == ""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("table", "table"), ("DIST", "dist"), ("unknown", "table"), (None, "table")],
    )
    def test_normalize_view(self, value, expected):
        assert sc.normalize_view(value) == expected

    def test_parse_stock_query_preserves_order_and_limits(self):
        stocks, limited = sc.parse_stock_query("C,A,X,B,D,E,F,A", ["A", "B", "C", "D", "E", "F"])
        assert stocks == ["C", "A", "B", "D", "E"]
        assert limited is True


class TestStockGroups:

    def test_filters_out_unknown_tickers(self):
        valid = ["BBCA.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ISAT.JK", "EXCL.JK"]
        groups = sc.available_stock_groups("JKSE", valid)
        assert groups["Big banks"] == ["BBCA.JK", "BMRI.JK", "BBNI.JK"]
        assert groups["Telco"] == ["TLKM.JK", "ISAT.JK", "EXCL.JK"]

    def test_drops_groups_with_fewer_than_two_valid_tickers(self):
        groups = sc.available_stock_groups("JKSE", ["BBCA.JK", "TLKM.JK", "HMSP.JK", "GGRM.JK"])
        assert list(groups) == ["Tobacco"]

    def test_unknown_market_returns_empty(self):
        assert sc.available_stock_groups("NASDAQ", ["AAPL", "MSFT"]) == {}

    def test_declared_groups_have_two_to_max_unique_tickers(self):
        for market, groups in sc.STOCK_GROUPS.items():
            for name, stocks in groups.items():
                assert 2 <= len(set(stocks)) <= sc.MAX_STOCKS, (market, name)
                assert len(set(stocks)) == len(stocks), (market, name)


def _prices(dates, closes):
    return pd.DataFrame({"date": dates, "close": closes, "adjClose": closes})


def _business_prices(start, periods, close=100.0):
    dates = pd.bdate_range(start, periods=periods)
    return _prices(dates.strftime("%Y-%m-%d"), [close] * periods)


class TestHistoryHelpers:

    def test_normalize_view_accepts_history(self):
        assert sc.normalize_view("history") == "history"
        assert sc.normalize_view("HISTORY") == "history"

    def test_build_performance_frame_handles_messy_prices(self):
        raw = pd.DataFrame({
            "date": ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02", "bad", "2024-01-04"],
            "close": [12, 10, "x", 11, 5, -3],
        })
        frame = sc.build_performance_frame(raw, None)
        assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-01-02", "2024-01-03"]
        assert frame["price"].tolist() == [10, 11, 12]
        assert frame["total"].tolist() == [10, 11, 12]

    @pytest.mark.parametrize("prices", [None, pd.DataFrame(), pd.DataFrame({"date": ["2024-01-01"]})])
    def test_build_performance_frame_empty_prices(self, prices):
        frame = sc.build_performance_frame(prices, None)
        assert frame.empty
        assert list(frame.columns) == ["date", "price", "total", "value"]

    def test_dividend_reinvested_on_next_trading_day(self):
        prices = _prices(["2024-01-01", "2024-01-03", "2024-01-04"], [100, 100, 110])
        # Ex-date 2024-01-02 has no price, so it is reinvested at the 2024-01-03 close.
        divs = pd.DataFrame({"date": ["2024-01-02"], "adjDividend": [10]})
        frame = sc.build_performance_frame(prices, divs, include_dividends=True)
        assert frame["total"].tolist() == pytest.approx([100, 110, 121])
        assert frame["value"].tolist() == pytest.approx([100, 110, 121])
        price_only = sc.build_performance_frame(prices, divs, include_dividends=False)
        assert price_only["value"].tolist() == pytest.approx([100, 100, 110])

    def test_dividends_outside_price_history_ignored(self):
        prices = _prices(["2024-01-01", "2024-01-02"], [100, 100])
        divs = pd.DataFrame({"date": ["2023-12-01", "2024-02-01"], "adjDividend": [5, 5]})
        frame = sc.build_performance_frame(prices, divs)
        assert frame["total"].tolist() == pytest.approx([100, 100])

    def test_rebase_to_common_start_date(self):
        old = sc.build_performance_frame(_prices(["2020-01-01", "2021-01-04", "2022-01-03"], [50, 100, 150]), None)
        new = sc.build_performance_frame(_prices(["2021-01-04", "2022-01-03"], [20, 10]), None)
        long_df, common_start, limiting = sc.rebase_performance({"OLD": old, "NEW": new}, pd.Timestamp("2019-01-01"))
        assert common_start == pd.Timestamp("2021-01-04")
        assert limiting == "NEW"
        starts = long_df.groupby("stock")["date"].min()
        assert (starts == common_start).all()
        assert long_df[long_df["date"] == common_start]["value"].tolist() == [100.0, 100.0]
        assert long_df[long_df["stock"] == "OLD"]["value"].tolist() == pytest.approx([100, 150])
        assert long_df[long_df["stock"] == "NEW"]["value"].tolist() == pytest.approx([100, 50])

    def test_rebase_uses_period_start_when_all_stocks_cover_it(self):
        frame = sc.build_performance_frame(_prices(["2020-01-01", "2020-06-01", "2021-01-01"], [100, 200, 300]), None)
        long_df, common_start, limiting = sc.rebase_performance({"A": frame, "B": frame}, pd.Timestamp("2020-03-01"))
        assert common_start == pd.Timestamp("2020-03-01")
        assert limiting is None
        # Base is the last close on or before the start (100), so 2020-06-01 is 200.
        assert long_df[long_df["stock"] == "A"]["value"].tolist() == pytest.approx([100, 200, 300])

    def test_rebase_empty_input(self):
        long_df, common_start, limiting = sc.rebase_performance({"A": sc.build_performance_frame(None, None)}, pd.Timestamp("2020-01-01"))
        assert long_df.empty
        assert common_start is None
        assert limiting is None

    def test_total_return_exceeds_price_return_with_dividend(self):
        prices = _business_prices("2022-01-03", 600)
        divs = pd.DataFrame({"date": ["2022-06-01"], "adjDividend": [5]})
        frame = sc.build_performance_frame(prices, divs)
        long_df, _, _ = sc.rebase_performance({"A": frame}, pd.Timestamp("2022-01-03"))
        summary = sc.performance_summary(long_df, ["A"])
        assert summary.loc["A", "Price return"] == pytest.approx(0)
        assert summary.loc["A", "Total return"] == pytest.approx(0.05)
        assert summary.loc["A", "Total return"] > summary.loc["A", "Price return"]
        assert np.isfinite(summary.loc["A", "Annualised total return"])

    def test_max_drawdown(self):
        frame = sc.build_performance_frame(
            _prices(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], [100, 200, 100, 150]), None
        )
        long_df, _, _ = sc.rebase_performance({"A": frame}, None)
        summary = sc.performance_summary(long_df, ["A"])
        assert summary.loc["A", "Max drawdown"] == pytest.approx(-0.5)
        assert summary.loc["A", "Price return"] == pytest.approx(0.5)

    def test_annualised_na_under_one_year(self):
        frame = sc.build_performance_frame(_business_prices("2024-01-01", 100), None)
        long_df, _, _ = sc.rebase_performance({"A": frame}, None)
        summary = sc.performance_summary(long_df, ["A"])
        assert np.isnan(summary.loc["A", "Annualised total return"])
        assert sc.format_signed_pct(summary.loc["A", "Annualised total return"]) == "N/A"

    def test_annualised_over_two_years(self):
        frame = sc.build_performance_frame(_prices(["2020-01-01", "2022-01-01"], [100, 121]), None)
        long_df, _, _ = sc.rebase_performance({"A": frame}, None)
        summary = sc.performance_summary(long_df, ["A"])
        assert summary.loc["A", "Annualised total return"] == pytest.approx(0.1, abs=1e-3)

    def test_summary_missing_stock_is_na(self):
        frame = sc.build_performance_frame(_prices(["2024-01-01", "2024-01-02"], [100, 110]), None)
        long_df, _, _ = sc.rebase_performance({"A": frame}, None)
        summary = sc.performance_summary(long_df, ["A", "MISSING"])
        assert summary.loc["MISSING"].isna().all()
        assert sc.performance_summary(pd.DataFrame(columns=["stock", "date", "price", "total", "value"]), ["A"]).loc["A"].isna().all()

    def test_format_signed_pct(self):
        assert sc.format_signed_pct(0.1234) == "+12.3%"
        assert sc.format_signed_pct(-0.05) == "-5.0%"
        assert sc.format_signed_pct(np.nan) == "N/A"
        assert sc.format_signed_pct(None) == "N/A"

    def test_thin_series_keeps_first_and_last(self):
        frame = sc.build_performance_frame(_business_prices("2020-01-01", 2000), None)
        long_df, _, _ = sc.rebase_performance({"A": frame}, None)
        thinned = sc.thin_series(long_df, max_points=100)
        assert len(thinned) <= 100
        assert thinned["date"].iloc[0] == long_df["date"].iloc[0]
        assert thinned["date"].iloc[-1] == long_df["date"].iloc[-1]


class TestYearlyDividends:

    TODAY = pd.Timestamp("2026-06-30")
    YEARS = list(range(2017, 2027))

    @pytest.fixture
    def divs(self):
        return pd.DataFrame({
            "date": ["2020-05-01", "2020-11-01", "2022-05-01", "2026-03-01", "2026-03-01", "2026-12-01"],
            "adjDividend": [10, 10, 30, 5, 5, 99],
        })

    @pytest.fixture
    def prices(self):
        return _prices(
            ["2021-12-30", "2021-12-31", "2022-12-30", "2023-06-30", "2026-06-29", "2026-06-30"],
            [190, 200, 300, 250, 400, 500],
        )

    def test_per_share_zero_fills_from_first_year(self, divs, prices):
        result = sc.yearly_dividends(divs, prices, self.YEARS, "Per share", today=self.TODAY)
        values = dict(zip(result["year"], result["value"]))
        assert list(values) == list(range(2020, 2027))
        assert values[2020] == pytest.approx(20)
        assert values[2021] == 0
        assert values[2022] == pytest.approx(30)
        # Exact duplicate row is dropped and the December ex-date is in the future.
        assert values[2026] == pytest.approx(5)

    def test_current_year_flagged_as_ytd(self, divs, prices):
        result = sc.yearly_dividends(divs, prices, self.YEARS, "Per share", today=self.TODAY)
        assert result.loc[result["is_ytd"], "year"].tolist() == [2026]

    def test_growth_index_base_year_is_first_positive_in_window(self, divs, prices):
        result = sc.yearly_dividends(divs, prices, range(2021, 2027), "Growth index", today=self.TODAY)
        values = dict(zip(result["year"], result["value"]))
        # 2021 paid nothing, so the base is 2022 (30) and 2021 is not shown.
        assert list(values) == [2022, 2023, 2024, 2025, 2026]
        assert values[2022] == pytest.approx(100)
        assert values[2023] == 0
        assert values[2026] == pytest.approx(5 / 30 * 100)

    def test_yield_uses_year_end_close_and_latest_close(self, divs, prices):
        result = sc.yearly_dividends(divs, prices, self.YEARS, "Yield at year-end price", today=self.TODAY)
        values = dict(zip(result["year"], result["value"]))
        # 2020 has no price history, so it is absent rather than zero.
        assert 2020 not in values
        assert values[2021] == 0
        assert values[2022] == pytest.approx(30 / 300 * 100)
        assert values[2023] == 0
        assert 2024 not in values
        assert values[2026] == pytest.approx(5 / 500 * 100)

    @pytest.mark.parametrize("raw_divs", [None, pd.DataFrame(), pd.DataFrame({"date": ["2020-01-01"], "adjDividend": ["x"]})])
    @pytest.mark.parametrize("measure", list(sc._DIV_MEASURES))
    def test_no_dividends_returns_empty(self, raw_divs, measure):
        result = sc.yearly_dividends(raw_divs, _prices(["2020-12-31"], [100]), self.YEARS, measure, today=self.TODAY)
        assert result.empty
        assert list(result.columns) == ["year", "value", "is_ytd"]

    def test_yield_without_prices_is_empty(self, divs):
        result = sc.yearly_dividends(divs, pd.DataFrame(), self.YEARS, "Yield at year-end price", today=self.TODAY)
        assert result.empty

    def test_growth_index_all_zero_in_window_is_empty(self):
        divs = pd.DataFrame({"date": ["2015-05-01"], "adjDividend": [10]})
        result = sc.yearly_dividends(divs, None, self.YEARS, "Growth index", today=self.TODAY)
        assert result.empty

    def test_unknown_measure_raises(self):
        with pytest.raises(ValueError):
            sc.yearly_dividends(None, None, self.YEARS, "Bogus", today=self.TODAY)
