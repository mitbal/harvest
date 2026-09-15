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
        result = sc.direction_aware_percentile([1, 2, 2, 4], 2, "higher_better")
        assert result["percent"] == pytest.approx(25)
        assert result["tie_count"] == 2
        assert "tied with 1" in result["text"]

    def test_lower_better_percentile(self):
        result = sc.direction_aware_percentile([5, 10, 15, 20], 10, "lower_better")
        assert result["percent"] == pytest.approx(50)
        assert "Better than 50%" in result["text"]

    def test_neutral_percentile_avoids_directional_claim(self):
        result = sc.direction_aware_percentile([0.8, 1.0, 1.2], 1.0, "neutral")
        assert "Better" not in result["text"]
        assert "percentile" in result["text"]

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
