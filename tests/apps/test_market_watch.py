"""
tests/apps/test_market_watch.py
Tests for calc_daily_return_for_date and module constants.
Stubs are installed by tests/apps/conftest.py.
"""
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


# conftest.py already installed all stubs.
with patch.dict("os.environ", {
    "REDIS_URL": "redis://stub",
    "FMP_API_KEY": "stub",
    "SUPABASE_URL": "https://stub",
    "SUPABASE_KEY": "stub",
}):
    import apps.screener.market_watch as mw


# ─────────────────────────────────────────────────────────────────────────────

def _make_prices_df():
    return pd.DataFrame({
        "symbol": ["A", "A", "A", "B", "B", "B"],
        "date":   pd.to_datetime([
            "2023-06-01", "2023-06-02", "2023-06-05",
            "2023-06-01", "2023-06-02", "2023-06-05",
        ]),
        "close":  [100.0, 105.0, 110.0, 200.0, 196.0, 202.0],
    })


# ══════════════════════════════════════════════════════════════════════════════

class TestCalcDailyReturnForDate:

    def test_normal_return_calculation(self):
        df = _make_prices_df()
        result = mw.calc_daily_return_for_date(df, pd.Timestamp("2023-06-02"))
        assert "return_1d_pct" in result.columns
        assert abs(result.loc["A", "return_1d_pct"] - 5.0) < 1e-6
        assert result.loc["B", "return_1d_pct"] < 0

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame(columns=["symbol", "date", "close"])
        result = mw.calc_daily_return_for_date(empty, pd.Timestamp("2023-06-02"))
        assert result.empty

    def test_no_previous_date_returns_empty(self):
        df = pd.DataFrame({
            "symbol": ["A"],
            "date":   pd.to_datetime(["2023-06-01"]),
            "close":  [100.0],
        })
        result = mw.calc_daily_return_for_date(df, pd.Timestamp("2023-06-01"))
        assert result.empty

    def test_only_common_symbols_returned(self):
        df = pd.DataFrame({
            "symbol": ["A", "A", "B"],
            "date":   pd.to_datetime(["2023-06-01", "2023-06-02", "2023-06-01"]),
            "close":  [100.0, 105.0, 200.0],
        })
        result = mw.calc_daily_return_for_date(df, pd.Timestamp("2023-06-02"))
        assert "A" in result.index
        assert "B" not in result.index

    def test_missing_previous_close_remains_unavailable(self):
        df = pd.DataFrame({
            "symbol": ["A", "A", "B"],
            "date": pd.to_datetime(["2023-06-01", "2023-06-02", "2023-06-02"]),
            "close": [100.0, 105.0, 50.0],
        })
        result = mw.calc_daily_return_for_date(df, pd.Timestamp("2023-06-02"))
        assert "B" in result.index
        assert pd.isna(result.loc["B", "prev_close"])
        assert pd.isna(result.loc["B", "return_1d_pct"])

    def test_zero_previous_close_does_not_create_infinite_return(self):
        df = pd.DataFrame({
            "symbol": ["A", "A"],
            "date": pd.to_datetime(["2023-06-01", "2023-06-02"]),
            "close": [0.0, 10.0],
        })
        result = mw.calc_daily_return_for_date(df, pd.Timestamp("2023-06-02"))
        assert pd.isna(result.loc["A", "return_1d_pct"])

    def test_output_columns(self):
        df = _make_prices_df()
        result = mw.calc_daily_return_for_date(df, pd.Timestamp("2023-06-02"))
        assert "close" in result.columns
        assert "prev_close" in result.columns
        assert "return_1d_pct" in result.columns

    def test_indexed_by_symbol(self):
        df = _make_prices_df()
        result = mw.calc_daily_return_for_date(df, pd.Timestamp("2023-06-02"))
        assert result.index.name == "symbol"

    def test_positive_returns_when_price_rises(self):
        df = _make_prices_df()
        result = mw.calc_daily_return_for_date(df, pd.Timestamp("2023-06-05"))
        assert result.loc["A", "return_1d_pct"] > 0


class TestMarketWatchConstants:

    def test_index_symbols_non_empty(self):
        assert len(mw._INDEX_SYMBOLS) > 0

    def test_ihsg_present(self):
        assert "^JKSE" in mw._INDEX_SYMBOLS
        assert mw._INDEX_SYMBOLS["^JKSE"] == "IHSG"

    def test_fx_symbols_non_empty(self):
        assert len(mw._FX_SYMBOLS) > 0

    def test_usdidr_present(self):
        assert "USDIDR" in mw._FX_SYMBOLS

    def test_index_colors_match_labels(self):
        for _sym, label in mw._INDEX_SYMBOLS.items():
            assert label in mw._INDEX_COLORS

    def test_fx_label_reverse_map(self):
        for sym, (label, _flag, _color) in mw._FX_SYMBOLS.items():
            assert mw._FX_LABEL_TO_SYM[label] == sym


class TestLiveProfileParsing:

    def test_derives_previous_close_from_percentage(self):
        result = mw.parse_live_profile_item({
            "symbol": "AAA",
            "price": 110.0,
            "changesPercentage": 10.0,
        })
        assert result["prev_close"] == pytest.approx(100.0)
        assert result["return_1d_pct"] == 10.0

    def test_derives_return_from_absolute_change(self):
        result = mw.parse_live_profile_item({
            "symbol": "AAA",
            "price": 105.0,
            "changes": 5.0,
        })
        assert result["prev_close"] == 100.0
        assert result["return_1d_pct"] == pytest.approx(5.0)

    def test_rejects_invalid_or_incomplete_rows(self):
        assert mw.parse_live_profile_item({"symbol": "AAA", "price": "bad"}) is None
        assert mw.parse_live_profile_item({"symbol": "AAA", "price": 100.0}) is None
        assert mw.parse_live_profile_item({
            "symbol": "AAA", "price": 0.0, "changesPercentage": -100.0,
        }) is None


class TestSizeColorOptions:

    def test_size_options_present(self):
        assert "Market Cap" in mw._SIZE_OPTIONS
        assert "Dividend Yield" in mw._SIZE_OPTIONS

    def test_color_option_col_map_non_empty(self):
        assert len(mw._COLOR_OPTION_COL_MAP) > 0

    def test_1d_return_mapped_to_none(self):
        assert mw._COLOR_OPTION_COL_MAP["1D Return %"] is None

    def test_usd_return_map_structure(self):
        for label, (col, days) in mw._USD_RETURN_MAP.items():
            assert isinstance(col, str)
            assert isinstance(days, int)
            assert days > 0


class TestBuildTreeInput:

    def _make_df_tree(self):
        return pd.DataFrame({
            "sector":   ["Tech", "Energy", "Tech"],
            "industry": ["Software", "Oil", "Hardware"],
            "yield":    [2.5, 5.0, 1.0],
            "mktCap_B": [100.0, 200.0, 50.0],
        }, index=pd.Index(["AAA", "BBB", "CCC"], name="stock"))

    def test_same_size_and_color_var_no_error(self):
        """Regression: picking the same variable (e.g. 'Dividend Yield') for
        both size and color must not raise
        ``ValueError: Columns must be same length as key``."""
        df_tree = self._make_df_tree()
        result = mw.build_tree_input(
            df_tree,
            size_col="yield",
            size_var="Dividend Yield",
            color_col_data=df_tree["yield"],
            color_var_label="Dividend Yield",
        )
        assert list(result.columns) == ["sector", "industry", "Dividend Yield"]
        assert len(result) == 3

    def test_same_size_and_color_var_feeds_prep_treemap(self):
        """The shared column must flow through hd.prep_treemap cleanly."""
        df_tree = self._make_df_tree()
        tree_input = mw.build_tree_input(
            df_tree,
            size_col="yield",
            size_var="Dividend Yield",
            color_col_data=df_tree["yield"],
            color_var_label="Dividend Yield",
        )
        tree_data = mw.hd.prep_treemap(
            tree_input,
            size_var="Dividend Yield",
            color_var="Dividend Yield",
            color_threshold=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            add_label="color_var",
            group_secs=True,
        )
        assert len(tree_data) > 0

    def test_distinct_size_and_color_var(self):
        df_tree = self._make_df_tree()
        result = mw.build_tree_input(
            df_tree,
            size_col="mktCap_B",
            size_var="Market Cap",
            color_col_data=df_tree["yield"],
            color_var_label="Dividend Yield",
        )
        assert list(result.columns) == [
            "sector", "industry", "Market Cap", "Dividend Yield",
        ]
        assert len(result) == 3

    def test_inf_values_sanitized_and_dropped(self):
        df_tree = self._make_df_tree()
        df_tree.loc["BBB", "yield"] = np.inf
        result = mw.build_tree_input(
            df_tree,
            size_col="yield",
            size_var="Dividend Yield",
            color_col_data=df_tree["yield"],
            color_var_label="Dividend Yield",
        )
        assert "BBB" not in result.index
        assert np.isfinite(result["Dividend Yield"]).all()

    def test_nan_rows_dropped(self):
        df_tree = self._make_df_tree()
        df_tree.loc["CCC", "yield"] = np.nan
        result = mw.build_tree_input(
            df_tree,
            size_col="yield",
            size_var="Dividend Yield",
            color_col_data=df_tree["yield"],
            color_var_label="Dividend Yield",
        )
        assert "CCC" not in result.index


class TestTopMovers:

    def test_filters_by_sign_and_ignores_missing_values(self):
        df = pd.DataFrame({"return_1d_pct": [3.0, 0.0, -2.0, np.nan, 1.0, -5.0]})
        gainers, losers = mw.select_top_movers(df)
        assert gainers["return_1d_pct"].tolist() == [3.0, 1.0]
        assert losers["return_1d_pct"].tolist() == [-5.0, -2.0]

    def test_all_flat_returns_empty_frames(self):
        df = pd.DataFrame({"return_1d_pct": [0.0, 0.0]})
        gainers, losers = mw.select_top_movers(df)
        assert gainers.empty
        assert losers.empty


class TestReturnColorThresholds:

    def test_scales_widen_with_horizon(self):
        assert max(mw.get_return_color_threshold("1D Return %")) == 10
        assert max(mw.get_return_color_threshold("1M Return %")) == 25
        assert max(mw.get_return_color_threshold("10Y Return %")) == 100

    def test_usd_and_total_variants_share_the_period_scale(self):
        expected = mw.get_return_color_threshold("1Y Return %")
        assert mw.get_return_color_threshold("1Y USD Return %") == expected
        assert mw.get_return_color_threshold("Total 1Y USD Return %") == expected


class TestNormalizeOnSharedDate:

    def test_rebases_all_series_on_earliest_shared_date(self):
        prices = pd.DataFrame({
            "index": ["A", "A", "A", "B", "B"],
            "date": pd.to_datetime([
                "2023-01-02", "2023-01-03", "2023-01-04",
                "2023-01-03", "2023-01-04",
            ]),
            "close": [90.0, 100.0, 110.0, 200.0, 220.0],
        })
        result, shared_start = mw.normalize_on_shared_date(prices, "index")
        assert shared_start == pd.Timestamp("2023-01-03")
        assert result["date"].min() == shared_start
        baseline = result[result["date"] == shared_start]
        assert baseline["normalized"].tolist() == [100.0, 100.0]

    def test_no_shared_date_returns_empty_frame(self):
        prices = pd.DataFrame({
            "index": ["A", "B"],
            "date": pd.to_datetime(["2023-01-02", "2023-01-03"]),
            "close": [100.0, 200.0],
        })
        result, shared_start = mw.normalize_on_shared_date(prices, "index")
        assert result.empty
        assert shared_start is None


class TestUsdIdrPeriodFactor:

    @patch("requests.get")
    def test_uses_requested_boundary_not_oldest_buffer_row(self, mock_get):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "historical": [
                {"date": "2023-01-10", "close": 15_500.0},
                {"date": "2023-01-03", "close": 15_000.0},
                {"date": "2022-12-20", "close": 14_000.0},
            ]
        }
        mock_get.return_value = response
        factor = mw.get_usdidr_period_fx_factor("key", "2023-01-10", 7)
        assert factor == pytest.approx(15_000 / 15_500)

    @patch("requests.get")
    def test_response_order_does_not_change_factor(self, mock_get):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "historical": [
                {"date": "2023-01-03", "close": 15_000.0},
                {"date": "2023-01-10", "close": 15_500.0},
            ]
        }
        mock_get.return_value = response
        factor = mw.get_usdidr_period_fx_factor("key", "2023-01-10", 7)
        assert factor == pytest.approx(15_000 / 15_500)


class TestLatestSessionDate:

    def test_before_open_uses_previous_session(self):
        from datetime import datetime, time, date
        # Friday 2 Oct 2026, 02:01 in Jakarta: IDX has not opened yet
        now = datetime(2026, 10, 2, 2, 1)
        assert mw.latest_session_date(now, time(9, 0)) == date(2026, 10, 1)

    def test_after_open_uses_today(self):
        from datetime import datetime, time, date
        now = datetime(2026, 10, 2, 9, 30)
        assert mw.latest_session_date(now, time(9, 0)) == date(2026, 10, 2)

    def test_weekend_and_monday_pre_open_roll_back_to_friday(self):
        from datetime import datetime, time, date
        assert mw.latest_session_date(datetime(2026, 10, 4, 12, 0), time(9, 0)) == date(2026, 10, 2)
        assert mw.latest_session_date(datetime(2026, 10, 5, 8, 0), time(9, 0)) == date(2026, 10, 2)


class TestIsSessionComplete:

    def _prices(self, n_prev, n_today):
        rows = [("S%d" % i, "2023-06-01") for i in range(n_prev)]
        rows += [("S%d" % i, "2023-06-02") for i in range(n_today)]
        df = pd.DataFrame(rows, columns=["symbol", "date"])
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = 1.0
        return df

    def test_full_day_is_complete(self):
        assert mw.is_session_complete(self._prices(10, 10), pd.Timestamp("2023-06-02"))

    def test_partially_ingested_day_is_incomplete(self):
        assert not mw.is_session_complete(self._prices(10, 3), pd.Timestamp("2023-06-02"))

    def test_missing_date_is_incomplete(self):
        assert not mw.is_session_complete(self._prices(10, 10), pd.Timestamp("2023-06-05"))

    def test_first_date_without_history_is_complete(self):
        assert mw.is_session_complete(self._prices(10, 10), pd.Timestamp("2023-06-01"))


class TestColorLegend:

    def test_one_swatch_per_bin_with_open_ended_labels(self):
        html = mw.color_legend_html('1D Return %', ['#111111', '#222222', '#333333'], [-1, 1], '%')
        assert html.count('background:') == 3
        assert '&lt; -1%' in html or '< -1%' in html
        assert '-1% to 1%' in html
        assert '> 1%' in html

    def test_ratio_has_no_unit(self):
        html = mw.color_legend_html('PE Ratio', ['#111111', '#222222'], [15], '')
        assert '< 15<' in html and '> 15<' in html



class TestDodgeLabelPositions:

    def test_close_finishes_are_spread_apart(self):
        ends = pd.Series([101.0, 101.2, 95.0], index=["IHSG", "S&P 500", "Nikkei"])
        all_values = pd.Series([90.0, 110.0])
        pos = mw.dodge_label_positions(ends, all_values)
        assert abs(pos["S&P 500"] - pos["IHSG"]) >= 0.07 * 20 - 1e-9
        assert list(pos.index) == list(ends.index)

    def test_well_separated_labels_stay_put(self):
        ends = pd.Series([90.0, 110.0], index=["A", "B"])
        pos = mw.dodge_label_positions(ends, pd.Series([90.0, 110.0]))
        assert pos["A"] == pytest.approx(90.0)
        assert pos["B"] == pytest.approx(110.0)
