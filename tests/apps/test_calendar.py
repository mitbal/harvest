"""
tests/apps/test_calendar.py
Tests for the payout calendar page helpers.
Stubs are installed by tests/apps/conftest.py.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


with patch.dict("os.environ", {"REDIS_URL": "redis://stub", "FMP_API_KEY": "stub"}):
    import apps.calendar.calendar as cal


# ═════════════════════════════════════════════════════════════════════════════

class TestPrepDivTable:

    @pytest.fixture
    def full_year_df(self):
        rows = []
        for month in range(1, 13):
            rows.append({
                "date":        pd.Timestamp(f"2025-{month:02d}-10"),
                "symbol":      f"STCK{month:02d}.JK",
                "adjDividend": 100.0 * month,
                "price":       5000.0,
                "yield":       100.0 * month / 5000.0 * 100,
            })
        rows.append({
            "date":        pd.Timestamp("2025-05-20"),
            "symbol":      "EXTRA.JK",
            "adjDividend": 80.0,
            "price":       4000.0,
            "yield":       80.0 / 4000.0 * 100,
        })
        return pd.DataFrame(rows)

    def test_filters_to_requested_month(self, full_year_df):
        result = cal.prep_div_table(full_year_df, 3)
        assert (result["ex_date"].dt.month == 3).all()

    def test_output_columns(self, full_year_df):
        result = cal.prep_div_table(full_year_df, 1)
        expected = {"rank", "url_link", "ex_date", "div_yield", "dividend", "price"}
        assert expected.issubset(set(result.columns))

    def test_multi_row_month(self, full_year_df):
        result = cal.prep_div_table(full_year_df, 5)
        assert len(result) == 2

    def test_rank_starts_at_one(self, full_year_df):
        result = cal.prep_div_table(full_year_df, 7)
        assert result["rank"].iloc[0] == 1

    def test_url_link_format(self, full_year_df):
        result = cal.prep_div_table(full_year_df, 1)
        for url in result["url_link"]:
            assert url.startswith("https://panendividen.com/stock_picker?market=ID&stock=")

    def test_url_link_us_market(self):
        us_df = pd.DataFrame([{
            "date":        pd.Timestamp("2025-03-10"),
            "symbol":      "AAPL",
            "adjDividend": 0.25,
            "price":       200.0,
            "yield":       0.125,
        }])
        result = cal.prep_div_table(us_df, 3, market="US")
        assert result["url_link"].iloc[0] == "https://panendividen.com/stock_picker?market=US&stock=AAPL"

    def test_url_link_uses_site_url(self, full_year_df):
        with patch.object(cal, "SITE_URL", "http://localhost:8501"):
            result = cal.prep_div_table(full_year_df, 1)
        assert result["url_link"].iloc[0].startswith("http://localhost:8501/stock_picker?")

    @pytest.mark.parametrize("symbol, expected", [
        ("BBCA.JK", "BBCA"),
        ("AAPL", "AAPL"),
        ("BRK.B", "BRK.B"),
        ("BF-B", "BF-B"),
        ("3M", "3M"),
    ])
    def test_display_text_extracts_ticker(self, symbol, expected):
        import re
        pattern = cal.TICKER_DISPLAY_PATTERN
        market = "ID" if symbol.endswith(".JK") else "US"
        url = f"https://panendividen.com/stock_picker?market={market}&stock={symbol}"
        assert re.search(pattern, url).group(1) == expected

    def test_empty_input_returns_empty_df(self):
        empty = pd.DataFrame({
            "date":        pd.Series(dtype="datetime64[ns]"),
            "symbol":      pd.Series(dtype=str),
            "adjDividend": pd.Series(dtype=float),
            "price":       pd.Series(dtype=float),
            "yield":       pd.Series(dtype=float),
        })
        result = cal.prep_div_table(empty, 6)
        assert result.empty

    def test_yield_and_date_stay_sortable_types(self, full_year_df):
        # The table formats them; numeric/datetime values make column sorting correct
        result = cal.prep_div_table(full_year_df, 2)
        assert pd.api.types.is_float_dtype(result["div_yield"])
        assert pd.api.types.is_datetime64_any_dtype(result["ex_date"])
        assert result["div_yield"].iloc[0] == pytest.approx(4.0)

    def test_december_month(self, full_year_df):
        result = cal.prep_div_table(full_year_df, 12)
        assert len(result) == 1
        assert result["ex_date"].iloc[0] == pd.Timestamp("2025-12-10")

    def test_sorted_highest_yield_first(self, full_year_df):
        result = cal.prep_div_table(full_year_df, 5)
        assert result["div_yield"].tolist() == sorted(result["div_yield"], reverse=True)

    def test_does_not_modify_input(self, full_year_df):
        before = full_year_df.copy()
        cal.prep_div_table(full_year_df, 5)
        pd.testing.assert_frame_equal(full_year_df, before)


class TestFormatDividend:

    @pytest.mark.parametrize("value, expected", [
        (350.0, "350"),
        (1358.18, "1,358.18"),
        (0.24, "0.24"),
        (0.2635, "0.2635"),
        (0.009, "0.009"),
        (107.1, "107.1"),
    ])
    def test_trims_trailing_zeros(self, value, expected):
        assert cal.format_dividend(value) == expected

class TestProjectToNextYear:

    def test_shifts_one_year_keeping_month_and_day(self):
        result = cal.project_to_next_year(pd.Series(["2025-03-10", "2025-11-10"]))
        assert list(result) == [pd.Timestamp("2026-03-10"), pd.Timestamp("2026-11-10")]

    def test_weekday_follows_projected_year(self):
        # 10 Mar 2025 is a Monday; 10 Mar 2026 is a Tuesday
        result = cal.project_to_next_year(pd.Series([pd.Timestamp("2025-03-10")]))
        assert result.iloc[0].day_name() == "Tuesday"

    def test_leap_day_maps_to_feb_28(self):
        # 28 Feb 2025 is a Friday, so no weekend roll applies
        result = cal.project_to_next_year(pd.Series([pd.Timestamp("2024-02-29")]))
        assert result.iloc[0] == pd.Timestamp("2025-02-28")

    def test_saturday_rolls_forward_to_monday(self):
        # 14 Mar 2026 is a Saturday
        result = cal.project_to_next_year(pd.Series([pd.Timestamp("2025-03-14")]))
        assert result.iloc[0] == pd.Timestamp("2026-03-16")

    def test_sunday_rolls_forward_to_monday(self):
        # 15 Mar 2026 is a Sunday
        result = cal.project_to_next_year(pd.Series([pd.Timestamp("2025-03-15")]))
        assert result.iloc[0] == pd.Timestamp("2026-03-16")

    def test_saturday_first_of_month_rolls_to_monday(self):
        # 1 Aug 2026 is a Saturday
        result = cal.project_to_next_year(pd.Series([pd.Timestamp("2025-08-01")]))
        assert result.iloc[0] == pd.Timestamp("2026-08-03")

    def test_saturday_end_of_month_rolls_back_to_friday(self):
        # 30 May 2026 is a Saturday; Monday would be 1 Jun
        result = cal.project_to_next_year(pd.Series([pd.Timestamp("2025-05-30")]))
        assert result.iloc[0] == pd.Timestamp("2026-05-29")

    def test_sunday_last_of_month_rolls_back_to_friday(self):
        # 31 May 2026 is a Sunday; Monday would be 1 Jun
        result = cal.project_to_next_year(pd.Series([pd.Timestamp("2025-05-31")]))
        assert result.iloc[0] == pd.Timestamp("2026-05-29")

    def test_never_lands_on_weekend(self):
        dates = pd.Series(pd.date_range("2025-01-01", "2025-12-31", freq="D"))
        result = cal.project_to_next_year(dates)
        assert (result.dt.weekday < 5).all()
        assert (result.dt.month == dates.dt.month).all()
        assert (result.dt.year == 2026).all()

    def test_idx_lebaran_rolls_past_cuti_bersama(self):
        # 20 Mar 2026 is cuti bersama, then a weekend, then two more cuti bersama days
        result = cal.project_to_next_year(pd.Series([pd.Timestamp("2025-03-20")]), market="ID")
        assert result.iloc[0] == pd.Timestamp("2026-03-25")

    def test_idx_year_end_closure_rolls_back(self):
        # 31 Dec 2026 is the IDX bourse holiday; the next trading day is in January
        result = cal.project_to_next_year(pd.Series([pd.Timestamp("2025-12-31")]), market="ID")
        assert result.iloc[0] == pd.Timestamp("2026-12-30")

    def test_nyse_holiday_rolls_forward(self):
        # 3 Jul 2026 is Independence Day (observed) on NYSE
        result = cal.project_to_next_year(pd.Series([pd.Timestamp("2025-07-03")]), market="US")
        assert result.iloc[0] == pd.Timestamp("2026-07-06")

    def test_nyse_open_on_dec_31(self):
        result = cal.project_to_next_year(pd.Series([pd.Timestamp("2025-12-31")]), market="US")
        assert result.iloc[0] == pd.Timestamp("2026-12-31")

    @pytest.mark.parametrize("market", ["ID", "US"])
    def test_never_lands_on_closed_day(self, market):
        dates = pd.Series(pd.date_range("2025-01-01", "2025-12-31", freq="D"))
        result = cal.project_to_next_year(dates, market=market)
        closed = set(cal.market_holidays(market, [2026]))
        assert not any(d.date() in closed for d in result)
        assert (result.dt.weekday < 5).all()
        assert (result.dt.month == dates.dt.month).all()

    def test_keeps_input_index(self):
        dates = pd.Series([pd.Timestamp("2025-03-14"), pd.Timestamp("2025-03-10")], index=[7, 3])
        result = cal.project_to_next_year(dates)
        assert list(result.index) == [7, 3]

    def test_empty_input(self):
        result = cal.project_to_next_year(pd.Series([], dtype="datetime64[ns]"))
        assert result.empty


class TestMarketHolidays:

    def test_idx_names_in_english(self):
        closed = cal.market_holidays("ID", [2026])
        assert closed[pd.Timestamp("2026-03-21").date()] == "Eid al-Fitr"
        assert closed[pd.Timestamp("2026-03-20").date()] == "Eid al-Fitr Joint Holiday"

    def test_idx_year_end_closure_named(self):
        closed = cal.market_holidays("ID", [2026])
        assert closed[pd.Timestamp("2026-12-31").date()] == "IDX Year-End Holiday"

    def test_nyse_names(self):
        closed = cal.market_holidays("US", [2026])
        assert closed[pd.Timestamp("2026-11-26").date()] == "Thanksgiving Day"
        assert pd.Timestamp("2026-12-31").date() not in closed


class TestPayoutFrequency:

    @staticmethod
    def _events(counts):
        """counts: {symbol: payments} -> one row per payment on distinct dates."""
        rows = []
        for symbol, n in counts.items():
            for i in range(n):
                rows.append({
                    "date": pd.Timestamp("2026-01-05") + pd.Timedelta(days=28 * i),
                    "symbol": symbol,
                    "adjDividend": 1.0 + i,
                    "price": 100.0,
                })
        return pd.DataFrame(rows)

    def test_labels(self):
        assert cal.payout_frequency_label(1) == "Once per year"
        assert cal.payout_frequency_label(3) == "3 times per year"
        assert cal.payout_frequency_label(4) == "4 or more times per year"
        assert cal.payout_frequency_label(5) == "4 or more times per year"
        assert cal.payout_frequency_label(12) == "4 or more times per year"

    def test_quarterly_and_monthly_share_bucket(self):
        summary = cal.payout_frequency(self._events({"O": 12, "KO": 4, "COST": 5}))
        assert summary["frequency"].tolist() == ["4 or more times per year"]
        assert summary.loc[0, "count"] == 3
        assert summary.loc[0, "stocks"] == "COST, KO, O"

    def test_buckets_in_order_and_empty_ones_dropped(self):
        summary = cal.payout_frequency(self._events({"A": 1, "B": 12, "C": 2, "D": 1}))
        assert summary["frequency"].tolist() == ["Once per year", "Twice per year", "4 or more times per year"]
        assert summary["count"].tolist() == [2, 1, 1]
        assert summary.loc[0, "stocks"] == "A, D"

    def test_payouts_rolled_onto_same_day_both_counted(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-03-16", "2026-03-16"]),
            "symbol": ["X", "X"],
            "adjDividend": [10.0, 25.0],
            "price": [100.0, 100.0],
        })
        assert cal.payout_frequency(df)["frequency"].tolist() == ["Twice per year"]

    def test_exact_duplicate_rows_counted_once(self):
        df = pd.concat([self._events({"X": 1})] * 2)
        assert cal.payout_frequency(df)["frequency"].tolist() == ["Once per year"]

    def test_empty_input(self):
        summary = cal.payout_frequency(self._events({}))
        assert summary.empty
        assert list(summary.columns) == ["frequency", "count", "stocks"]


class TestAddYield:

    @staticmethod
    def _df(rows):
        return pd.DataFrame(rows, columns=["date", "symbol", "adjDividend", "price"])

    def test_computes_yield_percent(self):
        result = cal.add_yield(self._df([["2026-03-10", "A", 50.0, 1000.0]]))
        assert result.loc[0, "yield"] == pytest.approx(5.0)

    @pytest.mark.parametrize("dividend, price", [
        (10.0, 0.0),            # zero price -> inf
        (10.0, -5.0),           # negative price
        (10.0, float("nan")),   # missing price
        (float("nan"), 100.0),  # missing dividend
        (0.0, 143.73),          # non-payer row (e.g. CBRE 2024)
        (-1.0, 100.0),          # negative dividend
    ])
    def test_drops_invalid_rows(self, dividend, price):
        df = self._df([["2026-03-10", "BAD", dividend, price], ["2026-03-11", "GOOD", 1.0, 100.0]])
        result = cal.add_yield(df)
        assert result["symbol"].tolist() == ["GOOD"]
        assert np.isfinite(result["yield"]).all()

    def test_all_invalid_gives_empty_frame(self):
        result = cal.add_yield(self._df([["2026-03-10", "A", 1.0, 0.0]]))
        assert result.empty
        assert "yield" in result.columns

    def test_does_not_modify_input(self):
        df = self._df([["2026-03-10", "A", 1.0, 0.0]])
        cal.add_yield(df)
        assert "yield" not in df.columns and len(df) == 1

    def test_numeric_strings_accepted(self):
        result = cal.add_yield(self._df([["2026-03-10", "A", "50", "1000"]]))
        assert result.loc[0, "yield"] == pytest.approx(5.0)


class TestSyariahSymbols:

    def test_returns_flagged_symbols(self):
        scores = pd.DataFrame({"symbol": ["BBRI.JK", "TLKM.JK", "BBCA.JK"], "is_syariah": [False, True, False]})
        assert cal.syariah_symbols(scores) == {"TLKM.JK"}

    def test_missing_column_returns_none(self):
        scores = pd.DataFrame({"symbol": ["TLKM.JK"], "sector": ["Telecom"]})
        assert cal.syariah_symbols(scores) is None

    def test_missing_flag_treated_as_not_syariah(self):
        scores = pd.DataFrame({"symbol": ["A.JK", "B.JK"], "is_syariah": [None, True]})
        assert cal.syariah_symbols(scores) == {"B.JK"}

    def test_duplicate_score_rows_do_not_duplicate_events(self):
        scores = pd.DataFrame({"symbol": ["TLKM.JK", "TLKM.JK"], "is_syariah": [True, True]})
        events = pd.DataFrame({"symbol": ["TLKM.JK", "TLKM.JK", "BBCA.JK"], "date": ["2026-05-01", "2026-11-01", "2026-06-01"]})
        filtered = events[events["symbol"].isin(cal.syariah_symbols(scores))]
        assert len(filtered) == 2


class TestParseQueryParams:

    YEARS = [2023, 2024, 2025]

    def parse(self, qp):
        return cal.parse_query_params(qp, self.YEARS, default_year=2025, current_month=10)

    def test_defaults(self):
        assert self.parse({}) == (0, 2, 9)

    def test_monthly_view_year_and_month(self):
        assert self.parse({"view": "Monthly", "year": "2023", "month": "6"}) == (1, 0, 5)

    @pytest.mark.parametrize("qp", [
        {"year": "abc"},
        {"year": "1999"},
        {"month": "13"},
        {"month": "0"},
        {"month": "x"},
        {"view": "weekly"},
    ])
    def test_bad_values_fall_back(self, qp):
        assert self.parse(qp) == (0, 2, 9)


class TestYieldSummary:

    @pytest.fixture
    def stats(self):
        return pd.DataFrame({
            "date": pd.to_datetime(["2026-03-10", "2026-03-10", "2026-03-20", "2026-11-05"]),
            "yield": [2.0, 4.0, 6.0, 1.0],
        })

    def test_by_month(self, stats):
        summary = cal.yield_summary(stats, "month")
        assert summary["month"].tolist() == [3, 11]
        assert summary["month_name"].tolist() == ["Mar", "Nov"]
        assert summary["events"].tolist() == [3, 1]
        assert summary["avg_yield"].tolist() == [4.0, 1.0]

    def test_by_day(self, stats):
        summary = cal.yield_summary(stats[stats["date"].dt.month == 3], "day")
        assert summary["day"].tolist() == [10, 20]
        assert summary["events"].tolist() == [2, 1]
        assert summary["avg_yield"].tolist() == [3.0, 6.0]
        assert "month_name" not in summary.columns
