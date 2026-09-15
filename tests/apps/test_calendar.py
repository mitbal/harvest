"""
tests/apps/test_calendar.py
Tests for prep_div_month.
Stubs are installed by tests/apps/conftest.py.
"""
from unittest.mock import patch

import pandas as pd
import pytest


with patch.dict("os.environ", {"REDIS_URL": "redis://stub", "FMP_API_KEY": "stub"}):
    import apps.calendar.calendar as cal


# ═════════════════════════════════════════════════════════════════════════════

class TestPrepDivMonth:

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
        result = cal.prep_div_month(full_year_df, month_idx=3)
        for ex_date in result["ex_date"]:
            assert "Mar" in ex_date

    def test_output_columns(self, full_year_df):
        result = cal.prep_div_month(full_year_df, month_idx=1)
        expected = {"rank", "url_link", "ex_date", "div_yield", "dividend", "price"}
        assert expected.issubset(set(result.columns))

    def test_multi_row_month(self, full_year_df):
        result = cal.prep_div_month(full_year_df, month_idx=5)
        assert len(result) == 2

    def test_rank_starts_at_one(self, full_year_df):
        result = cal.prep_div_month(full_year_df, month_idx=7)
        assert result["rank"].iloc[0] == 1

    def test_url_link_format(self, full_year_df):
        result = cal.prep_div_month(full_year_df, month_idx=1)
        for url in result["url_link"]:
            assert url.startswith("https://panendividen.com/stock_picker?stock=")

    def test_empty_input_returns_empty_df(self):
        empty = pd.DataFrame({
            "date":        pd.Series(dtype="datetime64[ns]"),
            "symbol":      pd.Series(dtype=str),
            "adjDividend": pd.Series(dtype=float),
            "price":       pd.Series(dtype=float),
            "yield":       pd.Series(dtype=float),
        })
        result = cal.prep_div_month(empty, month_idx=6)
        assert result.empty

    def test_div_yield_has_percent_sign(self, full_year_df):
        result = cal.prep_div_month(full_year_df, month_idx=2)
        for dy in result["div_yield"]:
            assert "%" in dy

    def test_ex_date_formatted_as_dd_mon(self, full_year_df):
        result = cal.prep_div_month(full_year_df, month_idx=1)
        assert result["ex_date"].iloc[0] == "10 Jan"

    def test_december_month(self, full_year_df):
        result = cal.prep_div_month(full_year_df, month_idx=12)
        assert len(result) == 1
        assert "Dec" in result["ex_date"].iloc[0]

    def test_sorted_highest_yield_first(self, full_year_df):
        result = cal.prep_div_month(full_year_df, month_idx=5)
        yields_raw = result["div_yield"].str.rstrip("%").astype(float).tolist()
        assert yields_raw == sorted(yields_raw, reverse=True)
