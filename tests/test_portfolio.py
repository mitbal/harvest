"""
tests/test_portfolio.py
Portfolio parsing, validation and KPI maths in harvest/portfolio.py.

Pure pandas, so no Streamlit stubs are needed.
"""
import json
from unittest.mock import patch

import pandas as pd
import pytest

from harvest import portfolio as hpf


# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizePortfolio:

    def test_normalizes_symbols_and_numeric_separators(self):
        result = hpf.normalize_portfolio(pd.DataFrame({
            "Symbol": [" bbca.jk "],
            "Available Lot": ["1,000"],
            "Average Price": ["9,500"],
        }))

        assert result.to_dict(orient="records") == [{
            "Symbol": "BBCA",
            "Available Lot": 1000,
            "Average Price": 9500.0,
        }]

    def test_combines_duplicate_symbols_at_weighted_average_cost(self):
        result = hpf.normalize_portfolio(pd.DataFrame({
            "Symbol": ["BBCA", "bbca"],
            "Available Lot": [10, 30],
            "Average Price": [9000, 10000],
        }))

        assert result.loc[0, "Available Lot"] == 40
        assert result.loc[0, "Average Price"] == 9750

    @pytest.mark.parametrize("column,value", [
        ("Available Lot", 0),
        ("Available Lot", float("inf")),
        ("Average Price", -1),
        ("Average Price", "not-a-number"),
    ])
    def test_rejects_invalid_numeric_values(self, column, value):
        portfolio = pd.DataFrame({
            "Symbol": ["BBCA"],
            "Available Lot": [value if column == "Available Lot" else 10],
            "Average Price": [value if column == "Average Price" else 9500],
        })

        with pytest.raises(ValueError, match=column):
            hpf.normalize_portfolio(portfolio)

    def test_requires_expected_columns(self):
        with pytest.raises(ValueError, match="Average Price"):
            hpf.normalize_portfolio({"Symbol": ["BBCA"], "Available Lot": [10]})

    def test_rejects_portfolio_without_symbols(self):
        with pytest.raises(ValueError, match="at least one stock"):
            hpf.normalize_portfolio({"Symbol": ["  "], "Available Lot": [10], "Average Price": [9500]})


class TestRawPortfolioParsing:

    def test_parses_stockbit_row(self):
        raw = "bbca 10 unused 9,500 a b c d e f g"
        result = hpf.parse_raw_portfolio(raw)

        assert result.loc[0].to_dict() == {
            "Symbol": "BBCA",
            "Available Lot": 10,
            "Average Price": 9500.0,
        }

    def test_rejects_incomplete_stockbit_row(self):
        with pytest.raises(ValueError, match="multiple of 11"):
            hpf.parse_raw_portfolio("BBCA 10 incomplete")

    def test_rejects_empty_paste(self):
        with pytest.raises(ValueError, match="Paste"):
            hpf.parse_raw_portfolio("   ")


class TestPortfolioPersistence:

    def test_serializes_as_json_safe_records(self):
        records = hpf.portfolio_to_records(pd.DataFrame({
            "Symbol": ["BBCA"],
            "Available Lot": [10],
            "Average Price": [9500],
        }))

        assert records == [{
            "Symbol": "BBCA",
            "Available Lot": 10,
            "Average Price": 9500.0,
        }]


class TestParseCompanyProfiles:

    def test_parses_wrapped_json_payload(self):
        raw = json.dumps({"date": "2026-10-01", "content": json.dumps([
            {"symbol": "BBCA.JK", "price": 9500, "sector": "Financials", "lastDiv": 250},
        ])})

        result = hpf.parse_company_profiles(raw)

        assert result["Symbol"].tolist() == ["BBCA"]
        assert result["price"].tolist() == [9500]

    def test_parses_parquet_payload(self):
        frame = pd.DataFrame({
            "symbol": ["bbca.jk", "BBCA.JK"],
            "price": [9500, 9600],
            "sector": ["Financials", "Financials"],
            "lastDiv": [250.0, 260.0],
        })

        result = hpf.parse_company_profiles(frame.to_parquet())

        # Duplicates after cleaning keep the first row
        assert result["Symbol"].tolist() == ["BBCA"]
        assert result["price"].tolist() == [9500]

    def test_rejects_missing_columns(self):
        raw = json.dumps([{"symbol": "BBCA.JK", "price": 9500, "sector": "Financials"}])
        with pytest.raises(KeyError, match="lastDiv"):
            hpf.parse_company_profiles(raw)

    def test_rejects_unparseable_payload(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            hpf.parse_company_profiles(b"not json")


class TestAnnualDividendFromHistory:

    def test_uses_finalized_fiscal_year_sum(self):
        history = pd.DataFrame({
            "date": ["2026-04-20"], "adjDividend": [110.0],
            "fiscal_year": [2025], "dividend_type": ["final"],
        })
        with patch.object(hpf.hd, "calc_latest_finalized_dividend_sum", return_value=160.0):
            assert hpf.annual_dividend_from_history(history, fallback=60.0) == 160.0

    @pytest.mark.parametrize("history", [
        None,
        pd.DataFrame({"date": ["2026-04-20"], "adjDividend": [110.0]}),
    ])
    def test_falls_back_without_fiscal_year_data(self, history):
        assert hpf.annual_dividend_from_history(history, fallback=60.0) == 60.0


class TestMergeMarketData:

    @pytest.fixture
    def profiles(self):
        return pd.DataFrame({
            "Symbol": ["BBCA"], "price": [9500], "sector": ["Financials"], "lastDiv": [250.0],
        })

    def test_excludes_unknown_symbols(self, profiles):
        portfolio = pd.DataFrame({
            "Symbol": ["BBCA", "MISSING"], "Available Lot": [10, 5], "Average Price": [9000, 1000],
        })

        result, unknown = hpf.merge_market_data(portfolio, profiles)

        assert result["Symbol"].tolist() == ["BBCA"]
        assert result.loc[0, "last_price"] == 9500
        assert result.loc[0, "div_rate"] == 250.0
        assert unknown == ["MISSING"]

    def test_rejects_when_nothing_matches(self, profiles):
        portfolio = pd.DataFrame({"Symbol": ["MISSING"], "Available Lot": [5], "Average Price": [1000]})
        with pytest.raises(ValueError, match="MISSING"):
            hpf.merge_market_data(portfolio, profiles)


class TestMarketDataValidation:

    def test_rejects_when_every_row_is_invalid(self):
        frame = pd.DataFrame({
            "Symbol": ["BBCA"],
            "last_price": [None],
            "div_rate": [250],
            "sector": ["Financials"],
        })

        with pytest.raises(ValueError, match="BBCA"):
            hpf.validate_market_data(frame)

    def test_excludes_invalid_rows_and_keeps_the_rest(self):
        frame = pd.DataFrame({
            "Symbol": ["BBCA", "BAD"],
            "last_price": [9500, None],
            "div_rate": [250, 10],
            "sector": ["Financials", "Energy"],
        })

        result, excluded = hpf.validate_market_data(frame)

        assert result["Symbol"].tolist() == ["BBCA"]
        assert excluded == ["BAD"]

    def test_accepts_zero_dividend_and_fills_sector(self):
        frame = pd.DataFrame({
            "Symbol": ["BBCA"],
            "last_price": [9500],
            "div_rate": [0],
            "sector": [None],
        })

        result, excluded = hpf.validate_market_data(frame)

        assert excluded == []
        assert result.loc[0, "div_rate"] == 0
        assert result.loc[0, "sector"] == "Unclassified"


class TestPortfolioKpis:

    @pytest.fixture
    def holdings(self):
        return hpf.enrich_holdings(pd.DataFrame({
            "Symbol":        ["BBCA", "TLKM"],
            "Available Lot": [10.0, 20.0],
            "Average Price": [9500.0, 3800.0],
            "div_rate":      [250.0, 200.0],
            "last_price":    [10000.0, 3600.0],
            "sector":        ["Financials", "Telecom"],
        }))

    def test_per_holding_columns(self, holdings):
        bbca = holdings.loc[0]
        assert bbca["total_invested"] == 10 * 100 * 9500
        assert bbca["market_value"] == 10 * 100 * 10000
        assert bbca["total_dividend"] == 10 * 100 * 250
        assert bbca["yield_on_cost"] == pytest.approx(250 / 9500 * 100)
        assert bbca["yield_on_price"] == pytest.approx(250 / 10000 * 100)

    def test_portfolio_summary(self, holdings):
        summary = hpf.summarize_portfolio(holdings, target_million=240)

        invested = 10 * 100 * 9500 + 20 * 100 * 3800
        value = 10 * 100 * 10000 + 20 * 100 * 3600
        assert summary["annual_dividend"] == 650_000
        assert summary["total_invested"] == invested
        assert summary["market_value"] == value
        assert summary["market_delta"] == value - invested
        assert summary["target_progress"] == pytest.approx(650_000 / 240_000_000 * 100)
        assert summary["yield_on_cost"] == pytest.approx(650_000 / invested * 100)
        assert summary["yield_on_price"] == pytest.approx(650_000 / value * 100)

    def test_zero_target_does_not_divide_by_zero(self, holdings):
        assert hpf.summarize_portfolio(holdings, target_million=0)["target_progress"] == 0.0


class TestProjectIncome:

    def test_year_one_is_one_year_of_growth(self):
        result = hpf.project_income(1_000_000, 10_000_000, 10.0, 3)
        assert result["years"].tolist() == ["Year 01", "Year 02", "Year 03"]
        assert result.loc[0, "returns"] == pytest.approx(1_100_000)
        assert result.loc[2, "returns"] == pytest.approx(1_331_000)

    def test_yield_includes_reinvested_dividends_in_cost(self):
        result = hpf.project_income(1_000_000, 10_000_000, 10.0, 2)
        # Year 1: 1.1M on 10M + 1M reinvested; Year 2: 1.21M on 11M + 1.1M
        assert result.loc[0, "yield"] == pytest.approx(1_100_000 / 11_000_000 * 100)
        assert result.loc[1, "yield"] == pytest.approx(1_210_000 / 12_100_000 * 100)

    def test_zero_growth_keeps_income_flat(self):
        result = hpf.project_income(500_000, 10_000_000, 0.0, 5)
        assert (result["returns"] == 500_000).all()


class TestBuildLastYearDividends:

    @pytest.fixture
    def holdings(self):
        return pd.DataFrame({
            "Symbol": ["BBCA", "TLKM"],
            "current_lot": [10.0, 20.0],
            "last_price": [10000.0, 4000.0],
        })

    def test_uses_plain_symbol_and_trailing_year_window(self, holdings):
        divs = {"BBCA.JK": pd.DataFrame({
            "date": ["2024-01-10", "2026-03-15", "2026-09-01"],
            "adjDividend": [100.0, 200.0, 50.0],
        })}

        result = hpf.build_last_year_dividends(holdings, divs, today="2026-10-01")

        assert len(result) == 1
        assert result[0]["Symbol"].tolist() == ["BBCA", "BBCA"]
        assert result[0]["adjDividend"].tolist() == [200.0, 50.0]
        assert result[0]["Lot"].tolist() == [10.0, 10.0]

    def test_accepts_datetime_dates(self, holdings):
        divs = {"TLKM.JK": pd.DataFrame({
            "date": pd.to_datetime(["2026-06-01"]).tz_localize("Asia/Jakarta"),
            "adjDividend": [180.0],
        })}

        result = hpf.build_last_year_dividends(holdings, divs, today="2026-10-01")

        assert result[0]["Symbol"].tolist() == ["TLKM"]
        assert result[0]["yield"].tolist() == [pytest.approx(4.5)]


class TestFormatIdr:

    @pytest.mark.parametrize("value,expected", [
        (49_186_300, "IDR 49.2M"),
        (708_800_000, "IDR 708.8M"),
        (1_250_000_000, "IDR 1.25B"),
        (2_500_000_000_000, "IDR 2.5T"),
        (240_000_000, "IDR 240M"),
        (12_500, "IDR 12.5K"),
        (950, "IDR 950"),
        (0, "IDR 0"),
        (-43_010_000, "-IDR 43M"),
    ])
    def test_compact_values(self, value, expected):
        assert hpf.format_idr(value) == expected

    def test_signed_adds_plus_for_gains(self):
        assert hpf.format_idr(4_100_000, signed=True) == "+IDR 4.1M"
        assert hpf.format_idr(0, signed=True) == "IDR 0"
