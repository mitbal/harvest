"""
tests/apps/test_porto_overview.py
Tests for _is_positive_number, constants, and KPI math.
Stubs are installed by tests/apps/conftest.py.
"""
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


with patch.dict("os.environ", {"REDIS_URL": "redis://stub", "FMP_API_KEY": "stub"}):
    import apps.porto.porto_overview as po


# ══════════════════════════════════════════════════════════════════════════════

class TestIsPositiveNumber:

    def test_valid_integer_string(self):
        assert po._is_positive_number("100") is True

    def test_valid_float_string(self):
        assert po._is_positive_number("3.14") is True

    def test_zero_is_not_positive(self):
        assert po._is_positive_number("0") is False

    def test_negative_is_not_positive(self):
        assert po._is_positive_number("-5") is False

    def test_non_numeric_string(self):
        assert po._is_positive_number("abc") is False

    def test_empty_string(self):
        assert po._is_positive_number("") is False

    def test_none_is_not_positive(self):
        assert po._is_positive_number(None) is False

    def test_very_small_positive(self):
        assert po._is_positive_number("0.0001") is True

    def test_comma_number_fails(self):
        assert po._is_positive_number("1,000") is False


class TestRequiredColumns:

    def test_required_columns_set(self):
        assert po.REQUIRED_COLUMNS == {"Symbol", "Available Lot", "Average Price"}


class TestPasteRawConstant:

    def test_value(self):
        assert po.PASTE_RAW_FIELDS_PER_ROW == 11


class TestNormalizePortfolio:

    def test_normalizes_symbols_and_numeric_separators(self):
        result = po.normalize_portfolio(pd.DataFrame({
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
        result = po.normalize_portfolio(pd.DataFrame({
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
            po.normalize_portfolio(portfolio)

    def test_requires_expected_columns(self):
        with pytest.raises(ValueError, match="Average Price"):
            po.normalize_portfolio({"Symbol": ["BBCA"], "Available Lot": [10]})


class TestRawPortfolioParsing:

    def test_parses_stockbit_row(self):
        raw = "bbca 10 unused 9,500 a b c d e f g"
        result = po.parse_raw_portfolio(raw)

        assert result.loc[0].to_dict() == {
            "Symbol": "BBCA",
            "Available Lot": 10,
            "Average Price": 9500.0,
        }

    def test_rejects_incomplete_stockbit_row(self):
        with pytest.raises(ValueError, match="multiple of 11"):
            po.parse_raw_portfolio("BBCA 10 incomplete")


class TestPortfolioPersistence:

    def test_serializes_as_json_safe_records(self):
        records = po.portfolio_to_records(pd.DataFrame({
            "Symbol": ["BBCA"],
            "Available Lot": [10],
            "Average Price": [9500],
        }))

        assert records == [{
            "Symbol": "BBCA",
            "Available Lot": 10,
            "Average Price": 9500.0,
        }]


class TestMarketDataValidation:

    def test_rejects_missing_market_price(self):
        frame = pd.DataFrame({
            "Symbol": ["BBCA"],
            "last_price": [None],
            "div_rate": [250],
            "sector": ["Financials"],
        })

        with pytest.raises(ValueError, match="Market Price.*BBCA"):
            po.validate_market_data(frame)

    def test_accepts_zero_dividend_and_fills_sector(self):
        frame = pd.DataFrame({
            "Symbol": ["BBCA"],
            "last_price": [9500],
            "div_rate": [0],
            "sector": [None],
        })

        result = po.validate_market_data(frame)

        assert result.loc[0, "div_rate"] == 0
        assert result.loc[0, "sector"] == "Unclassified"


class TestPortfolioDataLoading:

    def test_rejects_partial_market_data_match(self):
        redis_client = MagicMock()
        redis_client.get.return_value = json.dumps([{
            "symbol": "BBCA.JK",
            "price": 9500,
            "sector": "Financials",
            "lastDiv": 250,
        }])
        portfolio = pd.DataFrame({
            "Symbol": ["BBCA", "MISSING"],
            "Available Lot": [10, 5],
            "Average Price": [9000, 1000],
        })

        with patch.object(po, "connect_redis", return_value=redis_client):
            with pytest.raises(ValueError, match="MISSING"):
                po.get_company_profile_data(portfolio)

    def test_dividend_failure_does_not_discard_other_stocks(self):
        portfolio = pd.DataFrame({"Symbol": ["FAIL", "BBCA"]})
        bbca_history = pd.DataFrame({
            "date": ["2025-04-01"],
            "adjDividend": [250],
            "dividend_type": ["final"],
        })

        with patch.object(
            po.hd,
            "get_dividend_history_single_stock_dag",
            side_effect=[RuntimeError("network error"), bbca_history],
        ):
            result = po.get_dividend_data(portfolio)

        assert list(result) == ["BBCA.JK"]


class TestPortfolioKpiMath:

    @pytest.fixture
    def enriched_df(self):
        return pd.DataFrame({
            "Symbol":        ["BBCA", "TLKM"],
            "Available Lot": [10.0, 20.0],
            "Average Price": [9500.0, 3800.0],
            "div_rate":      [250.0, 200.0],
            "last_price":    [10000.0, 3600.0],
            "sector":        ["Financials", "Telecom"],
        })

    def test_total_invested(self, enriched_df):
        df = enriched_df.copy()
        df["total_invested"] = df["Available Lot"] * df["Average Price"] * 100
        assert df.loc[0, "total_invested"] == 10 * 9500 * 100
        assert df.loc[1, "total_invested"] == 20 * 3800 * 100

    def test_yield_on_cost(self, enriched_df):
        df = enriched_df.copy()
        df["yield_on_cost"] = df["div_rate"] / df["Average Price"] * 100
        assert abs(df.loc[0, "yield_on_cost"] - 250 / 9500 * 100) < 1e-6

    def test_yield_on_price(self, enriched_df):
        df = enriched_df.copy()
        df["yield_on_price"] = df["div_rate"] / df["last_price"] * 100
        assert abs(df.loc[0, "yield_on_price"] - 250 / 10000 * 100) < 1e-6

    def test_total_dividend(self, enriched_df):
        df = enriched_df.copy()
        df["total_dividend"] = (df["div_rate"] * df["Available Lot"] * 100).astype(int)
        assert df.loc[0, "total_dividend"] == 250_000

    def test_annual_dividend_sum(self, enriched_df):
        df = enriched_df.copy()
        df["total_dividend"] = df["div_rate"] * df["Available Lot"] * 100
        assert df["total_dividend"].sum() == 650_000

    def test_achieve_percentage_positive(self, enriched_df):
        df = enriched_df.copy()
        df["total_dividend"] = df["div_rate"] * df["Available Lot"] * 100
        pct = df["total_dividend"].sum() / 240 * 100 / 1_000_000
        assert pct > 0

    def test_market_delta(self, enriched_df):
        df = enriched_df.copy()
        df["total_invested"]  = df["Available Lot"] * df["Average Price"] * 100
        current_value = (df["Available Lot"] * df["last_price"] * 100).sum()
        total_invested = df["total_invested"].sum()
        delta = current_value - total_invested
        bbca_gain = 10 * (10000 - 9500) * 100
        tlkm_loss = 20 * (3800 - 3600) * 100
        assert abs(delta - (bbca_gain - tlkm_loss)) < 1
