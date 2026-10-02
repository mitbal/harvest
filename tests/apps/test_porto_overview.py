"""
tests/apps/test_porto_overview.py
Data loading in the Portfolio Analytics page: Redis profiles, the live FMP
fallback, dividend history fetching and the Supabase retry.

Portfolio maths lives in harvest/portfolio.py and is tested in tests/test_portfolio.py.
Stubs are installed by tests/apps/conftest.py.
"""
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


with patch.dict("os.environ", {"REDIS_URL": "redis://stub", "FMP_API_KEY": "stub"}):
    import apps.porto.porto_overview as po


def _redis_with(rows):
    client = MagicMock()
    client.get.return_value = json.dumps(rows)
    return client


BBCA_PROFILE = {"symbol": "BBCA.JK", "price": 9500, "sector": "Financials", "lastDiv": 250}


# ══════════════════════════════════════════════════════════════════════════════

class TestCompanyProfileData:

    @pytest.fixture(autouse=True)
    def redis_env(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://stub")

    def test_excludes_symbols_without_market_data(self):
        portfolio = pd.DataFrame({
            "Symbol": ["BBCA", "MISSING"],
            "Available Lot": [10, 5],
            "Average Price": [9000, 1000],
        })

        with patch.object(po, "connect_redis", return_value=_redis_with([BBCA_PROFILE])), \
             patch.object(po.hd, "get_company_profile", return_value=pd.DataFrame()):
            result, unknown = po.get_company_profile_data(portfolio)

        assert result["Symbol"].tolist() == ["BBCA"]
        assert unknown == ["MISSING"]

    def test_requires_redis_url(self, monkeypatch):
        monkeypatch.delenv("REDIS_URL")
        portfolio = pd.DataFrame({"Symbol": ["BBCA"], "Available Lot": [10], "Average Price": [9000]})
        with pytest.raises(EnvironmentError, match="REDIS_URL"):
            po.get_company_profile_data(portfolio)

    def test_fetches_live_profiles_for_symbols_missing_from_cache(self):
        portfolio = pd.DataFrame({
            "Symbol": ["BNGA", "BBRI"],
            "Available Lot": [760, 1500],
            "Average Price": [1360.71, 3082.4],
        })
        live_profiles = pd.DataFrame({
            "price": [1800, 4200],
            "sector": ["Financial Services", "Financial Services"],
            "lastDiv": [122.84, 343.4],
        }, index=pd.Index(["BNGA.JK", "BBRI.JK"], name="symbol"))

        with patch.object(po, "connect_redis", return_value=_redis_with([BBCA_PROFILE])), \
             patch.object(po.hd, "get_company_profile", return_value=live_profiles) as get_profile, \
             patch.object(po.hd, "get_dividend_history_single_stock_dag", return_value=None):
            result, unknown = po.get_company_profile_data(portfolio)

        get_profile.assert_called_once_with(["BNGA.JK", "BBRI.JK"])
        assert result["Symbol"].tolist() == ["BNGA", "BBRI"]
        assert result["last_price"].tolist() == [1800, 4200]
        # No dividend history: keep FMP's figure
        assert result["div_rate"].tolist() == [122.84, 343.4]
        assert unknown == []

    def test_live_profiles_use_finalized_fiscal_year_dividend(self):
        portfolio = pd.DataFrame({
            "Symbol": ["BNGA"], "Available Lot": [10], "Average Price": [1500],
        })
        live_profiles = pd.DataFrame(
            {"price": [1800], "sector": ["Financial Services"], "lastDiv": [60.0]},
            index=pd.Index(["BNGA.JK"], name="symbol"),
        )
        history = pd.DataFrame({
            "date": ["2025-09-01", "2026-04-20"],
            "adjDividend": [50.0, 110.0],
            "fiscal_year": [2025, 2025],
            "dividend_type": ["interim", "final"],
        })

        with patch.object(po, "connect_redis", return_value=_redis_with([BBCA_PROFILE])), \
             patch.object(po.hd, "get_company_profile", return_value=live_profiles), \
             patch.object(po.hd, "get_dividend_history_single_stock_dag", return_value=history), \
             patch.object(po.hd, "calc_latest_finalized_dividend_sum", return_value=160.0) as calc:
            result, _ = po.get_company_profile_data(portfolio)

        calc.assert_called_once()
        assert result["div_rate"].tolist() == [160.0]

    def test_live_profile_failure_keeps_cached_holdings(self):
        portfolio = pd.DataFrame({
            "Symbol": ["BBCA", "BNGA"], "Available Lot": [10, 5], "Average Price": [9000, 1500],
        })

        with patch.object(po, "connect_redis", return_value=_redis_with([BBCA_PROFILE])), \
             patch.object(po.hd, "get_company_profile", side_effect=RuntimeError("FMP down")):
            result, unknown = po.get_company_profile_data(portfolio)

        assert result["Symbol"].tolist() == ["BBCA"]
        assert unknown == ["BNGA"]


class TestDividendData:

    @staticmethod
    def _history(amount):
        return pd.DataFrame({
            "date": ["2025-04-01", "2025-06-01"],
            "adjDividend": [amount, 999],
            "dividend_type": ["final", "special"],
        })

    def test_failure_does_not_discard_other_stocks(self):
        portfolio = pd.DataFrame({"Symbol": ["FAIL", "BBCA"]})

        def fetch(stock):
            if stock == "FAIL.JK":
                raise RuntimeError("network error")
            return self._history(250)

        with patch.object(po.hd, "get_dividend_history_single_stock_dag", side_effect=fetch):
            result = po.get_dividend_data(portfolio)

        assert list(result) == ["BBCA.JK"]

    def test_keeps_portfolio_order_and_drops_special_dividends(self):
        symbols = ["BBCA", "TLKM", "ASII", "BBRI", "UNVR", "PTBA", "ITMG", "ADRO", "HEXA"]
        amounts = {f"{s}.JK": float(i) for i, s in enumerate(symbols)}

        with patch.object(
            po.hd,
            "get_dividend_history_single_stock_dag",
            side_effect=lambda stock: self._history(amounts[stock]),
        ):
            result = po.get_dividend_data(pd.DataFrame({"Symbol": symbols}))

        assert list(result) == [f"{s}.JK" for s in symbols]
        for stock, history in result.items():
            assert history["adjDividend"].tolist() == [amounts[stock]]


class TestRunDb:

    def test_reconnects_once_after_failure(self):
        stale, fresh = MagicMock(name="stale"), MagicMock(name="fresh")
        get_conn = MagicMock(side_effect=[stale, fresh])
        get_conn.clear = MagicMock()

        def operation(conn):
            if conn is stale:
                raise RuntimeError("JWT expired")
            return "ok"

        with patch.object(po, "get_db_connection", get_conn):
            assert po.run_db(operation) == "ok"

        get_conn.clear.assert_called_once()

    def test_raises_when_retry_also_fails(self):
        get_conn = MagicMock(return_value=MagicMock())
        get_conn.clear = MagicMock()

        with patch.object(po, "get_db_connection", get_conn):
            with pytest.raises(RuntimeError, match="down"):
                po.run_db(MagicMock(side_effect=RuntimeError("down")))
