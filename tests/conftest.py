"""
conftest.py — shared pytest fixtures for the entire test suite.

IMPORTANT: This file also installs all module stubs (Streamlit, Redis, etc.)
at collection time so that app modules can be imported without live
infrastructure.  This block must run before any test or app module is loaded.
"""
import logging
import sys
import types
from unittest.mock import MagicMock

import datetime
import pytest
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Module-level stub installation (runs at pytest collection time)
# ─────────────────────────────────────────────────────────────────────────────

def _build_st_stub():
    st = MagicMock(name="streamlit")

    # cache decorators: handle @st.cache_data (no parens) AND @st.cache_data(ttl=x)
    def _make_cache_dec(*a, **kw):
        # Called as @st.cache_data(fn) — bare decorator
        if len(a) == 1 and callable(a[0]) and not kw:
            return a[0]
        # Called as @st.cache_data(ttl=60) — with keyword args; return a decorator
        return lambda fn: fn

    st.cache_data     = _make_cache_dec
    st.cache_resource = _make_cache_dec

    st.session_state  = {}
    _qp = MagicMock()
    _qp.get = lambda key, default="": default
    _qp.__contains__ = lambda s, k: False
    _qp.__setitem__  = lambda s, k, v: None
    _qp.__delitem__  = lambda s, k: None
    st.query_params  = _qp

    def _columns(spec, **kwargs):
        n = len(spec) if hasattr(spec, "__len__") else spec
        return [MagicMock() for _ in range(n)]

    st.columns = _columns

    st.sidebar = MagicMock()
    st.sidebar.radio    = MagicMock(return_value="Indonesian Stock (JKSE)")
    st.sidebar.toggle   = MagicMock(return_value=False)
    st.sidebar.selectbox = MagicMock(return_value="JKSE")

    for attr in ("title", "caption", "set_page_config", "html", "markdown",
                 "subheader", "divider", "info", "error", "warning",
                 "success", "write", "dataframe", "altair_chart", "stop",
                 "button", "checkbox", "selectbox", "number_input", "slider",
                 "color_picker", "radio", "text_input", "text_area"):
        setattr(st, attr, MagicMock())

    def _ctx_mgr(*a, **kw):
        m = MagicMock()
        m.__enter__ = MagicMock(return_value=MagicMock())
        m.__exit__  = MagicMock(return_value=False)
        return m

    st.container = _ctx_mgr
    st.expander  = _ctx_mgr
    st.form      = _ctx_mgr
    st.tabs = MagicMock(return_value=[MagicMock() for _ in range(6)])

    st.secrets = {
        "connections": {
            "supabase": {"EMAIL_ADDRESS": "x@x.com", "PASSWORD": "secret"}
        }
    }
    _user = MagicMock()
    _user.is_logged_in = False
    _user.name  = "Test User"
    _user.email = "test@example.com"
    st.user = _user
    st.connection = MagicMock(return_value=MagicMock())
    return st


# Install streamlit stub
_st_stub = _build_st_stub()
sys.modules["streamlit"] = _st_stub

for _sub in [
    "streamlit.components",
    "streamlit.components.v1",
    "streamlit.runtime",
    "streamlit.runtime.scriptrunner",
]:
    sys.modules.setdefault(_sub, MagicMock(name=_sub))

# Third-party packages
for _pkg in ["redis", "streamlit_echarts5", "supabase",
             "st_supabase_connection", "st_vortree", "lesley"]:
    sys.modules.setdefault(_pkg, MagicMock(name=_pkg))

# Stub harvest.plot to avoid the streamlit_echarts5 → PIL import chain
_plot_stub = types.ModuleType("harvest.plot")
sys.modules["harvest.plot"] = _plot_stub

# Stub pythonjsonlogger with a real Formatter so setup_logging works
_json_logger_pkg = types.ModuleType("pythonjsonlogger")
_json_logger_mod = types.ModuleType("pythonjsonlogger.jsonlogger")


class _FakeJsonFormatter(logging.Formatter):
    def __init__(self, *a, **kw):
        super().__init__()


_json_logger_mod.JsonFormatter = _FakeJsonFormatter
sys.modules["pythonjsonlogger"] = _json_logger_pkg
sys.modules["pythonjsonlogger.jsonlogger"] = _json_logger_mod

# Patch harvest.utils.setup_logging to return a real logger
import harvest.utils as _hu


def _real_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


_hu.setup_logging = _real_logger

# Redis client stub with a working .get() method
_redis_client = MagicMock()
_redis_client.get = MagicMock(return_value=None)
sys.modules["redis"].from_url = MagicMock(return_value=_redis_client)



# ─────────────────────────────────────────────────────────────────────────────
# Price data
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def price_df():
    """5 years of daily close prices for a single stock."""
    dates = pd.date_range("2018-01-01", "2022-12-31", freq="B")
    np.random.seed(42)
    close = np.linspace(1000, 2000, len(dates)) + np.random.normal(0, 20, len(dates))
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": close})


@pytest.fixture
def price_df_short():
    """Only 1 year of price data — useful for edge-case tests."""
    dates = pd.date_range("2022-01-01", "2022-12-31", freq="B")
    close = np.linspace(1000, 1200, len(dates))
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": close})


@pytest.fixture
def multi_stock_price_df():
    """Long-format price DataFrame for two symbols."""
    dates = pd.date_range("2021-01-01", "2022-12-31", freq="B")
    rows = []
    for sym, base in [("AAAA", 1000), ("BBBB", 500)]:
        for i, d in enumerate(dates):
            rows.append({"symbol": sym, "date": d, "close": base + i * 0.5})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Dividend data
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def div_df():
    """Preprocessed dividend DataFrame (output of preprocess_div)."""
    years = list(range(2015, 2023))
    divs  = [50, 60, 55, 70, 80, 0, 90, 100]
    return pd.DataFrame({"year": years, "adjDividend": divs})


@pytest.fixture
def div_raw_df():
    """Raw dividend history as returned by FMP (date strings, adjDividend)."""
    dates = ["2022-05-10", "2022-11-08", "2021-05-11", "2021-11-09",
             "2020-05-12", "2019-05-14", "2018-05-15"]
    divs  = [100, 50, 90, 45, 80, 70, 60]
    return pd.DataFrame({"date": dates, "adjDividend": divs})


# ─────────────────────────────────────────────────────────────────────────────
# Financial (income statement) data
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fin_df():
    """8 quarters of quarterly income-statement data (IDR)."""
    dates = pd.date_range("2020-01-01", periods=8, freq="QE").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "date":             dates.tolist(),
        "calendarYear":     [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021],
        "period":           ["Q1", "Q2", "Q3", "Q4"] * 2,
        "revenue":          [1_000_000, 1_100_000, 1_200_000, 1_300_000,
                             1_400_000, 1_500_000, 1_600_000, 1_700_000],
        "netIncome":        [100_000, 110_000, 120_000, 130_000,
                             140_000, 150_000, 160_000, 170_000],
        "eps":              [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        "reportedCurrency": ["IDR"] * 8,
    })


@pytest.fixture
def fin_df_usd():
    """8 quarters of quarterly income-statement data (USD)."""
    dates = pd.date_range("2020-01-01", periods=8, freq="QE").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "date":             dates.tolist(),
        "calendarYear":     [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021],
        "period":           ["Q1", "Q2", "Q3", "Q4"] * 2,
        "revenue":          [100, 110, 120, 130, 140, 150, 160, 170],
        "netIncome":        [10, 11, 12, 13, 14, 15, 16, 17],
        "eps":              [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
        "reportedCurrency": ["USD"] * 8,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio data
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def porto_df():
    """Simple 3-stock portfolio DataFrame."""
    return pd.DataFrame({
        "Symbol":        ["BBCA", "TLKM", "ASII"],
        "Available Lot": [10, 20, 5],
        "Average Price": [9500, 3800, 6000],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Div-score universe DataFrame (mirrors what Redis returns)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def universe_df():
    """Small universe DataFrame similar to the one stored in Redis."""
    stocks = ["BBCA.JK", "TLKM.JK", "ASII.JK"]
    return pd.DataFrame({
        "stock":                   stocks,
        "sector":                  ["Financials", "Telecom", "Automotive"],
        "industry":                ["Banks", "Telecom", "Auto"],
        "mktCap":                  [5e14, 2e14, 1e14],
        "yield":                   [2.5, 4.0, 3.0],
        "lastDiv":                 [250, 200, 180],
        "avgFlatAnnualDivIncrease":[10, 5, 8],
        "numDividendYear":         [15, 10, 8],
        "positiveYear":            [12, 8, 6],
        "numOfYear":               [20, 15, 12],
        "maximumCutPct":           [-20, -15, -30],
        "max10CutPct":             [-10, -5, -15],
        "peRatio":                 [15.0, 12.0, 10.0],
        "psRatio":                 [3.0, 2.5, 1.5],
        "revenueGrowth":           [10.0, 5.0, 8.0],
        "netIncomeGrowth":         [12.0, 4.0, 7.0],
        "medianProfitMargin":      [25.0, 20.0, 15.0],
        "earningTTM":              [1e13, 8e12, 5e12],
        "revenueTTM":              [4e13, 4e13, 3e13],
        "revenueGrowthTTM":        [8.0, 3.0, 5.0],
        "netIncomeGrowthTTM":      [10.0, 2.0, 4.0],
        "price":                   [9500, 3800, 6000],
        "changes":                 [100, -50, 200],
        "beta":                    [0.8, 0.9, 1.1],
        "return_7d":               [0.01, -0.02, 0.03],
        "return_1m":               [0.03, -0.01, 0.05],
        "return_1y":               [0.15,  0.08, 0.20],
        "return_10y":              [2.0,   1.5,  3.0],
        "total_return_1y":         [0.18,  0.10, 0.22],
        "total_return_10y":        [2.5,   2.0,  3.5],
        "is_syariah":              [False, False, True],
    }).set_index("stock")
