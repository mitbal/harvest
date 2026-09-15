# tests/apps/conftest.py
"""
Session-scoped stub installation for the apps test package.

All Streamlit, Redis, and third-party module stubs are installed ONCE here,
before any app test module is imported.  Individual test files should NOT
re-install stubs themselves.
"""
import json
import logging
import sys
import types as _types
from unittest.mock import MagicMock

import pandas as pd


def _build_st_stub():
    """Return a MagicMock that mimics the Streamlit API needed by app modules."""
    st = MagicMock(name="streamlit")

    # Decorators must be transparent pass-throughs
    def _make_cache_dec(*a, **kw):
        if len(a) == 1 and callable(a[0]) and not kw:
            return a[0]
        return lambda fn: fn

    st.cache_data     = _make_cache_dec
    st.cache_resource = _make_cache_dec

    st.session_state = {
        "porto_df": pd.DataFrame({
            "Symbol": ["BBCA", "TLKM"],
            "Available Lot": [10.0, 20.0],
            "Average Price": [9500.0, 3800.0]
        })
    }  # pre-populate to prevent early-exit
    _qp = {}
    st.query_params = _qp

    # Layout helpers — return a properly sized list so tuple unpacking works
    def _columns(spec, **kwargs):
        n = len(spec) if hasattr(spec, "__len__") else spec
        cols = []
        for _ in range(n):
            col = MagicMock()
            col.radio = _radio
            col.selectbox = _selectbox
            col.number_input = _number_input
            col.toggle = _toggle
            col.checkbox = _checkbox
            col.slider = _slider
            col.text_input = _text_input
            col.date_input = _date_input
            col.multiselect = _multiselect
            col.button = _button
            cols.append(col)
        return cols

    st.columns = _columns

    # Input element stubs to return valid default/current values during import
    def _radio(label, options, index=0, **kwargs):
        if options and index < len(options):
            return options[index]
        return None

    def _selectbox(label, options, index=0, **kwargs):
        if options and index < len(options):
            return options[index]
        return None

    def _number_input(label, min_value=None, max_value=None, value=0.0, **kwargs):
        return value

    def _toggle(label, value=False, **kwargs):
        return value

    def _checkbox(label, value=False, **kwargs):
        return value

    def _slider(label, min_value=None, max_value=None, value=0.0, **kwargs):
        return value

    def _text_input(label, value="", **kwargs):
        return value

    def _date_input(label, value=None, **kwargs):
        import datetime
        if value is None:
            return datetime.date.today()
        return value

    def _multiselect(label, options, default=None, **kwargs):
        if default:
            return default
        if options:
            return options[:2] if len(options) >= 2 else options
        return []

    def _button(label, **kwargs):
        return False

    def _segmented_control(label, options, default=None, **kwargs):
        if default in options:
            return default
        return options[0] if options else None

    def _data_editor(data, **kwargs):
        return data

    st.radio = _radio
    st.selectbox = _selectbox
    st.number_input = _number_input
    st.toggle = _toggle
    st.checkbox = _checkbox
    st.slider = _slider
    st.text_input = _text_input
    st.date_input = _date_input
    st.multiselect = _multiselect
    st.button = _button
    st.segmented_control = _segmented_control
    st.data_editor = _data_editor

    st.sidebar = MagicMock()
    st.sidebar.radio = _radio
    st.sidebar.selectbox = _selectbox
    st.sidebar.toggle = _toggle
    st.sidebar.checkbox = _checkbox
    st.sidebar.number_input = _number_input
    st.sidebar.slider = _slider
    st.sidebar.text_input = _text_input
    st.sidebar.date_input = _date_input
    st.sidebar.multiselect = _multiselect
    st.sidebar.data_editor = _data_editor

    _df_mock = MagicMock()
    _df_mock.selection = {"rows": []}
    st.dataframe = MagicMock(return_value=_df_mock)
    st.sidebar.dataframe = MagicMock(return_value=_df_mock)

    # No-op display calls
    for attr in ("title", "caption", "set_page_config", "html", "markdown",
                 "subheader", "divider", "info", "error", "warning",
                 "success", "write", "altair_chart"):
        setattr(st, attr, MagicMock())

    st.stop = MagicMock()  # no-op — module-level st.stop() must NOT raise

    # Context managers
    def _ctx_mgr(*a, **kw):
        m = MagicMock()
        m.__enter__ = MagicMock(return_value=MagicMock())
        m.__exit__ = MagicMock(return_value=False)
        return m

    st.container    = _ctx_mgr
    st.expander     = _ctx_mgr
    st.form         = _ctx_mgr
    def _tabs(tabs_list, **kwargs):
        return [MagicMock() for _ in range(len(tabs_list))]
    st.tabs         = _tabs

    st.secrets = {
        "connections": {
            "supabase": {"EMAIL_ADDRESS": "x@x.com", "PASSWORD": "secret"}
        }
    }

    _user = MagicMock()
    _user.is_logged_in = False
    _user.name = "Test User"
    _user.email = "test@example.com"
    st.user = _user

    st.connection = MagicMock(return_value=MagicMock())

    return st


def _build_real_logger(name="test"):
    """Return an actual Python logger so .info / .error / .warning work."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


# ── Install everything before any test module imports ──────────────────────

_st_stub = _build_st_stub()
sys.modules["streamlit"] = _st_stub

for _sub in [
    "streamlit.components",
    "streamlit.components.v1",
    "streamlit.runtime",
    "streamlit.runtime.scriptrunner",
]:
    sys.modules.setdefault(_sub, MagicMock(name=_sub))

# Third-party packages used inside app modules
for _pkg in ["redis", "streamlit_echarts5", "supabase",
             "st_supabase_connection", "st_vortree", "lesley"]:
    sys.modules.setdefault(_pkg, MagicMock(name=_pkg))

# Stub harvest.plot so it doesn't pull in streamlit_echarts5 / PIL etc.
class _PlotStubModule(_types.ModuleType):
    def __getattr__(self, name):
        return MagicMock()

_plot_stub = _PlotStubModule("harvest.plot")
sys.modules["harvest.plot"] = _plot_stub

# Stub pythonjsonlogger used by harvest.utils
_json_logger_pkg = _types.ModuleType("pythonjsonlogger")
_json_logger_mod = _types.ModuleType("pythonjsonlogger.jsonlogger")


class _FakeJsonFormatter(logging.Formatter):
    def __init__(self, *a, **kw):
        super().__init__()


_json_logger_mod.JsonFormatter = _FakeJsonFormatter
sys.modules["pythonjsonlogger"] = _json_logger_pkg
sys.modules["pythonjsonlogger.jsonlogger"] = _json_logger_mod

# Make harvest.utils return a real logger so .info/.error don't crash
import harvest.utils as _hu  # noqa: E402 - stubs must be installed before import
_hu.setup_logging = lambda name, level=logging.INFO: _build_real_logger(name)

# Redis client stub — must have a working .get() method
def _mock_redis_get(key):
    if "years" in key:
        return '[{"year": 2025}, {"year": 2026}]'
    elif "score" in key:
        return json.dumps([
            {
                "symbol": "BBCA.JK",
                "sector": "Financials",
                "industry": "Banks",
                "mktCap": 5e14,
                "yield": 2.5,
                "lastDiv": 250,
                "avgFlatAnnualDivIncrease": 10,
                "numDividendYear": 15,
                "positiveYear": 12,
                "numOfYear": 20,
                "maximumCutPct": -20,
                "max10CutPct": -10,
                "peRatio": 15.0,
                "psRatio": 3.0,
                "revenueGrowth": 10.0,
                "netIncomeGrowth": 12.0,
                "medianProfitMargin": 25.0,
                "earningTTM": 1e13,
                "revenueTTM": 4e13,
                "revenueGrowthTTM": 8.0,
                "netIncomeGrowthTTM": 10.0,
                "price": 9500,
                "changes": 100,
                "beta": 0.8,
                "return_7d": 0.01,
                "return_1m": 0.03,
                "return_1y": 0.15,
                "return_10y": 2.0,
                "total_return_1y": 0.18,
                "total_return_10y": 2.5,
                "is_syariah": False
            },
            {
                "symbol": "TLKM.JK",
                "sector": "Telecom",
                "industry": "Telecom",
                "mktCap": 2e14,
                "yield": 4.0,
                "lastDiv": 200,
                "avgFlatAnnualDivIncrease": 5,
                "numDividendYear": 10,
                "positiveYear": 8,
                "numOfYear": 15,
                "maximumCutPct": -15,
                "max10CutPct": -5,
                "peRatio": 12.0,
                "psRatio": 2.5,
                "revenueGrowth": 5.0,
                "netIncomeGrowth": 4.0,
                "medianProfitMargin": 20.0,
                "earningTTM": 8e12,
                "revenueTTM": 4e13,
                "revenueGrowthTTM": 3.0,
                "netIncomeGrowthTTM": 2.0,
                "price": 3800,
                "changes": -50,
                "beta": 0.9,
                "return_7d": -0.02,
                "return_1m": -0.01,
                "return_1y": 0.08,
                "return_10y": 1.5,
                "total_return_1y": 0.10,
                "total_return_10y": 2.0,
                "is_syariah": False
            },
            {
                "symbol": "STUB.JK",
                "sector": "Financials",
                "industry": "Banks",
                "mktCap": 5e14,
                "yield": 2.5,
                "lastDiv": 250,
                "avgFlatAnnualDivIncrease": 10,
                "numDividendYear": 15,
                "positiveYear": 12,
                "numOfYear": 20,
                "maximumCutPct": -20,
                "max10CutPct": -10,
                "peRatio": 15.0,
                "psRatio": 3.0,
                "revenueGrowth": 10.0,
                "netIncomeGrowth": 12.0,
                "medianProfitMargin": 25.0,
                "earningTTM": 1e13,
                "revenueTTM": 4e13,
                "revenueGrowthTTM": 8.0,
                "netIncomeGrowthTTM": 10.0,
                "price": 9500,
                "changes": 100,
                "beta": 0.8,
                "return_7d": 0.01,
                "return_1m": 0.03,
                "return_1y": 0.15,
                "return_10y": 2.0,
                "total_return_1y": 0.18,
                "total_return_10y": 2.5,
                "is_syariah": False
            }
        ])
    elif "div_cal" in key:
        return json.dumps([
            {"date": "2025-01-10", "symbol": "BBCA.JK", "adjDividend": 250, "price": 9500},
            {"date": "2025-05-12", "symbol": "TLKM.JK", "adjDividend": 200, "price": 3800}
        ])
    return None

_redis_client = MagicMock()
_redis_client.get = MagicMock(side_effect=_mock_redis_get)
sys.modules["redis"].from_url = MagicMock(return_value=_redis_client)
