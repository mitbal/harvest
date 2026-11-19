"""Dividend Calendar: the year selectbox keeps every change, not only every other one."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest


REPO = Path(__file__).resolve().parents[1]
PAGE = str(REPO / "apps" / "calendar" / "calendar.py")
YEARS = [2023, 2024, 2025]


def _redis_data():
    data = {
        "div_cal_years_jkse": [{"year": y} for y in YEARS],
        "div_score_jkse": [{"symbol": "BBRI.JK", "is_syariah": False}, {"symbol": "TLKM.JK", "is_syariah": True}],
        "div_cal_years_sp500": [{"year": y} for y in YEARS],
        "div_score_sp500": [{"symbol": "AAPL"}],
    }
    for year in YEARS:
        data[f"div_cal_jkse_{year}"] = [
            {"date": f"{year}-{month:02d}-10", "symbol": symbol, "adjDividend": 100.0, "price": 5000.0}
            for month in range(1, 13) for symbol in ("BBRI.JK", "TLKM.JK")
        ]
        data[f"div_cal_sp500_{year}"] = [
            {"date": f"{year}-{month:02d}-10", "symbol": "AAPL", "adjDividend": 0.25, "price": 200.0}
            for month in range(1, 13)
        ]
    return {key: json.dumps(rows) for key, rows in data.items()}


@pytest.fixture
def app(monkeypatch):
    import streamlit as st

    monkeypatch.chdir(REPO)
    monkeypatch.setenv("REDIS_URL", "redis://stub")
    st.cache_data.clear()
    st.cache_resource.clear()
    fake = MagicMock()
    fake.get.side_effect = _redis_data().get
    with patch("redis.from_url", return_value=fake):
        yield AppTest.from_file(PAGE, default_timeout=60)
    st.cache_resource.clear()


def _year_box(app):
    return next(box for box in app.selectbox if box.label == "Based on year")


def _stock_list(app):
    return next(control for control in app.button_group if control.label == "Stock list")


def _syariah(app):
    return next(toggle for toggle in app.toggle if toggle.label == "Syariah only")


@pytest.mark.parametrize("view", ["monthly", "yearly", "tables"])
def test_year_changes_twice_in_a_row(app, view):
    app.query_params["view"] = view
    app.query_params["year"] = "2025"
    app.run()
    assert not app.exception
    assert _year_box(app).value == 2025

    for year in (2024, 2023):
        _year_box(app).set_value(year).run()
        assert not app.exception
        assert _year_box(app).value == year
        assert app.query_params["year"] == [str(year)]
        assert app.title[0].value == f"Dividend Calendar {year + 1}"


def test_defaults_leave_list_and_syariah_out_of_the_url(app):
    app.run()
    assert not app.exception
    assert _stock_list(app).value == "JKSE"
    assert _syariah(app).value is False
    assert "list" not in app.query_params and "syariah" not in app.query_params


def test_shared_link_opens_the_shared_selection(app):
    # What a jkse-2024-syariah share page redirects to
    app.query_params.update({"view": "yearly", "year": "2024", "syariah": "1"})
    app.run()
    assert not app.exception
    assert _stock_list(app).value == "JKSE"
    assert _syariah(app).value is True
    assert app.query_params["syariah"] == ["1"]

    # Turning it off takes it out of the URL
    _syariah(app).set_value(False).run()
    assert not app.exception
    assert "syariah" not in app.query_params


@pytest.mark.parametrize("list_param", ["sp500", "SP500"])
def test_list_param_opens_sp500_without_syariah(app, list_param):
    # syariah=1 is JKSE only, so it is dropped rather than carried over
    app.query_params.update({"view": "yearly", "year": "2024", "list": list_param, "syariah": "1"})
    app.run()
    assert not app.exception
    assert _stock_list(app).value == "S&P500"
    assert not any(toggle.label == "Syariah only" for toggle in app.toggle)
    assert app.query_params["list"] == ["sp500"]
    assert "syariah" not in app.query_params

    # Back to JKSE: list= goes away and the Syariah toggle starts off
    _stock_list(app).set_value("JKSE").run()
    assert not app.exception
    assert "list" not in app.query_params
    assert _syariah(app).value is False
