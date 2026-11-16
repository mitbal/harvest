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
        "div_score_jkse": [{"symbol": "BBRI.JK", "is_syariah": False}],
    }
    for year in YEARS:
        data[f"div_cal_jkse_{year}"] = [
            {"date": f"{year}-{month:02d}-10", "symbol": "BBRI.JK", "adjDividend": 100.0, "price": 5000.0}
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


@pytest.mark.parametrize("view", ["monthly", "yearly"])
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
