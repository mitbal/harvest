"""
tests/apps/test_simulator_page.py
Tests for the Compounding Simulator page helpers.
Stubs are installed by tests/apps/conftest.py.
"""
from unittest.mock import patch

import pandas as pd
import pytest

import harvest.data as hd


# Importing the page runs the default (Fixed yield) mode only; the network-backed
# modes must not fetch anything.
with patch.dict("os.environ", {"REDIS_URL": "redis://stub", "FMP_API_KEY": "stub"}), \
        patch.object(hd, "get_dividend_history_single_stock") as _divs, \
        patch.object(hd, "get_daily_stock_price") as _prices:
    import apps.simulator.simulator as sim

    _IMPORT_FETCHED = _divs.called or _prices.called


def test_import_runs_no_network_fetch():
    assert not _IMPORT_FETCHED


class TestParseSimParam:

    def test_default_is_fixed(self):
        assert sim.parse_sim_param({}) == "fixed"

    @pytest.mark.parametrize("value", ["fixed", "stock", "portfolio"])
    def test_valid_values(self, value):
        assert sim.parse_sim_param({"sim": value}) == value

    def test_case_and_whitespace_insensitive(self):
        assert sim.parse_sim_param({"sim": " Stock "}) == "stock"

    @pytest.mark.parametrize("value", ["", "bogus", "1", "<script>"])
    def test_bad_value_falls_back_to_fixed(self, value):
        assert sim.parse_sim_param({"sim": value}) == "fixed"

    def test_every_mode_has_a_renderer(self):
        assert set(sim.RENDERERS) == set(sim.SIM_MODES)


class TestFixedYield:

    @pytest.fixture
    def default_run(self):
        initial = 120 * 1_000_000
        return hd.simulate_simple_compounding(initial, 10, 6.35 / 100), initial

    def test_summary_matches_old_metric_formulas(self, default_run):
        df, initial = default_run
        s = sim.fixed_yield_summary(df, initial)
        assert s["final_value"] == df["investment"].iloc[-1]
        assert s["total_income"] == df["returns"].sum()
        assert s["final_income"] == df["returns"].iloc[-1]
        assert s["yield_on_cost"] == pytest.approx(df["returns"].iloc[-1] / initial * 100)
        assert s["capital_multiple"] == pytest.approx(df["investment"].iloc[-1] / initial)

    def test_year_by_year_chains_start_and_end(self, default_run):
        df, initial = default_run
        table = sim.year_by_year_table(df, initial)
        assert list(table.columns) == ["year", "start_value", "income", "end_value", "yield_on_cost"]
        assert table["start_value"].iloc[0] == initial
        assert (table["start_value"].iloc[1:].values == table["end_value"].iloc[:-1].values).all()
        assert (table["start_value"] + table["income"] == table["end_value"]).all()
        assert table["yield_on_cost"].iloc[0] == pytest.approx(6.35, abs=0.01)

    def test_chart_data_splits_capital_and_reinvested(self, default_run):
        df, initial = default_run
        data = sim.asset_chart_data(df, initial)
        assert (data["capital"] == initial).all()
        assert (data["capital"] + data["reinvested"] == data["total"]).all()
        assert data["label"].tolist()[0] == "Y1"
        assert data["label"].tolist()[-1] == "Y10"

    def test_chart_builds(self, default_run):
        df, initial = default_run
        spec = sim.asset_value_chart(sim.asset_chart_data(df, initial)).to_dict()
        assert "layer" in spec


def test_chart_builds_for_extreme_long_label():
    df = hd.simulate_simple_compounding(10_000 * 1_000_000, 50, 0.999)
    data = sim.asset_chart_data(df, 10_000 * 1_000_000)
    assert len(data["total_label"].iloc[-1]) > 6
    spec = sim.asset_value_chart(data).to_dict()
    assert spec["layer"][1]["mark"]["align"] == "right"


@pytest.mark.parametrize("value,expected", [
    (222_090_000, "222M"),
    (12_500_000, "12.5M"),
    (1_000_000_000, "1B"),
    (1_430_000_000, "1.4B"),
    (2_100_000_000_000, "2.1T"),
    (950_000, "950,000"),
])
def test_short_idr(value, expected):
    assert sim.short_idr(value) == expected
