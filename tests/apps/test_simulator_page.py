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


class TestSubmissions:

    def test_stock_submission_keeps_raw_ticker_and_casts(self):
        s = sim.stock_submission(" bbca ", 120, 2014.0, 2024, 1)
        assert s == {"stock": " bbca ", "value_mio": 120, "start": 2014, "end": 2024, "drip": True}

    def test_stock_submission_none_ticker(self):
        assert sim.stock_submission(None, 1, 2014, 2015, False)["stock"] == ""

    def test_default_stock_submission(self):
        s = sim.default_stock_submission()
        assert s == {"stock": "BBCA.JK", "value_mio": 120, "start": 2014,
                     "end": sim.YEAR_MAX, "drip": True}

    def test_default_portfolio_submission(self):
        s = sim.default_portfolio_submission()
        assert s["rows"] == [("BJTM.JK", 60.0), ("SMSM.JK", 60.0)]
        assert (s["start"], s["end"], s["drip"]) == (2014, sim.YEAR_MAX, True)

    def test_portfolio_submission_cleans_blank_cells(self):
        edited = pd.DataFrame({
            "Ticker": ["BBCA.JK", None, "TLKM"],
            "Investment (Mio IDR)": [10.0, 5.0, float("nan")],
        }, index=[0, 3, 7])
        s = sim.portfolio_submission(edited, 2015, 2020, False)
        assert s["rows"] == [("BBCA.JK", 10.0), (None, 5.0), ("TLKM", None)]
        assert (s["start"], s["end"], s["drip"]) == (2015, 2020, False)

    def test_allocations_frame_round_trips_editor_output(self):
        rows = [("BJTM.JK", 60.0), (None, None)]
        frame = sim.allocations_frame(rows)
        assert list(frame.columns) == sim.ALLOCATION_COLUMNS
        assert frame["Investment (Mio IDR)"].dtype == float
        assert sim.portfolio_submission(frame, 2014, 2015, True)["rows"] == rows

    def test_allocations_frame_empty(self):
        frame = sim.allocations_frame([])
        assert frame.empty and list(frame.columns) == sim.ALLOCATION_COLUMNS

    def test_allocation_rows_scale_to_rupiah(self):
        rows = sim.allocation_rows([("BJTM.JK", 60.0), ("SMSM.JK", 1.5)])
        assert rows == [("BJTM.JK", 60_000_000.0), ("SMSM.JK", 1_500_000.0)]

    def test_allocation_rows_blank_investment_stays_nan(self):
        (ticker, value), = sim.allocation_rows([("BJTM.JK", None)])
        assert ticker == "BJTM.JK" and pd.isna(value)


class TestDripAdvantage:

    def test_positive(self):
        adv = sim.drip_advantage(105_400_000, 100_000_000)
        assert adv["value"] == "IDR 5,400,000"
        assert adv["note"] == "+5.4% vs no DRIP"
        assert adv["tone"] == "up"

    def test_negative(self):
        adv = sim.drip_advantage(98_800_000, 100_000_000)
        assert adv["value"] == "-IDR 1,200,000"
        assert adv["note"] == "-1.2% vs no DRIP"
        assert adv["tone"] == "down"

    def test_zero(self):
        adv = sim.drip_advantage(100_000_000, 100_000_000)
        assert adv["value"] == "IDR 0"
        assert adv["note"] == "0.0% vs no DRIP"
        assert adv["tone"] == "muted"

    def test_zero_baseline_has_no_percentage(self):
        adv = sim.drip_advantage(1_000, 0)
        assert adv["pct"] is None
        assert adv["note"] == ""
        assert adv["value"] == "IDR 1,000"

    def test_kpi_markup(self):
        markup = sim.drip_advantage_kpi(105_400_000, 100_000_000)
        assert "DRIP advantage" in markup
        assert "pdx-tone-up" in markup


class TestStockList:

    def test_idx_ticker_options_normalises_sorts_and_dedupes(self):
        frame = pd.DataFrame({"symbol": ["TLKM.JK", "bbca", "BBCA.JK", None, "AAPL", "", "ASII.JK"]})
        # "AAPL" becomes AAPL.JK (normalize_idx_ticker only checks the suffix)
        assert sim.idx_ticker_options(frame) == ("AAPL.JK", "ASII.JK", "BBCA.JK", "TLKM.JK")

    def test_idx_ticker_options_drops_other_markets(self):
        frame = pd.DataFrame({"stock": ["BBCA.JK", "MSFT.US"]})
        assert sim.idx_ticker_options(frame) == ("BBCA.JK",)

    @pytest.mark.parametrize("frame", [None, pd.DataFrame()])
    def test_idx_ticker_options_empty(self, frame):
        assert sim.idx_ticker_options(frame) == ()

    def test_select_options_finds_submitted_ticker(self):
        options, index = sim.stock_select_options(("ASII.JK", "BBCA.JK"), "BBCA.JK")
        assert options == ["ASII.JK", "BBCA.JK"] and index == 1

    def test_select_options_normalises_submitted_ticker(self):
        options, index = sim.stock_select_options(("ASII.JK", "BBCA.JK"), " bbca ")
        assert options[index] == "BBCA.JK" and len(options) == 2

    def test_select_options_adds_unknown_ticker(self):
        options, index = sim.stock_select_options(("ASII.JK", "BBCA.JK"), "zzzz")
        assert options == ["ASII.JK", "BBCA.JK", "ZZZZ.JK"] and index == 2

    def test_select_options_keeps_rejected_ticker_as_typed(self):
        options, index = sim.stock_select_options(("BBCA.JK",), "MSFT.US")
        assert options[index] == "MSFT.US"

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_select_options_blank_has_no_selection(self, value):
        assert sim.stock_select_options(("BBCA.JK",), value) == (["BBCA.JK"], None)

    def test_load_idx_tickers_reads_screener_table(self):
        frame = pd.DataFrame({"symbol": ["BBCA.JK", "TLKM.JK"]})
        with patch.object(sim, "read_redis_frame", return_value=frame) as read:
            assert sim.load_idx_tickers() == ("BBCA.JK", "TLKM.JK")
        read.assert_called_once_with("div_score_jkse")

    @pytest.mark.parametrize("effect", [ConnectionError("down"), None])
    def test_load_idx_tickers_falls_back_to_empty(self, effect):
        kwargs = {"side_effect": effect} if effect else {"return_value": None}
        with patch.object(sim, "read_redis_frame", **kwargs):
            assert sim.load_idx_tickers() == ()
