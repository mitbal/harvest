"""
tests/test_data.py
Unit tests for harvest/data.py — all pure-logic functions.

External calls (requests, FMP API, USD/IDR rate) are patched with
unittest.mock so the suite runs offline with zero infrastructure.
"""
import datetime
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

import harvest.data as hd


# ══════════════════════════════════════════════════════════════════════════════
# Market data sources
# ══════════════════════════════════════════════════════════════════════════════

class TestMarketDataSources:

    @patch("yfinance.Ticker")
    def test_yahoo_history_normalizes_prices_and_actions(self, ticker_mock):
        index = pd.date_range("2024-01-02", periods=2, freq="B", tz="Asia/Jakarta")
        ticker_mock.return_value.history.return_value = pd.DataFrame({
            "Close": [100.0, 101.0],
            "Adj Close": [99.0, 100.0],
            "Dividends": [0.0, 2.0],
        }, index=index)
        ticker_mock.return_value.history.return_value.index.name = "Date"

        result = hd.get_daily_stock_history_yahoo(
            "TEST.JK", start_from="2024-01-01", end_at="2024-01-31"
        )

        assert result.columns.tolist() == ["date", "close", "adjClose", "dividend"]
        assert result["date"].dt.tz is None
        assert result["dividend"].tolist() == [0.0, 2.0]
        ticker_mock.return_value.history.assert_called_once_with(
            start="2024-01-01",
            end="2024-02-01",
            auto_adjust=False,
            actions=True,
        )

    def test_fmp_dividend_error_does_not_print_api_key(self, capsys):
        response = MagicMock(status_code=429)
        error = hd.requests.exceptions.HTTPError(
            "quota error at https://example.test?apikey=secret", response=response
        )

        with patch("harvest.data.requests.get", side_effect=error):
            result = hd.get_dividend_history_single_stock_fmp("TEST", api_key="secret")

        assert result is None
        output = capsys.readouterr().out
        assert "HTTP 429" in output
        assert "secret" not in output


# ══════════════════════════════════════════════════════════════════════════════
# preprocess_div
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessDiv:

    def test_aggregates_interim_dividends(self):
        """Two payments in the same year should be summed."""
        raw = pd.DataFrame({
            "date": ["2020-05-10", "2020-11-10", "2021-05-10"],
            "adjDividend": [50, 50, 80],
        })
        result = hd.preprocess_div(raw)
        assert result.loc[result["year"] == 2020, "adjDividend"].values[0] == 100
        assert result.loc[result["year"] == 2021, "adjDividend"].values[0] == 80

    def test_fills_missing_years_with_zero(self):
        """Years with no dividend should be present and equal 0."""
        raw = pd.DataFrame({
            "date": ["2018-05-10", "2021-05-10"],
            "adjDividend": [50, 80],
        })
        result = hd.preprocess_div(raw)
        # 2019 and 2020 should have been inserted
        years_with_zero = result[result["adjDividend"] == 0]["year"].tolist()
        assert 2019 in years_with_zero
        assert 2020 in years_with_zero

    def test_single_year(self):
        """Single dividend entry should produce a single-row result."""
        raw = pd.DataFrame({
            "date": ["2022-05-10"],
            "adjDividend": [100],
        })
        result = hd.preprocess_div(raw)
        # Start year == end year → exactly one row
        assert len(result) >= 1
        assert result.loc[result["year"] == 2022, "adjDividend"].values[0] == 100

    def test_output_columns(self):
        raw = pd.DataFrame({
            "date": ["2021-05-10", "2022-05-10"],
            "adjDividend": [70, 80],
        })
        result = hd.preprocess_div(raw)
        assert "year" in result.columns
        assert "adjDividend" in result.columns


# ══════════════════════════════════════════════════════════════════════════════
# fiscal-year dividend freshness
# ══════════════════════════════════════════════════════════════════════════════

class TestDividendFreshness:

    def test_only_counts_requested_fiscal_year(self):
        dividends = pd.DataFrame({
            "date": ["2024-05-01", "2025-04-01", "2025-10-01"],
            "adjDividend": [80, 100, 40],
            "fiscal_year": [2023, 2024, 2024],
            "dividend_type": ["final", "final", "interim"],
        })

        result = hd.calc_fiscal_year_dividend_sum(
            dividends, fiscal_year=2024, as_of="2025-12-01"
        )

        assert result == 140

    def test_previous_fiscal_year_does_not_carry_forward(self):
        dividends = pd.DataFrame({
            "date": ["2025-04-01"],
            "adjDividend": [100],
            "fiscal_year": [2023],
            "dividend_type": ["final"],
        })

        result = hd.calc_fiscal_year_dividend_sum(
            dividends, fiscal_year=2024, as_of="2025-06-15"
        )

        assert result == 0

    def test_latest_finalized_year_is_allowed_during_reporting_grace(self):
        dividends = pd.DataFrame({
            "date": ["2025-05-01"],
            "adjDividend": [100],
            "fiscal_year": [2024],
            "dividend_type": ["final"],
        })

        assert hd.calc_latest_finalized_dividend_sum(
            dividends, as_of="2026-03-01"
        ) == 100
        assert hd.calc_latest_finalized_dividend_sum(
            dividends, as_of="2026-08-01"
        ) == 0

    def test_new_finalized_year_replaces_prior_year_during_grace(self):
        dividends = pd.DataFrame({
            "date": ["2025-05-01", "2026-03-15"],
            "adjDividend": [100, 120],
            "fiscal_year": [2024, 2025],
            "dividend_type": ["final", "final"],
        })

        assert hd.calc_latest_finalized_dividend_sum(
            dividends, as_of="2026-04-01"
        ) == 120

    def test_requires_final_and_excludes_special_dividend(self):
        dividends = pd.DataFrame({
            "date": ["2025-03-01", "2025-05-01", "2025-05-02"],
            "adjDividend": [25, 50, 200],
            "fiscal_year": [2024, 2024, 2024],
            "dividend_type": ["interim", "final", "special"],
        })

        result = hd.calc_fiscal_year_dividend_sum(
            dividends, fiscal_year=2024, as_of="2025-06-15"
        )

        assert result == 75

        without_final = dividends[dividends["dividend_type"] != "final"]
        assert hd.calc_fiscal_year_dividend_sum(
            without_final, fiscal_year=2024, as_of="2025-06-15"
        ) == 0

    def test_schedule_detects_an_overdue_dividend(self):
        dividends = pd.DataFrame({
            "date": ["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"],
            "adjDividend": [1, 1, 1, 1],
        })

        assert hd.is_dividend_schedule_current(dividends, as_of="2025-01-15")
        assert not hd.is_dividend_schedule_current(dividends, as_of="2025-04-15")


# ══════════════════════════════════════════════════════════════════════════════
# calc_div_stats
# ══════════════════════════════════════════════════════════════════════════════

class TestCalcDivStats:

    def test_returns_expected_keys(self, div_df):
        stats = hd.calc_div_stats(div_df)
        expected = {
            "maximum_cut_pct", "max_10y_cut_pct",
            "historical_mean_flat", "div_inc_2y_mean_flat",
            "div_inc_5y_mean_flat", "exponential_weighted_mean_flat",
            "historical_mean_pct", "div_inc_2y_mean_pct",
            "div_inc_5y_mean_pct", "exponential_weighted_mean_pct",
            "num_positive_year", "num_dividend_year", "pct_positive_year",
            "cagr_5y", "cagr_10y",
        }
        assert expected.issubset(set(stats.keys()))

    def test_empty_df_returns_zeros(self):
        empty = pd.DataFrame(columns=["year", "adjDividend"])
        stats = hd.calc_div_stats(empty)
        assert stats["maximum_cut_pct"] == 0
        assert stats["num_positive_year"] == 0

    def test_num_positive_year(self, div_df):
        stats = hd.calc_div_stats(div_df)
        # Must be a positive integer
        assert stats["num_positive_year"] > 0

    def test_pct_positive_year_in_range(self, div_df):
        stats = hd.calc_div_stats(div_df)
        assert 0 <= stats["pct_positive_year"] <= 100

    def test_cagr_none_when_insufficient_history(self):
        """Less than 6 years → cagr_5y should be None."""
        small = pd.DataFrame({
            "year": [2020, 2021, 2022],
            "adjDividend": [50, 60, 70],
        })
        stats = hd.calc_div_stats(small)
        assert stats["cagr_5y"] is None
        assert stats["cagr_10y"] is None

    def test_cagr_5y_calculated(self):
        """At least 6 data points with positive first dividend → cagr_5y not None."""
        years = list(range(2016, 2023))
        divs = [50, 55, 60, 65, 70, 75, 80]
        df = pd.DataFrame({"year": years, "adjDividend": divs})
        stats = hd.calc_div_stats(df)
        assert stats["cagr_5y"] is not None
        assert stats["cagr_5y"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# calc_div_score
# ══════════════════════════════════════════════════════════════════════════════

class TestCalcDivScore:

    def _make_df(self, **overrides):
        base = {
            "lastDiv": 200,
            "avgFlatAnnualDivIncrease": 10,
            "price": 5000,
            "numDividendYear": 10,
            "numOfYear": 15,
            "positiveYear": 8,
            "revenueGrowth": 0.05,
        }
        base.update(overrides)
        return pd.Series(base)

    def test_positive_score(self):
        s = self._make_df()
        score = hd.calc_div_score(s)
        assert score > 0

    def test_zero_last_div_gives_zero(self):
        s = self._make_df(lastDiv=0)
        score = hd.calc_div_score(s)
        assert score == 0


# ══════════════════════════════════════════════════════════════════════════════
# calc_growth_stats
# ══════════════════════════════════════════════════════════════════════════════

class TestCalcGrowthStats:

    def test_returns_expected_keys(self, fin_df):
        stats = hd.calc_growth_stats(fin_df, metric="revenue")
        assert "median_2y_revenue_growth" in stats
        assert "median_5y_revenue_growth" in stats
        assert "revenue_growth_TTM" in stats

    def test_positive_growth(self, fin_df):
        """Monotonically increasing revenue should yield positive growth."""
        stats = hd.calc_growth_stats(fin_df, metric="revenue")
        assert stats["revenue_growth_TTM"] > 0

    def test_net_income_metric(self, fin_df):
        stats = hd.calc_growth_stats(fin_df, metric="netIncome")
        assert "median_2y_netIncome_growth" in stats


# ══════════════════════════════════════════════════════════════════════════════
# calc_fin_stats
# ══════════════════════════════════════════════════════════════════════════════

class TestCalcFinStats:

    def test_returns_ttm_values(self, fin_df):
        stats = hd.calc_fin_stats(fin_df, target_currency="IDR")
        assert "earningTTM" in stats
        assert "revenueTTM" in stats
        assert "marginTTM" in stats

    def test_returns_growth_stats_keys(self, fin_df):
        """calc_fin_stats internally calls calc_growth_stats — verify those keys."""
        stats = hd.calc_fin_stats(fin_df, target_currency="IDR")
        assert "median_2y_revenue_growth" in stats
        assert "revenue_growth_TTM" in stats

    def test_median_profit_margin_present(self, fin_df):
        stats = hd.calc_fin_stats(fin_df, target_currency="IDR")
        assert "median_profit_margin" in stats
        # nanmedian of ~10% profit rows should be positive
        assert stats["median_profit_margin"] > 0

    @patch("harvest.data.get_usd_idr_rate", return_value=15_000)
    def test_usd_stats_keys_present(self, mock_rate, fin_df_usd):
        """USD financials: function should return all expected keys regardless."""
        stats = hd.calc_fin_stats(fin_df_usd, target_currency="IDR")
        for key in ("earningTTM", "revenueTTM", "marginTTM", "median_profit_margin"):
            assert key in stats

    def test_idr_returns_all_keys(self, fin_df):
        """IDR financials should return all expected keys."""
        stats = hd.calc_fin_stats(fin_df, target_currency="IDR")
        for key in ("earningTTM", "revenueTTM", "marginTTM", "median_profit_margin",
                    "median_2y_revenue_growth", "median_2y_netIncome_growth"):
            assert key in stats




# ══════════════════════════════════════════════════════════════════════════════
# calc_pe_history
# ══════════════════════════════════════════════════════════════════════════════

class TestCalcPeHistory:

    @patch("harvest.data.get_usd_idr_rate", return_value=15_000)
    def test_idr_currency(self, mock_rate, price_df, fin_df):
        result = hd.calc_pe_history(price_df, fin_df, currency="IDR")
        assert "pe" in result.columns
        assert len(result) > 0

    @patch("harvest.data.get_usd_idr_rate", return_value=15_000)
    def test_pe_positive_where_eps_positive(self, mock_rate, price_df, fin_df):
        result = hd.calc_pe_history(price_df, fin_df, currency="IDR")
        # PE should be positive when EPS is positive
        pos_eps = result.dropna(subset=["pe"])
        assert (pos_eps["pe"] > 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# calc_pe_stats
# ══════════════════════════════════════════════════════════════════════════════

class TestCalcPeStats:

    def _make_pe_df(self):
        dates = pd.date_range("2015-01-01", "2024-12-31", freq="B")
        pe = np.linspace(10, 20, len(dates))
        return pd.DataFrame({"date": dates, "pe": pe})

    def test_returns_expected_keys(self):
        pe_df = self._make_pe_df()
        stats = hd.calc_pe_stats(pe_df)
        for period in [1, 2, 3, 5, 10]:
            assert f"last_{period}y_mean" in stats
            assert f"last_{period}y_min_ci" in stats
            assert f"last_{period}y_max_ci" in stats

    def test_mean_within_range(self):
        pe_df = self._make_pe_df()
        stats = hd.calc_pe_stats(pe_df)
        assert 10 <= stats["last_1y_mean"] <= 20


# ══════════════════════════════════════════════════════════════════════════════
# simulate_simple_compounding
# ══════════════════════════════════════════════════════════════════════════════

class TestSimulateSimpleCompounding:

    def test_output_shape(self):
        result = hd.simulate_simple_compounding(
            initial_value=100_000_000, num_year=10, avg_yield=0.05
        )
        assert len(result) == 10
        assert "investment" in result.columns
        assert "returns" in result.columns

    def test_investment_grows(self):
        result = hd.simulate_simple_compounding(
            initial_value=100_000_000, num_year=5, avg_yield=0.05
        )
        assert result["investment"].iloc[-1] > result["investment"].iloc[0]

    def test_zero_yield_constant_investment(self):
        result = hd.simulate_simple_compounding(
            initial_value=100_000_000, num_year=5, avg_yield=0.0
        )
        # With zero yield returns are 0, investment stays flat
        assert (result["returns"] == 0).all()
        assert (result["investment"] == 100_000_000).all()


# ══════════════════════════════════════════════════════════════════════════════
# calc_best_buy_timing
# ══════════════════════════════════════════════════════════════════════════════

class TestCalcBestBuyTiming:

    def test_returns_none_on_empty_price_df(self):
        empty = pd.DataFrame(columns=["date", "close"])
        result = hd.calc_best_buy_timing(empty, None)
        assert result == (None, None, None)

    def test_returns_none_when_price_df_is_none(self):
        result = hd.calc_best_buy_timing(None, None)
        assert result == (None, None, None)

    def test_seasonality_with_multi_year_data(self, price_df, div_raw_df):
        seasonality, pre_ex, ex_drop = hd.calc_best_buy_timing(price_df, div_raw_df)
        # price_df spans > 2 years → seasonality_df should be produced
        assert seasonality is not None
        assert "month" in seasonality.columns
        assert "mean" in seasonality.columns
        assert len(seasonality) == 12

    def test_pre_ex_trajectory(self, price_df, div_raw_df):
        _seasonality, pre_ex, ex_drop = hd.calc_best_buy_timing(price_df, div_raw_df)
        assert pre_ex is not None
        assert "days_to_ex" in pre_ex.columns
        assert "mean" in pre_ex.columns

    def test_no_dividend_data(self, price_df):
        """No dividend history → pre_ex should be None, seasonality may still exist."""
        seasonality, pre_ex, ex_drop = hd.calc_best_buy_timing(price_df, None)
        assert pre_ex is None
        assert ex_drop is None

    def test_short_price_history_seasonality_none(self, price_df_short):
        """< 2 years of price data → seasonality_df should be None."""
        div_raw = pd.DataFrame({
            "date": ["2022-05-10"],
            "adjDividend": [100],
        })
        seasonality, _pre_ex, _drop = hd.calc_best_buy_timing(price_df_short, div_raw)
        assert seasonality is None


# ══════════════════════════════════════════════════════════════════════════════
# calc_pre_ex_best_days
# ══════════════════════════════════════════════════════════════════════════════

class TestCalcPreExBestDays:

    def test_empty_on_none_inputs(self):
        assert hd.calc_pre_ex_best_days(None, None) == []
        assert hd.calc_pre_ex_best_days(pd.DataFrame(), pd.DataFrame()) == []

    def test_returns_list_of_ints(self, price_df, div_raw_df):
        result = hd.calc_pre_ex_best_days(price_df, div_raw_df)
        assert isinstance(result, list)
        for v in result:
            assert isinstance(v, (int, np.integer))
            assert 0 < v <= 180

    def test_ignores_malformed_price_rows(self):
        prices = pd.DataFrame({
            "date": ["bad-date", "2024-01-08", "2024-01-09"],
            "close": ["N/A", 100, 90],
        })
        dividends = pd.DataFrame({"date": ["2024-01-10"]})

        result = hd.calc_pre_ex_best_days(prices, dividends, detail=True)

        assert result[0]["days_before"] == 1
        assert result[0]["low_price"] == 90


# ══════════════════════════════════════════════════════════════════════════════
# calc_post_ex_recovery_days
# ══════════════════════════════════════════════════════════════════════════════

class TestCalcPostExRecoveryDays:

    def test_censors_at_observed_followup(self):
        dates = pd.date_range("2024-01-01", "2024-01-19", freq="B")
        prices = pd.DataFrame({
            "date": dates,
            "close": np.where(dates < pd.Timestamp("2024-01-10"), 100, 90),
        })
        dividends = pd.DataFrame({"date": ["2024-01-10"]})

        result = hd.calc_post_ex_recovery_days(prices, dividends, detail=True)

        assert result[0]["recovered"] is False
        assert result[0]["days_after"] == 9
        assert result[0]["recover_date"] is None

    def test_records_actual_recovery_day(self):
        dates = pd.date_range("2024-01-01", "2024-01-19", freq="B")
        close = np.where(dates < pd.Timestamp("2024-01-10"), 100, 90)
        close[dates == pd.Timestamp("2024-01-15")] = 101
        prices = pd.DataFrame({"date": dates, "close": close})
        dividends = pd.DataFrame({"date": ["2024-01-10"]})

        result = hd.calc_post_ex_recovery_days(prices, dividends, detail=True)

        assert result[0]["recovered"] is True
        assert result[0]["days_after"] == 5
        assert result[0]["recover_date"] == "2024-01-15"


# ══════════════════════════════════════════════════════════════════════════════
# calc_aggregate_seasonality
# ══════════════════════════════════════════════════════════════════════════════

class TestCalcAggregateSeasonality:

    def test_empty_input(self):
        result = hd.calc_aggregate_seasonality([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_multi_stock_output(self, price_df):
        records = [
            {"symbol": "AAAA", "price_df": price_df},
            {"symbol": "BBBB", "price_df": price_df},
        ]
        result = hd.calc_aggregate_seasonality(records)
        assert len(result) == 12
        assert "month_name" in result.columns
        assert "mean" in result.columns

    def test_output_columns(self, price_df):
        records = [{"symbol": "AAAA", "price_df": price_df}]
        result = hd.calc_aggregate_seasonality(records)
        expected_cols = {"month", "month_name", "mean", "median", "std", "q25", "q75"}
        assert expected_cols.issubset(set(result.columns))

    def test_excludes_partial_years(self):
        prices = pd.DataFrame({
            "date": pd.date_range("2024-01-01", "2024-06-30", freq="B"),
            "close": 100,
        })

        result = hd.calc_aggregate_seasonality([{"symbol": "AAAA", "price_df": prices}])

        assert result.empty

    def test_removes_log_linear_price_trend(self):
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        elapsed = (dates - dates.min()).days.to_numpy(dtype=float)
        prices = pd.DataFrame({"date": dates, "close": np.exp(0.002 * elapsed) * 100})

        result = hd.calc_aggregate_seasonality([{"symbol": "AAAA", "price_df": prices}])

        np.testing.assert_allclose(result["median"], 100, atol=1e-8)

    def test_gives_each_stock_equal_weight(self):
        def yearly_prices(year, monthly_factor):
            dates = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
            elapsed = (dates - dates.min()).days.to_numpy(dtype=float)
            return pd.DataFrame({
                "date": dates,
                "close": np.exp(0.001 * elapsed) * monthly_factor,
            })

        factors_a = np.array([80] + [100] * 11)
        factors_b = np.array([120] + [100] * 11)
        a_one_year = yearly_prices(2022, factors_a)
        a_two_years = pd.concat([a_one_year, yearly_prices(2023, factors_a)], ignore_index=True)
        b_one_year = yearly_prices(2022, factors_b)

        one_year_result = hd.calc_aggregate_seasonality([
            {"symbol": "AAAA", "price_df": a_one_year},
            {"symbol": "BBBB", "price_df": b_one_year},
        ])
        two_year_result = hd.calc_aggregate_seasonality([
            {"symbol": "AAAA", "price_df": a_two_years},
            {"symbol": "BBBB", "price_df": b_one_year},
        ])

        np.testing.assert_allclose(one_year_result["mean"], two_year_result["mean"], atol=1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# calc_price_changes
# ══════════════════════════════════════════════════════════════════════════════

class TestCalcPriceChanges:

    def _make_stock_dict(self):
        dates = pd.date_range("2022-01-01", periods=60, freq="B")
        prices = np.linspace(1000, 1200, 60)
        df = pd.DataFrame({"date": dates, "price": prices})
        return {"AAAA": df}

    def test_output_has_expected_keys(self):
        result = hd.calc_price_changes(self._make_stock_dict())
        # Should return a dict keyed by date string
        assert isinstance(result, dict)
        first_key = next(iter(result))
        assert isinstance(first_key, str)

    def test_first_row_no_daily_change(self):
        result = hd.calc_price_changes(self._make_stock_dict())
        first_date = sorted(result.keys())[0]
        first_df = result[first_date]
        aaaa_row = first_df[first_df["stock"] == "AAAA"]
        assert aaaa_row["price_change"].iloc[0] is None or pd.isna(aaaa_row["price_change"].iloc[0])

    def test_daily_change_computed_for_later_rows(self):
        result = hd.calc_price_changes(self._make_stock_dict())
        later_date = sorted(result.keys())[5]
        df = result[later_date]
        aaaa = df[df["stock"] == "AAAA"]
        assert not pd.isna(aaaa["price_change"].iloc[0])


# ══════════════════════════════════════════════════════════════════════════════
# calculate_returns
# ══════════════════════════════════════════════════════════════════════════════

class TestCalculateReturns:
    """calculate_returns uses merge_asof which requires the merge key (date)
    to be globally sorted after groupby-equivalent operations.  We use a
    single-symbol DataFrame so dates are sorted after sort_values."""

    @pytest.fixture
    def single_stock_df(self):
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="B")
        return pd.DataFrame({
            "symbol": ["AAAA"] * len(dates),
            "date":   dates,
            "close":  np.linspace(1000, 2000, len(dates)),
        })

    def test_return_columns_created(self, single_stock_df):
        result = hd.calculate_returns(single_stock_df)
        for col in ["return_7d", "return_1m", "return_1y", "return_10y"]:
            assert col in result.columns

    def test_returns_are_fractional(self, single_stock_df):
        result = hd.calculate_returns(single_stock_df)
        # Returns should be relatively small for this dataset
        valid = result["return_7d"].dropna()
        assert (valid.abs() < 10).all()  # sanity: not thousands percent

    def test_date_converted_to_datetime(self):
        df = pd.DataFrame({
            "symbol": ["A", "A"],
            "date":   ["2022-01-03", "2022-01-04"],
            "close":  [100, 101],
        })
        result = hd.calculate_returns(df)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])


# ══════════════════════════════════════════════════════════════════════════════
# calc_daily_return_for_date  (pure function in market_watch, also importable)
# ══════════════════════════════════════════════════════════════════════════════

class TestCalcDailyReturnForDate:
    """Tests the standalone calc_daily_return_for_date logic from market_watch.
    
    We replicate the logic here (it has no Streamlit dependencies) to avoid
    importing the entire market_watch module which calls st at import time.
    """

    @staticmethod
    def _calc_daily_return_for_date(prices_df, target_date):
        """Mirror of market_watch.calc_daily_return_for_date."""
        if prices_df.empty:
            return pd.DataFrame()
        df = prices_df.sort_values(["symbol", "date"])
        dates_before = df[df["date"] < target_date]["date"].unique()
        if len(dates_before) == 0:
            return pd.DataFrame()
        prev_date = pd.Timestamp(max(dates_before))
        today_df = (
            df[df["date"] == target_date]
            .set_index("symbol")[["close"]]
        )
        prev_df = (
            df[df["date"] == prev_date]
            .set_index("symbol")[["close"]]
            .rename(columns={"close": "prev_close"})
        )
        merged = today_df.join(prev_df, how="inner")
        merged["return_1d_pct"] = (merged["close"] / merged["prev_close"] - 1) * 100
        return merged

    def test_normal_case(self):
        df = pd.DataFrame({
            "symbol": ["A", "A", "B", "B"],
            "date":   pd.to_datetime(["2023-01-02", "2023-01-03",
                                       "2023-01-02", "2023-01-03"]),
            "close":  [100, 105, 200, 196],
        })
        target = pd.Timestamp("2023-01-03")
        result = self._calc_daily_return_for_date(df, target)
        assert "return_1d_pct" in result.columns
        assert abs(result.loc["A", "return_1d_pct"] - 5.0) < 1e-6
        assert result.loc["B", "return_1d_pct"] < 0  # B fell

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame(columns=["symbol", "date", "close"])
        result = self._calc_daily_return_for_date(empty, pd.Timestamp("2023-01-03"))
        assert result.empty

    def test_no_prev_date_returns_empty(self):
        df = pd.DataFrame({
            "symbol": ["A"],
            "date":   pd.to_datetime(["2023-01-02"]),
            "close":  [100.0],
        })
        result = self._calc_daily_return_for_date(df, pd.Timestamp("2023-01-02"))
        assert result.empty


# ══════════════════════════════════════════════════════════════════════════════
# prep_treemap
# ══════════════════════════════════════════════════════════════════════════════

class TestPrepTreemap:

    @pytest.fixture
    def treemap_df(self):
        return pd.DataFrame({
            "stock":    ["AAAA.JK", "BBBB.JK", "CCCC.JK"],
            "sector":   ["Financials", "Financials", "Telecom"],
            "industry": ["Banks", "Insurance", "Telecom"],
            "mktCap":   [5e13, 2e13, 3e13],
            "yield":    [3.0, 2.5, 4.0],
        }).set_index("stock")

    def test_grouped_by_sector(self, treemap_df):
        result = hd.prep_treemap(treemap_df, size_var="mktCap", group_secs=True)
        assert isinstance(result, list)
        sector_names = [node["name"] for node in result]
        assert "Financials" in sector_names
        assert "Telecom" in sector_names

    def test_ungrouped(self, treemap_df):
        result = hd.prep_treemap(treemap_df, size_var="mktCap", group_secs=False)
        assert isinstance(result, list)
        assert len(result) == 1  # single root 'ALL' node
        assert result[0]["name"] == "ALL"

    def test_with_color_var(self, treemap_df):
        result = hd.prep_treemap(
            treemap_df, size_var="mktCap",
            color_var="yield", group_secs=True
        )
        assert isinstance(result, list)
        # Each sector node value contains [size, weighted color, color_grad].
        for sector_node in result:
            assert len(sector_node["value"]) == 3

    def test_parent_color_is_size_weighted(self):
        df = pd.DataFrame({
            "stock": ["LARGE", "SMALL"],
            "sector": ["Technology", "Technology"],
            "industry": ["Software", "Software"],
            "mktCap": [90.0, 10.0],
            "return": [10.0, -10.0],
        }).set_index("stock")
        result = hd.prep_treemap(
            df,
            size_var="mktCap",
            color_var="return",
            color_threshold=[-5, 0, 5],
            group_secs=True,
        )
        sector = result[0]
        industry = sector["children"][0]
        assert sector["value"][0] == 100.0
        assert sector["value"][1] == pytest.approx(8.0)
        assert industry["value"][1] == pytest.approx(8.0)
        assert 0 <= sector["value"][2] <= 100

    def test_same_size_and_color_preserves_parent_area_sum(self):
        df = pd.DataFrame({
            "stock": ["HIGH", "LOW"],
            "sector": ["Technology", "Technology"],
            "industry": ["Software", "Software"],
            "yield": [8.0, 2.0],
        }).set_index("stock")
        result = hd.prep_treemap(
            df,
            size_var="yield",
            color_var="yield",
            color_threshold=[0, 3, 6, 9],
            group_secs=True,
        )
        sector = result[0]
        industry = sector["children"][0]
        leaves = {node["name"]: node["value"][0] for node in industry["children"]}

        assert sector["value"][0] == pytest.approx(10.0)
        assert industry["value"][0] == pytest.approx(10.0)
        assert leaves["HIGH"] == pytest.approx(8.0)
        assert leaves["LOW"] == pytest.approx(2.0)

    def test_prep_treemap_does_not_mutate_input(self, treemap_df):
        hd.prep_treemap(
            treemap_df, size_var="mktCap", color_var="yield", group_secs=True
        )
        assert "color_grad" not in treemap_df.columns

    def test_without_color_var(self, treemap_df):
        result = hd.prep_treemap(
            treemap_df, size_var="mktCap",
            color_var=None, group_secs=True
        )
        for sector_node in result:
            # value should only have [size]
            assert len(sector_node["value"]) == 1


# ══════════════════════════════════════════════════════════════════════════════
# get_usd_idr_rate  (cache + fallback logic)
# ══════════════════════════════════════════════════════════════════════════════

class TestGetUsdIdrRate:

    def setup_method(self):
        """Reset the in-memory cache before each test."""
        hd._usd_idr_cache["rate"] = None
        hd._usd_idr_cache["timestamp"] = None

    @patch("harvest.data.requests.get")
    def test_live_fetch_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"price": 16_000.0}]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {"FMP_API_KEY": "dummy"}):
            rate = hd.get_usd_idr_rate()
        assert rate == 16_000.0

    @patch("harvest.data.requests.get", side_effect=Exception("timeout"))
    def test_fallback_on_error(self, _mock_get):
        with patch.dict("os.environ", {"FMP_API_KEY": "dummy"}):
            rate = hd.get_usd_idr_rate(fallback=15_000)
        assert rate == 15_000

    def test_cache_returned_when_fresh(self):
        import time
        hd._usd_idr_cache["rate"] = 17_000.0
        hd._usd_idr_cache["timestamp"] = time.time()
        rate = hd.get_usd_idr_rate()
        assert rate == 17_000.0

    def test_no_api_key_uses_fallback(self):
        import os
        os.environ.pop("FMP_API_KEY", None)
        rate = hd.get_usd_idr_rate(fallback=14_000)
        assert rate == 14_000


# ══════════════════════════════════════════════════════════════════════════════
# calc_ratio_history
# ══════════════════════════════════════════════════════════════════════════════

class TestCalcRatioHistory:

    @patch("harvest.data.get_usd_idr_rate", return_value=15_000)
    def test_pe_ratio_mode(self, _mock_rate, price_df, fin_df):
        n_shares = 1_000_000_000
        result = hd.calc_ratio_history(
            price_df, fin_df, n_shares=n_shares,
            ratio="pe", reported_currency="IDR", target_currency="IDR"
        )
        assert "pe" in result.columns

    @patch("harvest.data.get_usd_idr_rate", return_value=15_000)
    def test_ps_ratio_mode(self, _mock_rate, price_df, fin_df):
        n_shares = 1_000_000_000
        result = hd.calc_ratio_history(
            price_df, fin_df, n_shares=n_shares,
            ratio="ps", reported_currency="IDR", target_currency="IDR"
        )
        assert "pe" in result.columns  # column is still named 'pe' internally
