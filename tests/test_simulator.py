import pandas as pd
import pytest

import harvest.data as hd
import harvest.simulator as hs


def _price_history():
    return pd.DataFrame({
        'date': [
            '2020-01-02',
            '2020-06-01',
            '2020-12-31',
            '2021-06-01',
            '2021-12-31',
        ],
        'close': [100, 100, 100, 100, 100],
    })


def _dividend_history():
    return pd.DataFrame({
        'date': ['2020-06-01', '2021-06-01'],
        'adjDividend': [30, 30],
    })


class TestSimpleCompounding:
    def test_one_year_reports_ending_balance(self):
        result = hd.simulate_simple_compounding(100, 1, 0.10)

        assert result.loc[0, 'returns'] == 10
        assert result.loc[0, 'investment'] == 110
        assert result.loc[0, 'portfolio_value'] == (
            result.loc[0, 'holdings_value'] + result.loc[0, 'cash']
        )


class TestAllocationValidation:
    def test_blank_rows_are_skipped_without_shifting_values(self):
        result = hs.build_allocations([
            ('', 10),
            (' bbca.jk ', 20),
        ])

        assert result == {'BBCA.JK': 20.0}

    def test_duplicate_tickers_are_rejected(self):
        with pytest.raises(hs.SimulatorValidationError, match='unique'):
            hs.build_allocations([('BBCA.JK', 10), ('bbca.jk', 20)])

    def test_zero_allocation_is_rejected(self):
        with pytest.raises(hs.SimulatorValidationError, match='greater than zero'):
            hs.build_allocations([('BBCA.JK', 0)])


class TestHistoricalPortfolio:
    def test_whole_lots_cash_and_drip_reconcile(self):
        aggregate, detail, _ = hs.simulate_historical_portfolio(
            {'AAAA.JK': 25_000},
            {'AAAA.JK': _price_history()},
            {'AAAA.JK': _dividend_history()},
            2020,
            2021,
        )

        assert (
            aggregate['portfolio_value']
            == aggregate['holdings_value'] + aggregate['cash']
        ).all()
        drip_2021 = aggregate[
            (aggregate['strategy'] == hs.STRATEGY_DRIP)
            & (aggregate['year'] == 'Year 2021')
        ].iloc[0]
        no_drip_2021 = aggregate[
            (aggregate['strategy'] == hs.STRATEGY_NO_DRIP)
            & (aggregate['year'] == 'Year 2021')
        ].iloc[0]
        drip_lots = detail[
            (detail['Strategy'] == hs.STRATEGY_DRIP)
            & (detail['year'] == 'Year 2021')
            & (detail['stock'] == 'AAAA.JK')
        ]['lot'].iloc[0]
        no_drip_lots = detail[
            (detail['Strategy'] == hs.STRATEGY_NO_DRIP)
            & (detail['year'] == 'Year 2021')
            & (detail['stock'] == 'AAAA.JK')
        ]['lot'].iloc[0]

        assert drip_lots == 4
        assert no_drip_lots == 2
        assert drip_2021['portfolio_value'] == pytest.approx(40_000)
        assert no_drip_2021['portfolio_value'] == pytest.approx(37_000)

    def test_price_order_does_not_change_results(self):
        ascending, _, _ = hs.simulate_historical_portfolio(
            {'AAAA.JK': 25_000},
            {'AAAA.JK': _price_history()},
            {'AAAA.JK': _dividend_history()},
            2020,
            2021,
        )
        descending, _, _ = hs.simulate_historical_portfolio(
            {'AAAA.JK': 25_000},
            {'AAAA.JK': _price_history().iloc[::-1].reset_index(drop=True)},
            {'AAAA.JK': _dividend_history().iloc[::-1].reset_index(drop=True)},
            2020,
            2021,
        )

        pd.testing.assert_frame_equal(ascending, descending)

    def test_missing_dividends_are_zero_income_not_an_error(self):
        aggregate, _, transactions = hs.simulate_historical_portfolio(
            {'AAAA.JK': 25_000},
            {'AAAA.JK': _price_history()},
            {'AAAA.JK': None},
            2020,
            2021,
        )

        assert (aggregate['dividend_income'] == 0).all()
        assert len(transactions) == 1

    def test_small_dividend_remains_as_cash(self):
        dividends = pd.DataFrame({
            'date': ['2020-06-01'],
            'adjDividend': [1],
        })
        aggregate, detail, _ = hs.simulate_historical_portfolio(
            {'AAAA.JK': 20_000},
            {'AAAA.JK': _price_history()},
            {'AAAA.JK': dividends},
            2020,
            2020,
        )

        drip = aggregate[aggregate['strategy'] == hs.STRATEGY_DRIP].iloc[0]
        lots = detail[
            (detail['Strategy'] == hs.STRATEGY_DRIP)
            & (detail['stock'] == 'AAAA.JK')
        ]['lot'].iloc[0]
        assert lots == 2
        assert drip['cash'] == pytest.approx(200)

    def test_same_date_transactions_are_preserved(self):
        aggregate, _, transactions = hs.simulate_historical_portfolio(
            {'AAAA.JK': 20_000, 'BBBB.JK': 20_000},
            {'AAAA.JK': _price_history(), 'BBBB.JK': _price_history()},
            {'AAAA.JK': _dividend_history(), 'BBBB.JK': _dividend_history()},
            2020,
            2020,
        )

        assert not aggregate.empty
        dividend_rows = transactions[
            transactions['Activity'].str.startswith('Receive dividend')
        ]
        assert len(dividend_rows) == 4

    def test_missing_price_history_has_actionable_error(self):
        with pytest.raises(hs.SimulatorValidationError, match='No price data'):
            hs.simulate_historical_portfolio(
                {'AAAA.JK': 25_000},
                {'AAAA.JK': pd.DataFrame()},
                {'AAAA.JK': None},
                2020,
                2021,
            )
