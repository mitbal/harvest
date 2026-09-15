"""Regression tests for the dividend stock picker."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st


def _company_profiles(symbols):
    return pd.DataFrame(
        {
            'price': [9_500.0] * len(symbols),
            'changes': [100.0] * len(symbols),
            'beta': [0.8] * len(symbols),
            'currency': ['IDR'] * len(symbols),
            'description': ['Test company'] * len(symbols),
            'companyName': ['Test Company'] * len(symbols),
            'sector': ['Financials'] * len(symbols),
            'ipoDate': ['2000-01-01'] * len(symbols),
        },
        index=pd.Index(symbols, name='symbol'),
    )


_dates = pd.date_range('2022-01-03', periods=800, freq='B')
_price_df = pd.DataFrame({
    'date': _dates,
    'open': np.linspace(8_000, 9_500, len(_dates)),
    'high': np.linspace(8_100, 9_600, len(_dates)),
    'low': np.linspace(7_900, 9_400, len(_dates)),
    'close': np.linspace(8_000, 9_500, len(_dates)),
    'volume': 1_000_000,
})
_fin_df = pd.DataFrame({
    'date': ['2024-12-31', '2023-12-31'],
    'calendarYear': [2024, 2023],
    'revenue': [4e13, 3.5e13],
    'netIncome': [1e13, 8e12],
    'reportedCurrency': ['IDR', 'IDR'],
    'period': ['FY', 'FY'],
    'eps': [500, 450],
})
_div_df = pd.DataFrame({
    'date': ['2024-03-01', '2023-03-01', '2022-03-01'],
    'adjDividend': [250.0, 225.0, 200.0],
})


def _select_first(_label=None, options=None, **_kwargs):
    return options[0] if options else None


with patch.dict('os.environ', {'REDIS_URL': 'redis://stub', 'FMP_API_KEY': 'stub'}):
    with patch.object(st, 'selectbox', side_effect=_select_first):
        with patch('harvest.data.get_company_profile', side_effect=lambda symbols: _company_profiles(symbols)):
            with patch('harvest.data.get_shares_outstanding', return_value=pd.DataFrame({'outstandingShares': [1e9]})):
                with patch('harvest.data.get_financial_data', return_value=_fin_df):
                    with patch('harvest.data.get_daily_stock_price', return_value=_price_df):
                        with patch('harvest.data.get_dividend_history_single_stock', return_value=_div_df):
                            import apps.screener.stock_picker as sp


def _raw_stock_df():
    return pd.DataFrame({
        'price': [9_500.0, 3_800.0],
        'mktCap': [5e14, 2e14],
        'yield': [2.5, 4.0],
        'lastDiv': [250.0, 200.0],
        'avgFlatAnnualDivIncrease': [10.0, 5.0],
        'numDividendYear': [15, 10],
        'positiveYear': [12, 8],
        'numOfYear': [20, 15],
        'maximumCutPct': [-20.0, -15.0],
        'max10CutPct': [-10.0, -5.0],
        'peRatio': [15.0, 12.0],
        'revenueGrowth': [10.0, 5.0],
        'netIncomeGrowth': [12.0, 4.0],
        'medianProfitMargin': [25.0, 20.0],
        'earningTTM': [1e13, 8e12],
        'revenueTTM': [4e13, 4e13],
        'revenueGrowthTTM': [8.0, 3.0],
        'netIncomeGrowthTTM': [10.0, 2.0],
        'return_7d': [0.01, -0.02],
        'return_1m': [0.03, -0.01],
        'return_1y': [0.15, 0.08],
        'return_10y': [2.0, 1.5],
        'total_return_1y': [0.18, 0.10],
        'total_return_10y': [2.5, 2.0],
    }, index=['AAA.JK', 'BBB.JK'])


def test_processed_returns_are_percent_values():
    result = sp.get_processed_df(_raw_stock_df())

    assert result.loc['AAA.JK', 'return_1y'] == 15.0
    assert result.loc['AAA.JK', 'total_return_1y'] == 18.0


def test_margin_spike_reduces_dividend_score_through_normalized_payout():
    base = _raw_stock_df().loc[['AAA.JK']].copy()
    comparison = pd.concat([base.rename(index={'AAA.JK': 'SUPPORTED.JK'})] * 2)
    comparison.index = ['SUPPORTED.JK', 'SPIKE.JK']
    comparison[['price', 'mktCap', 'yield', 'lastDiv']] = [5_000.0, 1e14, 12.0, 600.0]
    comparison[['earningTTM', 'revenueTTM']] = [1.6e13, 4e13]
    comparison[['revenueGrowthTTM', 'netIncomeGrowthTTM']] = [5.0, 5.0]
    comparison.loc['SUPPORTED.JK', 'medianProfitMargin'] = 40.0
    comparison.loc['SPIKE.JK', 'medianProfitMargin'] = 10.0

    result = sp.get_processed_df(comparison)

    assert result.loc['SPIKE.JK', 'DScore'] < result.loc['SUPPORTED.JK', 'DScore'] * 0.4


def test_unconfirmed_income_growth_does_not_increase_dividend_score():
    base = _raw_stock_df().loc[['AAA.JK']].copy()
    comparison = pd.concat([base.rename(index={'AAA.JK': 'CONFIRMED.JK'})] * 2)
    comparison.index = ['CONFIRMED.JK', 'UNCONFIRMED.JK']
    comparison['revenueGrowthTTM'] = -10.0
    comparison.loc['CONFIRMED.JK', 'netIncomeGrowthTTM'] = -10.0
    comparison.loc['UNCONFIRMED.JK', 'netIncomeGrowthTTM'] = 150.0

    result = sp.get_processed_df(comparison)

    assert result.loc['UNCONFIRMED.JK', 'DScore'] == result.loc['CONFIRMED.JK', 'DScore']


def test_sector_rating_is_unavailable_without_enough_valid_peers():
    ranking_df = pd.DataFrame({
        'peRatio': [10.0, -2.0],
        'numDividendYear': [5, 6],
        'yield': [2.0, 3.0],
        'revenueGrowth': [4.0, 5.0],
        'netIncomeGrowth': [4.0, 5.0],
        'medianProfitMargin': [10.0, 11.0],
        'sector': ['Banks', 'Banks'],
    }, index=['AAA.JK', 'BBB.JK'])

    ratings = sp.calculate_stock_ratings('AAA.JK', ranking_df)

    assert np.isnan(ratings['sector'])
    assert np.isfinite(ratings['overall'])
    assert ratings['metrics']['sector_peer_count'] == 1


def test_table_presets_keep_the_default_view_focused():
    assert list(sp._TABLE_PRESETS)[0] == 'Essentials'
    assert len(sp._TABLE_PRESETS['Essentials']) <= 10
    assert {'Rank', 'sector', 'industry', 'yield', 'DScore', 'peRatio'} <= set(sp._TABLE_PRESETS['Essentials'])
