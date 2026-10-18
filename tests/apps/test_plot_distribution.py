import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


_PLOT_PATH = Path(__file__).parents[2] / 'harvest' / 'plot.py'
_SPEC = importlib.util.spec_from_file_location('harvest_plot_distribution_test', _PLOT_PATH)
plot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plot)


def test_comparison_colors_follow_ticker_mapping_not_value_order():
    df = pd.DataFrame({'metric': [1, 2, 3, 4, 5]})
    chart = plot.plot_card_distribution(
        df,
        'metric',
        comparison_vals={'HIGH': 4, 'LOW': 2},
        comparison_colors={'HIGH': '#111111', 'LOW': '#222222'},
    )
    spec = chart.to_dict()
    seen = {}
    for layer in spec['layer']:
        data_name = layer.get('data', {}).get('name')
        rows = spec.get('datasets', {}).get(data_name, [])
        if rows and 'label' in rows[0]:
            seen.setdefault(rows[0]['label'], set()).add(layer['mark']['color'])
    assert seen['HIGH'] == {'#111111'}
    assert seen['LOW'] == {'#222222'}


def test_empty_and_non_finite_distributions_return_empty_state_chart():
    for values in ([], [np.nan, np.inf, -np.inf]):
        chart = plot.plot_card_distribution(pd.DataFrame({'metric': values}), 'metric')
        spec = chart.to_dict()
        assert spec['mark']['type'] == 'text'
        assert spec['datasets']


def test_comparison_outlier_keeps_actual_value_in_tooltip_data():
    df = pd.DataFrame({'metric': [1, 2, 3, 4, 5]})
    chart = plot.plot_card_distribution(
        df,
        'metric',
        comparison_vals={'OUTLIER': 100},
        comparison_colors={'OUTLIER': '#123456'},
        x_range=(1, 5),
    )
    spec = str(chart.to_dict())
    assert "'actual_value': 100.0" in spec
    assert 'Above displayed p5-p95 range' in spec


def test_dividend_history_marks_current_year_as_partial_not_a_cut():
    from datetime import datetime

    this_year = datetime.today().year
    div_df = pd.DataFrame({
        'date': [f'{this_year - 2}-04-01', f'{this_year - 1}-04-01', f'{this_year}-04-01'],
        'adjDividend': [100.0, 120.0, 50.0],  # current year has only paid an interim so far
    })
    spec = plot.plot_dividend_history(div_df).to_dict()
    rows = next(iter(spec['datasets'].values()))
    status = {row['year']: row['status'] for row in rows}

    assert status[this_year] == 'Year to date'
    assert status[this_year - 1] == 'Raised or held'


def test_dividend_history_projection_of_a_cutter_stays_positive():
    div_df = pd.DataFrame({
        'date': ['2020-04-01', '2021-04-01', '2022-04-01'],
        'adjDividend': [160.0, 130.0, 100.0],
    })
    chart = plot.plot_dividend_history(div_df, extrapolote=True, n_future_years=5, last_val=100, inc_val=-30)
    spec = chart.to_dict()
    projected = [
        row['adjDividend']
        for rows in spec['datasets'].values()
        for row in rows
        if 'status' not in row
    ]

    assert len(projected) == 5
    assert all(v > 0 for v in projected)
    assert projected == sorted(projected, reverse=True)


def test_diverging_color_centres_the_ramp_on_mid():
    from harvest import chart_style as cs

    assert cs.diverging_color(50, 0, 50, 100) == cs.RED_YELLOW_GREEN[5]
    assert cs.diverging_color(0, 0, 50, 100) == cs.RED_YELLOW_GREEN[0]
    assert cs.diverging_color(100, 0, 50, 100) == cs.RED_YELLOW_GREEN[-1]
    # Mid below every value (stock looks cheap): the lowest value sits at the yellow centre.
    assert cs.diverging_color(10, 10, 10, 20) == cs.RED_YELLOW_GREEN[5]


def test_readable_text_color_uses_light_text_on_both_dark_ends():
    from harvest import chart_style as cs

    assert cs.readable_text_color(cs.RED_YELLOW_GREEN[0]) == '#FFFFFF'
    assert cs.readable_text_color(cs.RED_YELLOW_GREEN[-1]) == '#FFFFFF'
    assert cs.readable_text_color(cs.RED_YELLOW_GREEN[5]) == cs.CHART_TEXT
