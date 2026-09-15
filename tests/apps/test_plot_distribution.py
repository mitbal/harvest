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
