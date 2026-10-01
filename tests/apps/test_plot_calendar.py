import importlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load(name, spec):
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PLOT_PATH = Path(__file__).parents[2] / 'harvest' / 'plot.py'
plot = _load('harvest_plot_calendar_test', importlib.util.spec_from_file_location('harvest_plot_calendar_test', _PLOT_PATH))


def _real_lesley():
    # tests/apps/conftest.py stubs lesley; these tests need the real chart data.
    # Swap only the lesley entries so other modules imported meanwhile stay loaded.
    stubbed = {k: sys.modules.pop(k) for k in list(sys.modules) if k == 'lesley' or k.startswith('lesley.')}
    try:
        return importlib.import_module('lesley')
    finally:
        for k in [k for k in sys.modules if k == 'lesley' or k.startswith('lesley.')]:
            del sys.modules[k]
        sys.modules.update(stubbed)


plot.lesley = _real_lesley()


def _layers(chart, mark):
    spec = chart.to_dict()
    for layer in spec['layer']:
        mark_type = layer['mark'] if isinstance(layer['mark'], str) else layer['mark'].get('type')
        if mark_type == mark:
            yield layer, spec['datasets'][layer['data']['name']]


def _closed_days(chart):
    """Day numbers the date-label layers color red."""
    days = [r['day'] for _, rows in _layers(chart, 'text') for r in rows if r['is_closed']]
    assert days, 'no date label layer'
    return sorted(days)


def _holiday_tooltips(chart):
    """{day: holiday name} for every layer that shows a holiday tooltip."""
    found = {}
    for layer, rows in _layers(chart, 'text'):
        if layer['encoding'].get('tooltip', {}).get('field') == 'holiday':
            found.update({r['day']: r['holiday'] for r in rows})
    return found


def _sample():
    dates = pd.Series(pd.to_datetime(['2026-03-10', '2026-03-25']))
    return dates, pd.Series([5.0, 3.0])


def test_weekends_marked_closed():
    dates, values = _sample()
    chart = plot.month_plot(dates, values, month=3)
    # March 2026 weekends
    assert _closed_days(chart) == [1, 7, 8, 14, 15, 21, 22, 28, 29]


def test_holidays_marked_closed_like_weekends():
    dates, values = _sample()
    holidays = [pd.Timestamp('2026-03-19').date(), pd.Timestamp('2026-03-20').date()]
    chart = plot.month_plot(dates, values, month=3, closed_dates=holidays)
    assert _closed_days(chart) == [1, 7, 8, 14, 15, 19, 20, 21, 22, 28, 29]


def test_holiday_names_show_as_tooltip():
    dates, values = _sample()
    holidays = {pd.Timestamp('2026-03-19').date(): 'Day of Silence', pd.Timestamp('2026-04-03').date(): 'Good Friday'}
    chart = plot.month_plot(dates, values, month=3, closed_dates=holidays)
    assert _holiday_tooltips(chart) == {19: 'Day of Silence'}
    hover = list(_layers(chart, 'rect'))[-1]
    assert hover[0]['encoding']['tooltip']['field'] == 'holiday'
    assert [r['day'] for r in hover[1]] == [19]


def test_regular_days_have_no_tooltip():
    dates, values = _sample()
    chart = plot.month_plot(dates, values, month=3, closed_dates={pd.Timestamp('2026-03-19').date(): 'Day of Silence'})
    for layer, rows in _layers(chart, 'text'):
        if any(r['day'] == 10 for r in rows):
            assert layer['encoding']['tooltip'] == {'value': None}


def test_unnamed_holidays_have_no_tooltip():
    dates, values = _sample()
    chart = plot.month_plot(dates, values, month=3, closed_dates=[pd.Timestamp('2026-03-19').date()])
    assert _holiday_tooltips(chart) == {}


def test_holidays_in_other_months_ignored():
    dates, values = _sample()
    chart = plot.month_plot(dates, values, month=3, closed_dates=[pd.Timestamp('2026-04-03').date()])
    assert 3 not in _closed_days(chart)


def test_show_date_false_has_no_labels():
    dates, values = _sample()
    spec = plot.month_plot(dates, values, month=3, show_date=False).to_dict()
    assert 'layer' not in spec


def test_full_year_calendar_renders_with_holidays():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2026-03-10', '2026-08-03']),
        'symbol': ['A.JK', 'B.JK'],
        'yield': [5.0, 3.0],
    })
    chart = plot.plot_dividend_calendar(df, closed_dates=[pd.Timestamp('2026-08-17').date()])
    assert chart.to_dict()


def _color_scale(chart):
    spec = chart.to_dict()
    layers = spec.get('layer') or [l for col in spec['hconcat'] for l in col['vconcat'][0]['layer']]
    for layer in layers:
        color = layer.get('encoding', {}).get('color', {})
        if color.get('field') == 'values':
            return color['scale']
    raise AssertionError('no heatmap color scale')


def _dividends():
    return pd.DataFrame({
        'date': pd.to_datetime(['2026-01-12', '2026-01-20', '2026-03-10']),
        'symbol': ['A.JK', 'B.JK', 'C.JK'],
        'yield': [4.0, 9.0, 12.0],
    })


def test_month_view_uses_same_color_scale_as_full_year():
    df = _dividends()
    for sl in ['JKSE', 'S&P500']:
        month = _color_scale(plot.plot_dividend_month(df, 1, sl=sl))
        year = _color_scale(plot.plot_dividend_calendar(df, sl=sl))
        assert month == year


def test_us_domain_is_scaled_down():
    assert plot.dividend_calendar_domain('S&P500') == [x / 10 for x in plot.dividend_calendar_domain('JKSE')]


def test_month_view_scale_ignores_other_months():
    # The scale is fixed, so a high yield in March must not change January's colors
    df = _dividends()
    with_march = _color_scale(plot.plot_dividend_month(df, 1))
    without_march = _color_scale(plot.plot_dividend_month(df[df['date'].dt.month == 1], 1))
    assert with_march == without_march
