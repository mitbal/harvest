import importlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


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


def test_domains_increase_and_match_palette_length():
    for sl in ['JKSE', 'S&P500']:
        domain = plot.dividend_calendar_domain(sl)
        assert domain == sorted(domain) and len(set(domain)) == len(domain)
        assert len(domain) == len(plot.DIVIDEND_CALENDAR_COLORS)
    # US payouts are quarterly, so per-date yields sit far lower than JKSE
    assert plot.dividend_calendar_domain('S&P500')[-1] < plot.dividend_calendar_domain('JKSE')[-1] / 4


def test_month_view_scale_ignores_other_months():
    # The scale is fixed, so a high yield in March must not change January's colors
    df = _dividends()
    with_march = _color_scale(plot.plot_dividend_month(df, 1))
    without_march = _color_scale(plot.plot_dividend_month(df[df['date'].dt.month == 1], 1))
    assert with_march == without_march


def test_full_year_image_has_title_legend_and_4x3_grid():
    chart = plot.plot_dividend_calendar(_dividends(), title='Dividend Calendar 2026 · JKSE', legend=True)
    spec = chart.to_dict()
    # Title and color key share one row below the grid
    assert spec['resolve']['scale']['color'] == 'independent'
    grid, footer = spec['vconcat']
    title, legend = footer['hconcat']
    assert spec['datasets'][title['data']['name']][0]['text'] == 'Dividend Calendar 2026 · JKSE'
    assert legend['layer'][0]['mark']['type'] == 'rect'
    assert len(grid['hconcat']) == 4
    assert all(len(column['vconcat']) == 3 for column in grid['hconcat'])
    # Row-major months: first row is January-April
    assert [column['vconcat'][0]['layer'][0]['title'] for column in grid['hconcat']] == ['January', 'February', 'March', 'April']


def test_full_year_month_cells_are_square_and_bounded_by_height():
    # lesley 0.4.0 (the pinned version): cell = min(width / 7, height / 6)
    spec = plot.plot_dividend_calendar(_dividends(), width=238, height=170).to_dict()
    month = spec['hconcat'][0]['vconcat'][0]
    assert month['width'] == month['height'] == {'step': pytest.approx(170 / 6)}


def test_date_numbers_use_lesley_grid_and_font():
    chart = plot.month_plot(*_sample(), month=3, width=238, height=170)
    texts = [layer for layer, _ in _layers(chart, 'text')]
    cell = 170 / 6
    assert all(t['mark']['fontSize'] == pytest.approx(min(238 / 20, cell * 0.6)) for t in texts)
    weeks = texts[0]['encoding']['y']['scale']['domain']
    assert len(weeks) == 6


def test_full_year_default_has_no_title_or_legend():
    # porto_overview uses the plain grid
    spec = plot.plot_dividend_calendar(_dividends()).to_dict()
    assert 'title' not in spec and 'hconcat' in spec


def test_legend_labels_match_domain():
    for sl, expected in [('JKSE', ['1%', '2%', '3%', '4%', '6%', '8%', '12%+']),
                         ('S&P500', ['0.25%', '0.5%', '0.75%', '1%', '1.25%', '1.5%', '2%+'])]:
        spec = plot.plot_dividend_calendar_legend(sl).to_dict()
        rows = spec['datasets'][spec['data']['name']] if 'data' in spec else next(iter(spec['datasets'].values()))
        assert [r['label'] for r in rows] == expected
        assert [r['color'] for r in rows] == [c.lower() for c in plot.DIVIDEND_CALENDAR_COLORS]
        # Readable text on every swatch: black on the light greens, white on the dark ones
        assert [r['text_color'] for r in rows] == ['#000000'] * 4 + ['#FFFFFF'] * 3


def test_lightest_shade_stands_out_from_page_background():
    def contrast(a, b):
        la, lb = sorted([plot._relative_luminance(a), plot._relative_luminance(b)], reverse=True)
        return (la + 0.05) / (lb + 0.05)
    # The old 'Greens' start (#E5F5E0) had a 1.0 contrast with the #EFF1EF background
    assert contrast(plot.DIVIDEND_CALENDAR_COLORS[0], '#EFF1EF') >= 1.25


def test_heatmap_uses_custom_colors_and_domain():
    scale = _color_scale(plot.plot_dividend_month(_dividends(), 1))
    assert scale['domain'] == plot.dividend_calendar_domain('JKSE')
    assert scale['range'] == [c.lower() for c in plot.DIVIDEND_CALENDAR_COLORS]


def test_date_numbers_white_on_dark_cells():
    # 3% sits on a light shade, 6% and 12% on the dark ones; the 3rd (Saturday) stays red
    dates = pd.Series(pd.to_datetime(['2026-01-12', '2026-01-20', '2026-01-27']))
    values = pd.Series([3.0, 6.0, 12.0])
    chart = plot.month_plot(dates, values, month=1, domain=plot.dividend_calendar_domain('JKSE'),
                            color_range=plot.DIVIDEND_CALENDAR_COLORS)
    rows = {r['day']: r for _, layer_rows in _layers(chart, 'text') for r in layer_rows}
    assert not rows[12]['on_dark']
    assert rows[20]['on_dark'] and rows[27]['on_dark']
    assert rows[3]['is_closed'] and not rows[3]['on_dark']
    assert not rows[5]['on_dark'] and not rows[5]['is_closed']
    color = next(layer for layer, _ in _layers(chart, 'text'))['encoding']['color']
    assert color['condition'][0] == {'test': 'datum.is_closed', 'value': plot.CALENDAR_CLOSED_COLOR}
    assert color['condition'][1] == {'test': 'datum.on_dark', 'value': '#FFFFFF'}


def test_full_year_spec_compiles():
    # The 12 months share one color scale; the date layers must not add a conflicting one
    import vl_convert as vlc
    chart = plot.plot_dividend_calendar(_dividends(), width=250, height=200, title='t', legend=True)
    assert vlc.vegalite_to_svg(chart.to_json()).startswith('<svg')


def test_yields_outside_the_scale_are_clamped_for_color_only():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2026-03-10', '2026-03-11', '2026-03-12']),
        'symbol': ['LOW.JK', 'MID.JK', 'HIGH.JK'],
        'yield': [0.2, 3.0, 59.6],
    })
    spec = plot.plot_dividend_month(df, 3).to_dict()
    heatmap = spec['layer'][0]
    rows = spec['datasets'][heatmap['data']['name']]
    by_label = {r['labels'].split(': ')[1].split(' ')[0]: r for r in rows}
    domain = plot.dividend_calendar_domain('JKSE')
    assert by_label['LOW.JK']['values'] == domain[0]
    assert by_label['MID.JK']['values'] == 3.0
    assert by_label['HIGH.JK']['values'] == domain[-1]
    assert '(59.60%)' in by_label['HIGH.JK']['labels']


def test_scale_color_interpolates_and_clamps():
    domain, colors = [1, 3], ['#000000', '#ffffff']
    assert plot._scale_color(0.5, domain, colors) == '#000000'
    assert plot._scale_color(2, domain, colors) == '#808080'
    assert plot._scale_color(9, domain, colors) == '#ffffff'


def test_full_year_has_no_gridlines():
    config = plot.plot_dividend_calendar(_dividends()).to_dict()['config']
    assert config['axis']['grid'] is False and config['axisY']['grid'] is False


def test_readable_text_color():
    assert plot.readable_text_color('#B5E0A8') == '#000000'
    assert plot.readable_text_color('#14532D') == '#FFFFFF'
