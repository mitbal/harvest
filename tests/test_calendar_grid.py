import datetime as dt
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from harvest import calendar_grid as cg


# The test conftests stub harvest.plot; the grid needs its real colour helpers
_PLOT_PATH = Path(__file__).parents[1] / 'harvest' / 'plot.py'
_spec = importlib.util.spec_from_file_location('harvest_plot_grid_test', _PLOT_PATH)
real_plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(real_plot)


@pytest.fixture(autouse=True)
def _real_plot(monkeypatch):
    monkeypatch.setattr(cg, 'hp', real_plot)


def _df(rows):
    return pd.DataFrame(
        [{'date': pd.Timestamp(d), 'symbol': s, 'yield': y, 'adjDividend': div} for d, s, y, div in rows]
    )


EMPTY = _df([]).reindex(columns=['date', 'symbol', 'yield', 'adjDividend']).astype({'date': 'datetime64[ns]'})


def _days(weeks):
    return [cell for week in weeks for cell in week]


def _cell(weeks, day):
    return next(cell for cell in _days(weeks) if cell['day'] == day)


class TestMonthCells:

    def test_weeks_have_seven_days_monday_first(self):
        # May 2027 starts on a Saturday and ends on a Monday
        weeks = cg.month_cells(EMPTY, 2027, 5, {})
        assert all(len(week) == 7 for week in weeks)
        assert [c['day'] for c in weeks[0]] == [None] * 5 + [1, 2]
        assert [c['day'] for c in weeks[-1]] == [31] + [None] * 6
        assert cg.WEEKDAYS[0] == 'Mon'

    def test_padding_days_are_empty(self):
        pad = cg.month_cells(EMPTY, 2027, 5, {})[0][0]
        assert pad == {'day': None, 'date': None, 'closed': False, 'holiday': '', 'today': False, 'events': []}

    def test_every_day_of_the_month_present(self):
        days = [c['day'] for c in _days(cg.month_cells(EMPTY, 2028, 2, {})) if c['day']]
        assert days == list(range(1, 30))  # 2028 is a leap year

    def test_weekend_and_named_holiday_are_closed(self):
        weeks = cg.month_cells(EMPTY, 2027, 5, {dt.date(2027, 5, 13): 'Ascension Day'})
        assert _cell(weeks, 1)['closed'] and _cell(weeks, 1)['holiday'] == ''  # Saturday
        assert _cell(weeks, 13)['closed'] and _cell(weeks, 13)['holiday'] == 'Ascension Day'
        assert not _cell(weeks, 12)['closed']

    def test_closed_dates_as_a_list(self):
        weeks = cg.month_cells(EMPTY, 2027, 5, [pd.Timestamp('2027-05-12')])
        assert _cell(weeks, 12)['closed'] and _cell(weeks, 12)['holiday'] == ''

    def test_events_sorted_by_yield_and_scoped_to_month(self):
        df = _df([
            ('2027-05-12', 'SMGR.JK', 4.0, 300),
            ('2027-05-12', 'PTBA.JK', 11.2, 296),
            ('2027-05-12', 'ICBP.JK', 2.0, 100),
            ('2027-06-12', 'JUNE.JK', 9.0, 10),
            ('2026-05-12', 'LAST.JK', 9.0, 10),
        ])
        events = _cell(cg.month_cells(df, 2027, 5, {}), 12)['events']
        assert [e['symbol'] for e in events] == ['PTBA.JK', 'SMGR.JK', 'ICBP.JK']
        assert events[0] == {'symbol': 'PTBA.JK', 'yield': 11.2, 'dividend': 296.0}
        assert sum(len(c['events']) for c in _days(cg.month_cells(df, 2027, 5, {}))) == 3

    def test_today_marked_only_in_its_month(self):
        weeks = cg.month_cells(EMPTY, 2027, 5, {}, today=dt.date(2027, 5, 12))
        assert [c['day'] for c in _days(weeks) if c['today']] == [12]
        other = cg.month_cells(EMPTY, 2027, 6, {}, today=dt.date(2027, 5, 12))
        assert not any(c['today'] for c in _days(other))
        assert not any(c['today'] for c in _days(cg.month_cells(EMPTY, 2027, 5, {})))


def _grid(df, closed=None, sl='JKSE', market='ID', **kwargs):
    cells = cg.month_cells(df, 2027, 5, closed or {})
    return cg.month_grid_html(cells, sl, market, 'https://example.com', **kwargs)


class TestMonthGridHtml:

    def test_empty_month_renders_all_days_without_chips(self):
        markup = _grid(EMPTY)
        assert markup.count('class="pdx-dow"') == 7
        assert all(f'<b>{d}</b>' in markup for d in range(1, 32))
        assert 'pdx-ev' not in markup
        # 5 padding days before the 1st, 6 after the 31st
        assert markup.count('pdx-day-off') == 11

    def test_chip_link_title_and_colours(self):
        markup = _grid(_df([('2027-05-12', 'PTBA.JK', 11.2, 296)]))
        assert 'href="https://example.com/stock_picker?market=ID&amp;stock=PTBA.JK"' in markup
        assert 'title="12 May: PTBA (11.20%), dividend 296"' in markup
        assert '>PTBA</a>' in markup
        background = real_plot._scale_color(11.2, real_plot.dividend_calendar_domain('JKSE'),
                                            real_plot.DIVIDEND_CALENDAR_COLORS)
        assert f'background:{background};color:{real_plot.readable_text_color(background)}' in markup

    def test_yield_is_clamped_to_the_scale(self):
        markup = _grid(_df([('2027-05-12', 'HIGH.JK', 60.0, 1), ('2027-05-13', 'LOW.JK', 0.1, 1)]))
        colors = real_plot.DIVIDEND_CALENDAR_COLORS
        assert f'background:{colors[-1]}' in markup
        assert f'background:{colors[0]}' in markup

    def test_overflow_chip_counts_and_lists_the_rest(self):
        df = _df([('2027-05-12', f'S{i}', float(i), 1) for i in range(1, 6)])
        markup = _grid(df, sl='S&P500', market='US', max_events=3)
        assert markup.count('<a class="pdx-ev"') == 3
        assert '>S5</a>' in markup and '>S1</a>' not in markup
        assert '+2<span class="pdx-ev-more-word"> more</span>' in markup
        assert 'title="12 May: S2 (2.00%), S1 (1.00%)"' in markup

    def test_overflow_chip_sits_on_the_day_number_line(self):
        df = _df([('2027-05-12', f'S{i}', float(i), 1) for i in range(1, 6)])
        markup = _grid(df, sl='S&P500', market='US', max_events=3)
        assert ('<div class="pdx-day-hd"><b>12</b><span class="pdx-ev pdx-ev-more" '
                'title="12 May: S2 (2.00%), S1 (1.00%)">+2<span class="pdx-ev-more-word"> more</span></span></div>'
                '<a class="pdx-ev"') in markup
        # A day without overflow has only its number on that line
        assert '<div class="pdx-day-hd"><b>13</b></div>' in markup

    @pytest.mark.parametrize('year,month,weeks', [(2027, 2, 4), (2027, 5, 6), (2027, 6, 5)])
    def test_grid_carries_its_week_count(self, year, month, weeks):
        cells = cg.month_cells(EMPTY, year, month, {})
        assert len(cells) == weeks
        assert f'<div class="pdx-cal" style="--pdx-weeks:{weeks}">' in cg.month_grid_html(cells, 'JKSE', 'ID', '')

    def test_no_overflow_chip_at_the_limit(self):
        df = _df([('2027-05-12', f'S{i}', float(i), 1) for i in range(3)])
        assert 'pdx-ev-more' not in _grid(df)

    def test_closed_day_class_and_holiday_title(self):
        markup = _grid(EMPTY, closed={dt.date(2027, 5, 13): 'Ascension Day'})
        assert '<div class="pdx-day pdx-day-closed" title="Ascension Day"><div class="pdx-day-hd"><b>13</b>' in markup
        assert '<div class="pdx-day pdx-day-closed"><div class="pdx-day-hd"><b>1</b>' in markup  # Saturday, no title
        assert '<div class="pdx-day"><div class="pdx-day-hd"><b>12</b>' in markup

    def test_today_class(self):
        cells = cg.month_cells(EMPTY, 2027, 5, {}, today=dt.date(2027, 5, 12))
        assert '<div class="pdx-day pdx-day-today"><div class="pdx-day-hd"><b>12</b>' in cg.month_grid_html(cells, 'JKSE', 'ID', '')

    def test_symbols_and_holiday_names_are_escaped(self):
        df = _df([('2027-05-12', '<b>&"X', 5.0, 1)])
        markup = _grid(df, closed={dt.date(2027, 5, 13): '<script>alert("x")</script>'})
        assert '<script>' not in markup and '<b>&' not in markup
        assert '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;' in markup
        assert '>&lt;b&gt;&amp;&quot;X</a>' in markup
        # The link's query value is percent-encoded
        assert 'stock=%3Cb%3E%26%22X' in markup


class TestLegendHtml:

    @pytest.mark.parametrize('sl,last', [('JKSE', '12%+'), ('S&P500', '2%+')])
    def test_seven_swatches_last_with_plus(self, sl, last):
        markup = cg.legend_html(sl)
        assert markup.count('pdx-legend-key') == 7
        assert f'>{last}</span>' in markup
        for color in real_plot.DIVIDEND_CALENDAR_COLORS:
            assert f'background:{color.lower()}' in markup.lower()


@pytest.mark.parametrize('value,expected', [(350.0, '350'), (0.24, '0.24'), (0.2635, '0.2635'), (1250.5, '1,250.5')])
def test_format_dividend(value, expected):
    assert cg.format_dividend(value) == expected
