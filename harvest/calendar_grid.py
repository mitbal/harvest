"""HTML month grid for the Payout Calendar's Single Month view.

``month_cells`` turns dividend rows into weeks of day dicts; ``month_grid_html`` renders them with
the ``pdx-`` calendar classes from ``harvest.page_ui``. Chip colours use the same seven-step yield
scale as the Altair calendar in ``harvest.plot``, so both views agree.
"""
import calendar
from urllib.parse import quote

import pandas as pd

import harvest.plot as hp
from harvest.page_ui import esc


# Monday first, like hp.month_plot (list(calendar.day_abbr))
WEEKDAYS = list(calendar.day_abbr)


def format_dividend(value):
    """Dividend per share without trailing zeros: 350, 0.24, 0.2635."""
    return f'{value:,.4f}'.rstrip('0').rstrip('.')


def display_ticker(symbol):
    return str(symbol).removesuffix('.JK')


def yield_colors(pct, sl):
    """(background, text) for a yield (%) on the calendar's seven-step scale, clamped to its ends."""
    background = hp._scale_color(pct, hp.dividend_calendar_domain(sl), hp.DIVIDEND_CALENDAR_COLORS)
    return background, hp.readable_text_color(background)


def _padding_day():
    return {'day': None, 'date': None, 'closed': False, 'holiday': '', 'today': False, 'events': []}


def month_cells(df, year, month, closed_dates, today=None):
    """Weeks (Monday first) of seven day dicts for one month.

    Each dict has ``day`` (int, or None for padding), ``date``, ``closed`` (weekend or in
    closed_dates), ``holiday`` (name or ''), ``today`` and ``events``: that day's rows, highest
    yield first, as dicts with ``symbol``, ``yield`` and ``dividend``. df needs the columns date,
    symbol, yield and adjDividend. closed_dates is a list of dates or a {date: name} dict.
    """
    if closed_dates is None:
        closed_dates = {}
    elif not isinstance(closed_dates, dict):
        closed_dates = dict.fromkeys(closed_dates, '')
    holidays = {pd.Timestamp(d).date(): name or '' for d, name in closed_dates.items()}
    today = pd.Timestamp(today).date() if today is not None else None

    rows = df[(df['date'].dt.year == year) & (df['date'].dt.month == month)]
    rows = rows.sort_values('yield', ascending=False, kind='stable')
    events = {}
    for date, symbol, pct, dividend in zip(rows['date'], rows['symbol'], rows['yield'], rows['adjDividend']):
        events.setdefault(date.day, []).append({'symbol': symbol, 'yield': float(pct), 'dividend': float(dividend)})

    weeks = []
    for week in calendar.Calendar(firstweekday=0).monthdatescalendar(year, month):
        weeks.append([
            {
                'day': date.day,
                'date': date,
                'closed': date.weekday() >= 5 or date in holidays,
                'holiday': holidays.get(date, ''),
                'today': date == today,
                'events': events.get(date.day, []),
            } if date.month == month else _padding_day()
            for date in week
        ])
    return weeks


def _event_label(event):
    return f"{display_ticker(event['symbol'])} ({event['yield']:.2f}%)"


def _chip(event, day_label, sl, market, site_url):
    background, text = yield_colors(event['yield'], sl)
    href = f"{site_url}/stock_picker?market={quote(str(market))}&stock={quote(str(event['symbol']))}"
    title = f"{day_label}: {_event_label(event)}, dividend {format_dividend(event['dividend'])}"
    return (
        f'<a class="pdx-ev" href="{esc(href)}" title="{esc(title)}" '
        f'style="background:{background};color:{text}">{esc(display_ticker(event["symbol"]))}</a>'
    )


def _day_html(cell, sl, market, site_url, max_events):
    if cell['day'] is None:
        return '<div class="pdx-day pdx-day-off"></div>'

    classes = 'pdx-day' + (' pdx-day-closed' if cell['closed'] else '') + (' pdx-day-today' if cell['today'] else '')
    title = f' title="{esc(cell["holiday"])}"' if cell['holiday'] else ''
    day_label = f"{cell['day']} {calendar.month_abbr[cell['date'].month]}"

    shown, rest = cell['events'][:max_events], cell['events'][max_events:]
    chips = ''.join(_chip(event, day_label, sl, market, site_url) for event in shown)
    # The "+N more" pill shares the day-number line, so a busy day is one line shorter
    more = ''
    if rest:
        rest_title = f"{day_label}: " + ', '.join(_event_label(event) for event in rest)
        more = (
            f'<span class="pdx-ev pdx-ev-more" title="{esc(rest_title)}">'
            f'+{len(rest)}<span class="pdx-ev-more-word"> more</span></span>'
        )
    return f'<div class="{classes}"{title}><div class="pdx-day-hd"><b>{cell["day"]}</b>{more}</div>{chips}</div>'


def month_grid_html(cells, sl, market, site_url, max_events=3):
    """Grid markup: a weekday header row, one cell per day, one chip per event.

    A day with more than max_events events shows the highest yields and a "+N more" chip whose
    title lists the rest, on the day-number line. Each chip links to the stock's page on site_url.
    The grid carries its week count as --pdx-weeks, which the row height in page_ui uses to fit
    the month into the window.
    """
    header = ''.join(f'<span class="pdx-dow">{day}</span>' for day in WEEKDAYS)
    days = ''.join(_day_html(cell, sl, market, site_url, max_events) for week in cells for cell in week)
    return (
        f'<div class="pdx-cal-wrap"><div class="pdx-cal" style="--pdx-weeks:{len(cells)}">'
        f'{header}{days}</div></div>'
    )


def legend_html(sl):
    """The seven colour swatches with their yield breakpoints; the last one reads "12%+"."""
    domain = hp.dividend_calendar_domain(sl)
    keys = []
    for i, value in enumerate(domain):
        background, text = yield_colors(value, sl)
        label = f'{value:g}%' + ('+' if i == len(domain) - 1 else '')
        keys.append(f'<span class="pdx-ev pdx-legend-key" style="background:{background};color:{text}">{label}</span>')
    return f'<div class="pdx-cal-legend"><span class="pdx-note">Yield</span>{"".join(keys)}</div>'
