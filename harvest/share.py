"""Shareable Dividend Calendar images and the link-preview pages that carry them.

A share id names one Full Year calendar: ``{exch}-{year}`` with an optional ``-syariah`` suffix
(JKSE only), e.g. ``jkse-2025``, ``sp500-2024`` or ``jkse-2025-syariah``. ``year`` is the source
year, the same as the calendar page's ``?year=`` param; the image shows the year after it.

server.py serves ``/share/calendar/<id>.png`` (the image) and ``/share/calendar/<id>`` (a small page
whose Open Graph tags make the link unfurl with that image, and which sends people on to the live
calendar on the same host). Nothing here imports Streamlit, so it is safe to call from Starlette route handlers.
"""
import io
import os
import re
import time
import html
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from urllib.parse import urlencode, urlsplit

import redis
import altair as alt

import harvest.plot as hp
from harvest import chart_style as cs
from harvest.calendar_data import add_yield, decode_frame, market_holidays, project_to_next_year, syariah_symbols


logger = logging.getLogger('share')

SITE_URL = os.getenv('SITE_URL', 'https://panendividen.com').rstrip('/')

# main.py registers apps/calendar/calendar.py without a url_path, so Streamlit serves it at /calendar
CALENDAR_PAGE_PATH = '/calendar'
SHARE_PATH = '/share/calendar'

# Stock list per exchange key: (label on the page and in the image, holiday calendar market)
EXCHANGES = {'jkse': ('JKSE', 'ID'), 'sp500': ('S&P500', 'US')}

# Strict on purpose: the id ends up in URLs, HTML and Redis keys, so nothing else gets through
_SHARE_ID = re.compile(r'(jkse|sp500)-([0-9]{4})(-syariah)?')  # [0-9], not \d, which also matches non-ASCII digits
_MIN_YEAR, _MAX_YEAR = 2000, 2100

# Link previews (Facebook, X, LinkedIn, WhatsApp) crop to about 1.91:1, so the image is that
# shape at twice 1200x630 for sharp text; the 4x3 month grid is scaled to fit, never stretched
IMAGE_WIDTH, IMAGE_HEIGHT = 2400, 1260
_RENDER_SCALE = 2
_IMAGE_PADDING = 28  # chart pixels around the grid, before scaling
_BACKGROUND = cs.CHART_SURFACE  # the page canvas the calendar greens were picked against

# The same per-month size limits the page uses (CALENDAR_MONTH_WIDTH/HEIGHT in calendar.py)
_MONTH_WIDTH, _MONTH_HEIGHT = 250, 200


@dataclass(frozen=True)
class CalendarShare:
    exch: str
    year: int
    syariah: bool = False

    @property
    def share_id(self):
        return f'{self.exch}-{self.year}' + ('-syariah' if self.syariah else '')

    @property
    def stock_list(self):
        return EXCHANGES[self.exch][0]

    @property
    def market(self):
        return EXCHANGES[self.exch][1]

    @property
    def projected_year(self):
        return self.year + 1

    @property
    def label(self):
        """'JKSE', 'JKSE Syariah' or 'S&P500'."""
        return self.stock_list + (' Syariah' if self.syariah else '')


def parse_share_id(share_id):
    """CalendarShare for a valid id, else None. Only lowercase ids in the documented shape pass."""
    if not isinstance(share_id, str) or len(share_id) > 24:
        return None
    m = _SHARE_ID.fullmatch(share_id)
    if not m:
        return None
    exch, year, syariah = m.group(1), int(m.group(2)), bool(m.group(3))
    if syariah and exch != 'jkse':
        return None
    if not _MIN_YEAR <= year <= _MAX_YEAR:
        return None
    return CalendarShare(exch, year, syariah)


def make_share_id(exch, year, syariah=False):
    """Share id for the calendar page's current selection; the Syariah flag only applies to JKSE."""
    share_id = f'{exch}-{int(year)}' + ('-syariah' if syariah and exch == 'jkse' else '')
    if parse_share_id(share_id) is None:
        raise ValueError(f'not a shareable calendar: {exch!r} {year!r}')
    return share_id


def image_path(share_id):
    """Same-origin path of the PNG, for fetches from the page (works on localhost too)."""
    return f'{SHARE_PATH}/{share_id}.png'


def image_url(share_id, site_url=None):
    return f'{site_url or SITE_URL}{image_path(share_id)}'


def share_path(share_id):
    """Same-origin path of the link-preview page; the share bar resolves it against the page's own host."""
    return f'{SHARE_PATH}/{share_id}'


def share_url(share_id, site_url=None):
    """Absolute link-preview URL on SITE_URL, for the page's og:url."""
    return f'{site_url or SITE_URL}{share_path(share_id)}'


def calendar_page_url(share):
    """Host-relative link to the live Full Year calendar for the shared selection.

    Relative, so the share page sends people on within whichever host served it (localhost, staging).
    list= and syariah= are only added when they differ from the page's defaults (JKSE, unfiltered);
    the utm tags let main.py's VISIT log show the source.
    """
    params = {'view': 'yearly', 'year': share.year}
    if share.exch != 'jkse':
        params['list'] = share.exch
    if share.syariah:
        params['syariah'] = '1'
    params.update({'utm_source': 'share', 'utm_medium': 'calendar_image'})
    return f'{CALENDAR_PAGE_PATH}?{urlencode(params)}'


def share_filename(share):
    return f'dividend-calendar-{share.projected_year}-{share.exch}' + ('-syariah' if share.syariah else '') + '.png'


def site_host(site_url=None):
    """'panendividen.com' from 'https://panendividen.com'."""
    url = site_url or SITE_URL
    return urlsplit(url).netloc or url


def share_title(share):
    return f'Dividend Calendar {share.projected_year} · {share.label}'


def share_description(share):
    return (f'Projected ex-dividend dates for {share.label} stocks in {share.projected_year}, '
            f'based on what each company paid in {share.year}. Darker green means a higher dividend yield.')


def share_text(share):
    """Short text that goes with the native share sheet."""
    return f'{share_title(share)}, projected ex-dividend dates on {site_host()}'


# ---------------------------------------------------------------------------
# Data and chart
# ---------------------------------------------------------------------------

def _redis_get(key):
    # Same connection settings as seo.py: route handlers run outside any Streamlit session
    r = redis.from_url(os.environ['REDIS_URL'], socket_connect_timeout=10, socket_timeout=30)
    return r.get(key)


def load_calendar_frame(share):
    """The dividends the page's Full Year view draws for this selection, or None when there is no data."""
    raw = _redis_get(f'div_cal_{share.exch}_{share.year}')
    if raw is None:
        return None
    df = decode_frame(raw)
    if df.empty or 'date' not in df.columns:
        return None
    # The same steps as calendar.py: project onto next year's trading days, add yields, then filter
    df['date'] = project_to_next_year(df['date'], market=share.market)
    df = add_yield(df)

    if share.syariah:
        raw_score = _redis_get(f'div_score_{share.exch}')
        symbols = syariah_symbols(decode_frame(raw_score)) if raw_score is not None else None
        if symbols is None:
            return None
        df = df[df['symbol'].isin(symbols)]
    return None if df.empty else df.reset_index(drop=True)


def calendar_chart(df, share):
    """The page's Full Year chart with the site attribution under the grid, on a solid background."""
    # theme=None on the page means the app's Altair theme is what styles the chart there, so apply it here too
    hp.enable_chart_theme()
    chart = hp.plot_dividend_calendar(
        df, sl=share.stock_list, closed_dates=market_holidays(share.market, [share.projected_year]),
        width=_MONTH_WIDTH, height=_MONTH_HEIGHT, title=share_title(share), legend=True,
    )
    attribution = alt.TitleParams(
        f'{site_host()} · Dividend Calendar', orient='bottom', anchor='end',
        fontSize=13, fontWeight=600, color=cs.CHART_MUTED_TEXT, offset=10,
    )
    return chart.properties(title=attribution, background=_BACKGROUND, padding=_IMAGE_PADDING)


def _vega_lite_version():
    """'6.4' from Altair's 'v6.4.1', so vl-convert compiles with the Vega-Lite the spec was written for."""
    return '.'.join(alt.SCHEMA_VERSION.lstrip('v').split('.')[:2])


def render_png(chart):
    """PNG bytes of the chart centred on an IMAGE_WIDTH x IMAGE_HEIGHT canvas in the background colour."""
    # Imported here so the Streamlit page, which only needs the ids and URLs, never loads the renderer
    import vl_convert as vlc
    from PIL import Image

    png = vlc.vegalite_to_png(chart.to_dict(), vl_version=_vega_lite_version(), scale=_RENDER_SCALE,
                              allowed_base_urls=[])
    grid = Image.open(io.BytesIO(png)).convert('RGBA')

    # Pad the grid out to the preview's shape first, then scale, so nothing is stretched or cropped
    ratio = IMAGE_WIDTH / IMAGE_HEIGHT
    width, height = max(grid.width, round(grid.height * ratio)), max(grid.height, round(grid.width / ratio))
    canvas = Image.new('RGBA', (width, height), _BACKGROUND)
    canvas.alpha_composite(grid, ((width - grid.width) // 2, (height - grid.height) // 2))
    canvas = canvas.convert('RGB').resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)

    out = io.BytesIO()
    canvas.save(out, format='PNG', optimize=True)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class _PngCache:
    """Small thread-safe LRU with a TTL; the calendar data changes daily and an image is ~0.5 MB."""

    def __init__(self, max_entries, ttl_seconds):
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            if time.monotonic() >= hit[0]:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return hit[1]

    def set(self, key, value):
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()


_png_cache = _PngCache(max_entries=8, ttl_seconds=60 * 60)


def calendar_share_png(share_id):
    """PNG bytes for a share id, or None for an invalid id or a year with no data.

    Missing data is not cached, so the image appears as soon as the pipeline writes the key.
    """
    share = parse_share_id(share_id)
    if share is None:
        return None
    png = _png_cache.get(share.share_id)
    if png is not None:
        return png

    df = load_calendar_frame(share)
    if df is None:
        return None
    started = time.monotonic()
    png = render_png(calendar_chart(df, share))
    logger.info('rendered share image %s in %.2fs (%d bytes)', share.share_id, time.monotonic() - started, len(png))
    _png_cache.set(share.share_id, png)
    return png


# ---------------------------------------------------------------------------
# Link-preview page
# ---------------------------------------------------------------------------

def render_share_page(share):
    """HTML with Open Graph / Twitter tags for link previews; browsers go straight on to the live calendar.

    The tags are absolute on SITE_URL (crawlers need that); the redirect is host-relative, so it stays
    on the host that served this page.
    """
    esc = lambda value: html.escape(str(value), quote=True)  # noqa: E731
    title = share_title(share)
    description = share_description(share)
    target = calendar_page_url(share)
    canonical = f'{SITE_URL}{CALENDAR_PAGE_PATH}'
    image = image_url(share.share_id)
    page = share_url(share.share_id)
    # json-style quoting for the script: the target is built from validated parts, and escaping
    # "<" keeps "</script>" from ever closing the tag early
    target_js = '"' + target.replace('\\', '\\\\').replace('"', '\\"').replace('<', '\\u003c') + '"'
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} - Panen Dividen</title>
<meta name="description" content="{esc(description)}">
<link rel="canonical" href="{esc(canonical)}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Panen Dividen">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}">
<meta property="og:url" content="{esc(page)}">
<meta property="og:image" content="{esc(image)}">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="{IMAGE_WIDTH}">
<meta property="og:image:height" content="{IMAGE_HEIGHT}">
<meta property="og:image:alt" content="{esc(title)}: a 12-month grid of projected ex-dividend dates">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{esc(title)}">
<meta name="twitter:description" content="{esc(description)}">
<meta name="twitter:image" content="{esc(image)}">
<meta http-equiv="refresh" content="0; url={esc(target)}">
<script>location.replace({target_js});</script>
</head>
<body>
<p><a href="{esc(target)}">Open the {esc(title)}</a></p>
</body>
</html>
'''
