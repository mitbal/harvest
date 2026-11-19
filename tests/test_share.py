"""
tests/test_share.py
Shareable Dividend Calendar image (harvest/share.py), its routes in server.py and the share bar helper.

Redis is faked and harvest.plot is stubbed here (see tests/conftest.py), so the chart is a small
stand-in; apptests/test_share_image.py renders the real 4x3 calendar in a process with real modules.
"""
import io
import json
from unittest.mock import patch

import altair as alt
import pandas as pd
import pytest
from PIL import Image
from starlette.applications import Starlette
from starlette.testclient import TestClient

from harvest import page_ui as ui
from harvest import share
import server


PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
BOT_UA = 'facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)'
HUMAN_UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 '
            '(KHTML, like Gecko) Version/18.0 Safari/605.1.15')


@pytest.fixture(autouse=True)
def clear_png_cache():
    share._png_cache.clear()
    yield
    share._png_cache.clear()


def small_chart(df, calendar_share=None):
    """Stand-in for the stubbed calendar chart, finished the same way calendar_chart finishes it."""
    return alt.Chart(df).mark_rect().encode(x='day:O', y='yield:Q').properties(
        width=200, height=120, background=share._BACKGROUND, padding=share._IMAGE_PADDING,
    )


SMALL_DF = pd.DataFrame({'day': [1, 2, 3], 'yield': [2.5, 6.0, 9.5]})


# ═════════════════════════════════════════════════════════════════════════════

class TestParseShareId:

    @pytest.mark.parametrize('share_id, expected', [
        ('jkse-2025', share.CalendarShare('jkse', 2025, False)),
        ('sp500-2024', share.CalendarShare('sp500', 2024, False)),
        ('jkse-2025-syariah', share.CalendarShare('jkse', 2025, True)),
    ])
    def test_valid(self, share_id, expected):
        parsed = share.parse_share_id(share_id)
        assert parsed == expected
        assert parsed.share_id == share_id

    @pytest.mark.parametrize('share_id', [
        '', 'jkse', 'jkse-', 'jkse-25', 'jkse-20250', 'nyse-2025', 'JKSE-2025', 'jkse-2025-SYARIAH',
        'sp500-2024-syariah',          # Syariah filter is JKSE only
        'jkse-1999', 'jkse-2101',       # outside the plausible years
        'jkse-2025-', 'jkse-2025-syariah-x', 'jkse_2025', ' jkse-2025', 'jkse-2025\n',
        'jkse-2025.png', 'jkse-٢٠٢٥',   # Arabic-Indic digits match \\d in Python
        'jkse-2025"><script>alert(1)</script>', "jkse-2025' onload='x", 'jkse-2025/../../etc/passwd',
        'jkse-2025%22', 'jkse-2025?x=1', 'jkse-2025#x',
        None, 2025, b'jkse-2025',
    ])
    def test_invalid(self, share_id):
        assert share.parse_share_id(share_id) is None

    def test_rejects_long_input_before_matching(self):
        assert share.parse_share_id('jkse-2025' + '-syariah' * 1000) is None


class TestMakeShareId:

    def test_shapes(self):
        assert share.make_share_id('jkse', 2025) == 'jkse-2025'
        assert share.make_share_id('jkse', 2025, syariah=True) == 'jkse-2025-syariah'
        assert share.make_share_id('sp500', 2024) == 'sp500-2024'

    def test_syariah_ignored_outside_jkse(self):
        assert share.make_share_id('sp500', 2024, syariah=True) == 'sp500-2024'

    def test_numpy_year(self):
        import numpy as np
        assert share.make_share_id('jkse', np.int64(2025)) == 'jkse-2025'

    @pytest.mark.parametrize('exch, year', [('nyse', 2025), ('jkse', 1999), ('jkse"', 2025)])
    def test_invalid_raises(self, exch, year):
        with pytest.raises(ValueError):
            share.make_share_id(exch, year)


class TestUrls:

    def test_paths_and_urls(self):
        with patch.object(share, 'SITE_URL', 'https://panendividen.com'):
            assert share.image_path('jkse-2025') == '/share/calendar/jkse-2025.png'
            assert share.image_url('jkse-2025') == 'https://panendividen.com/share/calendar/jkse-2025.png'
            assert share.share_path('jkse-2025') == '/share/calendar/jkse-2025'
            assert share.share_url('jkse-2025') == 'https://panendividen.com/share/calendar/jkse-2025'

    @pytest.mark.parametrize('share_id, query', [
        ('jkse-2025', 'view=yearly&year=2025'),                    # JKSE is the page's default, so no list=
        ('sp500-2024', 'view=yearly&year=2024&list=sp500'),
        ('jkse-2025-syariah', 'view=yearly&year=2025&syariah=1'),
    ])
    def test_calendar_page_url_is_relative_and_keeps_the_selection(self, share_id, query):
        # Host-relative, so the redirect stays on whichever host served the share page
        with patch.object(share, 'SITE_URL', 'https://panendividen.com'):
            url = share.calendar_page_url(share.parse_share_id(share_id))
        assert url == f'/calendar?{query}&utm_source=share&utm_medium=calendar_image'

    def test_site_host(self):
        assert share.site_host('https://panendividen.com') == 'panendividen.com'
        assert share.site_host('http://localhost:8501') == 'localhost:8501'

    def test_titles_use_projected_year(self):
        s = share.parse_share_id('jkse-2025-syariah')
        assert share.share_title(s) == 'Dividend Calendar 2026 · JKSE Syariah'
        assert share.share_filename(s) == 'dividend-calendar-2026-jkse-syariah.png'
        assert share.share_filename(share.parse_share_id('sp500-2024')) == 'dividend-calendar-2025-sp500.png'
        assert 'based on what each company paid in 2025' in share.share_description(s)


class TestSharePage:

    def page(self, share_id='jkse-2025', site='https://panendividen.com'):
        with patch.object(share, 'SITE_URL', site):
            return share.render_share_page(share.parse_share_id(share_id))

    def test_open_graph_and_twitter_tags(self):
        html = self.page('sp500-2024')
        assert '<meta property="og:title" content="Dividend Calendar 2025 · S&amp;P500">' in html
        assert '<meta property="og:image" content="https://panendividen.com/share/calendar/sp500-2024.png">' in html
        assert f'<meta property="og:image:width" content="{share.IMAGE_WIDTH}">' in html
        assert f'<meta property="og:image:height" content="{share.IMAGE_HEIGHT}">' in html
        assert '<meta property="og:url" content="https://panendividen.com/share/calendar/sp500-2024">' in html
        assert '<meta name="twitter:card" content="summary_large_image">' in html
        assert 'property="og:description"' in html
        assert '<link rel="canonical" href="https://panendividen.com/calendar">' in html

    def test_redirects_people_to_the_live_calendar(self):
        html = self.page('jkse-2025-syariah')
        target = '/calendar?view=yearly&year=2025&syariah=1&utm_source=share&utm_medium=calendar_image'
        assert f'<meta http-equiv="refresh" content="0; url={target.replace("&", "&amp;")}">' in html
        assert f'location.replace("{target}")' in html
        assert f'href="{target.replace("&", "&amp;")}"' in html

    def test_redirect_stays_on_the_serving_host(self):
        # On localhost or staging the redirect must not jump to SITE_URL; the preview tags stay absolute
        html = self.page('sp500-2024', site='https://panendividen.com')
        target = '/calendar?view=yearly&year=2024&list=sp500&utm_source=share&utm_medium=calendar_image'
        assert f'location.replace("{target}")' in html
        assert 'url=https://' not in html and 'href="https://panendividen.com/calendar?' not in html
        assert '<link rel="canonical" href="https://panendividen.com/calendar">' in html
        assert '<meta name="twitter:image" content="https://panendividen.com/share/calendar/sp500-2024.png">' in html

    def test_escapes_interpolated_values(self):
        # SITE_URL comes from the environment, not the request, but it is escaped all the same
        html = self.page(site='https://x.test/"><script>alert(1)</script>')
        assert '<script>alert(1)' not in html
        assert '&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;' in html
        # The inline script only carries the relative target, so SITE_URL never reaches it
        script = html.split('<script>', 1)[1].split('</script>', 1)[0]
        assert script == 'location.replace("/calendar?view=yearly&year=2025&utm_source=share&utm_medium=calendar_image");'


class TestLoadCalendarFrame:

    ROWS = [
        {'date': '2025-03-10', 'symbol': 'BBRI.JK', 'adjDividend': 300.0, 'price': 5000.0},
        {'date': '2025-06-10', 'symbol': 'TLKM.JK', 'adjDividend': 150.0, 'price': 3000.0},
        {'date': '2025-07-10', 'symbol': 'ZERO.JK', 'adjDividend': 0.0, 'price': 3000.0},
    ]
    SCORES = [{'symbol': 'BBRI.JK', 'is_syariah': False}, {'symbol': 'TLKM.JK', 'is_syariah': True}]

    def fake_redis(self, data):
        return patch.object(share, '_redis_get', side_effect=lambda key: data.get(key))

    def test_projects_and_adds_yield(self):
        with self.fake_redis({'div_cal_jkse_2025': json.dumps(self.ROWS)}):
            df = share.load_calendar_frame(share.parse_share_id('jkse-2025'))
        assert df['symbol'].tolist() == ['BBRI.JK', 'TLKM.JK']  # the zero dividend is dropped
        assert (df['date'].dt.year == 2026).all()
        assert df.loc[0, 'yield'] == pytest.approx(6.0)

    def test_reads_the_date_content_envelope(self):
        envelope = json.dumps({'date': '2026-01-02', 'content': json.dumps(self.ROWS)})
        with self.fake_redis({'div_cal_sp500_2024': envelope}):
            df = share.load_calendar_frame(share.parse_share_id('sp500-2024'))
        assert len(df) == 2

    def test_syariah_filter(self):
        data = {'div_cal_jkse_2025': json.dumps(self.ROWS), 'div_score_jkse': json.dumps(self.SCORES)}
        with self.fake_redis(data):
            df = share.load_calendar_frame(share.parse_share_id('jkse-2025-syariah'))
        assert df['symbol'].tolist() == ['TLKM.JK']

    @pytest.mark.parametrize('data', [
        {},                                                             # year not in Redis
        {'div_cal_jkse_2025': json.dumps([])},                          # empty year
        {'div_cal_jkse_2025': json.dumps([{'symbol': 'A', 'adjDividend': 1.0, 'price': 0.0, 'date': '2025-01-02'}])},
    ])
    def test_no_data_gives_none(self, data):
        with self.fake_redis(data):
            assert share.load_calendar_frame(share.parse_share_id('jkse-2025')) is None

    def test_syariah_without_flag_gives_none(self):
        data = {'div_cal_jkse_2025': json.dumps(self.ROWS), 'div_score_jkse': json.dumps([{'symbol': 'BBRI.JK'}])}
        with self.fake_redis(data):
            assert share.load_calendar_frame(share.parse_share_id('jkse-2025-syariah')) is None


class TestCalendarSharePng:

    @pytest.fixture
    def fake_chart(self):
        with patch.object(share, 'load_calendar_frame', return_value=SMALL_DF) as load, \
             patch.object(share, 'calendar_chart', side_effect=small_chart):
            yield load

    def test_renders_landscape_png_on_solid_background(self, fake_chart):
        png = share.calendar_share_png('jkse-2025')
        assert png.startswith(PNG_SIGNATURE)
        image = Image.open(io.BytesIO(png))
        assert image.size == (share.IMAGE_WIDTH, share.IMAGE_HEIGHT)
        assert image.mode == 'RGB'  # no alpha channel, so no transparent background
        assert image.getpixel((0, 0)) == (0xEF, 0xF1, 0xEF)

    def test_cached_after_first_render(self, fake_chart):
        first = share.calendar_share_png('jkse-2025')
        assert share.calendar_share_png('jkse-2025') is first
        assert fake_chart.call_count == 1

    def test_invalid_id_never_loads(self, fake_chart):
        assert share.calendar_share_png('jkse-2025"') is None
        fake_chart.assert_not_called()

    def test_missing_data_is_not_cached(self):
        with patch.object(share, 'load_calendar_frame', side_effect=[None, SMALL_DF]) as load, \
             patch.object(share, 'calendar_chart', side_effect=small_chart):
            assert share.calendar_share_png('jkse-2030') is None
            assert share.calendar_share_png('jkse-2030').startswith(PNG_SIGNATURE)
        assert load.call_count == 2

    def test_vega_lite_version_matches_altair(self):
        assert share._vega_lite_version() == '.'.join(alt.SCHEMA_VERSION.lstrip('v').split('.')[:2])


class TestPngCache:

    def test_ttl_and_max_entries(self):
        cache = share._PngCache(max_entries=2, ttl_seconds=100)
        clock = [0.0]
        with patch.object(share.time, 'monotonic', side_effect=lambda: clock[0]):
            cache.set('a', b'1')
            cache.set('b', b'2')
            assert cache.get('a') == b'1'  # a is now the most recent
            cache.set('c', b'3')
            assert cache.get('b') is None  # least recently used went first
            assert cache.get('a') == b'1'
            clock[0] = 101
            assert cache.get('a') is None


# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def client():
    return TestClient(Starlette(routes=server.SHARE_ROUTES))


class TestShareRoutes:

    def test_png(self, client):
        with patch.object(share, 'calendar_share_png', return_value=PNG_SIGNATURE + b'rest') as render:
            res = client.get('/share/calendar/jkse-2025.png')
        assert res.status_code == 200
        assert res.headers['content-type'] == 'image/png'
        assert 'max-age' in res.headers['cache-control']
        assert res.content.startswith(PNG_SIGNATURE)
        render.assert_called_once_with('jkse-2025')

    def test_png_missing_data_is_404(self, client):
        with patch.object(share, 'calendar_share_png', return_value=None):
            assert client.get('/share/calendar/jkse-2099.png').status_code == 404

    @pytest.mark.parametrize('path', [
        '/share/calendar/nyse-2025.png', '/share/calendar/sp500-2024-syariah.png',
        '/share/calendar/jkse-2025%22%3E%3Cscript%3E.png', '/share/calendar/jkse-2025.png.png',
    ])
    def test_png_invalid_id_is_404_without_rendering(self, client, path):
        with patch.object(share, 'calendar_share_png') as render:
            assert client.get(path).status_code == 404
        render.assert_not_called()

    def test_png_render_error_is_503(self, client):
        with patch.object(share, 'calendar_share_png', side_effect=RuntimeError('redis down')):
            res = client.get('/share/calendar/jkse-2025.png')
        assert res.status_code == 503 and res.headers['retry-after'] == '300'

    def test_png_not_swallowed_by_page_route(self, client):
        with patch.object(share, 'calendar_share_png', return_value=PNG_SIGNATURE):
            res = client.get('/share/calendar/jkse-2025-syariah.png', headers={'user-agent': HUMAN_UA})
        assert res.headers['content-type'] == 'image/png'

    def test_page_for_bots_has_tags_and_warms_the_image(self, client):
        with patch.object(share, 'calendar_share_png', return_value=PNG_SIGNATURE) as render:
            res = client.get('/share/calendar/jkse-2025', headers={'user-agent': BOT_UA})
        assert res.status_code == 200
        assert res.headers['content-type'].startswith('text/html')
        assert 'og:image' in res.text and 'jkse-2025.png' in res.text
        render.assert_called_once_with('jkse-2025')

    def test_page_for_bots_with_no_data_is_404(self, client):
        with patch.object(share, 'calendar_share_png', return_value=None):
            assert client.get('/share/calendar/jkse-2099', headers={'user-agent': BOT_UA}).status_code == 404

    def test_page_for_people_redirects_without_rendering(self, client):
        with patch.object(share, 'calendar_share_png') as render:
            res = client.get('/share/calendar/sp500-2024', headers={'user-agent': HUMAN_UA})
        assert res.status_code == 200
        assert 'location.replace("/calendar?' in res.text and 'list=sp500' in res.text and 'utm_source=share' in res.text
        render.assert_not_called()

    @pytest.mark.parametrize('path', [
        '/share/calendar/jkse-25', '/share/calendar/JKSE-2025',
        '/share/calendar/jkse-2025%22%3E%3Cscript%3Ealert(1)%3C%2Fscript%3E',
    ])
    def test_page_invalid_id_is_404(self, client, path):
        res = client.get(path, headers={'user-agent': BOT_UA})
        assert res.status_code == 404
        assert '<script>' not in res.text


# ═════════════════════════════════════════════════════════════════════════════

class TestShareBar:

    def test_data(self):
        data = ui.share_bar_data('/share/calendar/jkse-2025.png', '/share/calendar/jkse-2025',
                                 'dividend-calendar-2026-jkse.png', 'Dividend Calendar 2026 · JKSE')
        assert data == {
            'image_url': '/share/calendar/jkse-2025.png',
            'share_url': '/share/calendar/jkse-2025',
            'filename': 'dividend-calendar-2026-jkse.png',
            'text': 'Dividend Calendar 2026 · JKSE',
            'title': 'Dividend Calendar 2026 · JKSE',
        }

    def test_mounts_component_with_key(self):
        with patch.object(ui, '_share_bar_component') as component:
            ui.share_bar('/a.png', 'https://x.test/a', 'a.png', 'text', key='cal_share', title='Title')
        mount = component.return_value
        mount.assert_called_once()
        assert mount.call_args.kwargs['key'] == 'cal_share'
        assert mount.call_args.kwargs['data']['title'] == 'Title'

    def test_script_uses_ccv2_only(self):
        for banned in ('Streamlit.', 'setComponentValue', 'setFrameHeight', 'postMessage'):
            assert banned not in ui._SHARE_BAR_JS
        assert 'export default function' in ui._SHARE_BAR_JS
        # Safari: the ClipboardItem gets a promise, created inside the click handler
        assert 'new ClipboardItem({ "image/png": png })' in ui._SHARE_BAR_JS

    def test_link_resolved_on_the_page_origin(self):
        # Copy link and Share use the page's own host, so a link copied on localhost stays there
        js = ui._SHARE_BAR_JS
        assert 'new URL(data.share_url, window.location.origin).href' in js
        assert 'copyText(link)' in js and 'url: link' in js
        assert 'copyText(data.share_url)' not in js and 'url: data.share_url' not in js
