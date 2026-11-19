"""Share image with the real calendar chart: tests/ stubs harvest.plot and lesley, so this runs here,
in the process tests/test_apptests.py starts with real modules."""
import io
import json
from unittest.mock import patch

import pytest
from PIL import Image

from harvest import share


ROWS = [
    {'date': f'2025-{month:02d}-{day:02d}', 'symbol': f'S{month}{day}.JK', 'adjDividend': 50.0 * month, 'price': 5000.0}
    for month in range(1, 13) for day in (3, 17)
]
SCORES = [{'symbol': row['symbol'], 'is_syariah': i % 2 == 0} for i, row in enumerate(ROWS)]


@pytest.fixture(autouse=True)
def fake_redis():
    data = {'div_cal_jkse_2025': json.dumps(ROWS), 'div_score_jkse': json.dumps(SCORES)}
    share._png_cache.clear()
    with patch.object(share, '_redis_get', side_effect=data.get):
        yield
    share._png_cache.clear()


@pytest.mark.parametrize('share_id', ['jkse-2025', 'jkse-2025-syariah'])
def test_full_calendar_renders_to_png(share_id):
    png = share.calendar_share_png(share_id)
    assert png.startswith(b'\x89PNG\r\n\x1a\n')
    image = Image.open(io.BytesIO(png))
    assert image.size == (share.IMAGE_WIDTH, share.IMAGE_HEIGHT)
    assert image.getpixel((5, 5)) == (0xEF, 0xF1, 0xEF)


def test_chart_carries_theme_attribution_and_background():
    s = share.parse_share_id('jkse-2025')
    spec = share.calendar_chart(share.load_calendar_frame(s), s).to_dict()
    assert spec['background'] == share._BACKGROUND
    assert spec['title']['text'] == f'{share.site_host()} · Dividend Calendar'
    # The app's Altair theme (harvest/plot.py) is merged into the config, as it is on the page
    assert spec['config']['font'].startswith('Inter')


def test_unknown_year_is_none():
    assert share.calendar_share_png('jkse-2019') is None
