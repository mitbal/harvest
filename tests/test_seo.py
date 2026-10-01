"""
tests/test_seo.py
Server-rendered stock pages (harvest/seo.py) and their routes in server.py.

Redis and the dividend/profile downloads are patched, so this runs offline.
"""
from unittest.mock import patch

import pandas as pd
import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from harvest import seo
import server


TABLE = pd.DataFrame(
    {
        'price': [6325, 2660, 150, 500],
        'lastDiv': [336.0, 222.08, 0.0, 0.0],
        'yield': [5.31, 8.35, 0.0, 0.0],
        'sector': ['Financial Services', 'Communication Services', 'Financial Services', 'Financial Services'],
        'industry': ['Banks - Regional', 'Telecom Services', 'Banks - Regional', 'Banks - Regional'],
        'mktCap': [7.7e14, 2.6e14, 1e12, 2e12],
        'numDividendYear': [22.0, 22.0, 2.0, 0.0],
        'avgFlatAnnualDivIncrease': [13.2, 5.0, 0.0, 0.0],
        'revenueGrowth': [7.5, 3.0, 1.0, 1.0],
        'netIncomeGrowth': [11.9, 2.0, 1.0, 1.0],
        'is_syariah': [False, True, float('nan'), False],
    },
    index=pd.Index(['BBCA.JK', 'TLKM.JK', 'PURI.JK', 'DPUM.JK'], name='symbol'),
)

DIVIDENDS = pd.DataFrame({
    'date': ['2026-03-30', '2025-12-03', '2025-03-21', '2024-11-21'],
    'adjDividend': [281.0, 55.0, 250.0, 50.0],
    'payment_date': ['2026-04-08', '2025-12-22', '2025-04-11', '2024-12-11'],
    'fiscal_year': [2025, 2025, 2024, 2024],
    'dividend_type': ['final', 'interim', 'final', 'interim'],
})


@pytest.fixture(autouse=True)
def fake_data():
    with patch.object(seo, 'load_score_table', return_value=(TABLE, None)), \
         patch.object(seo, 'load_dividend_history', return_value=DIVIDENDS), \
         patch.object(seo, 'load_company_profile', return_value={'companyName': 'PT Bank Central Asia Tbk', 'description': 'A bank.'}):
        yield


@pytest.fixture
def client():
    async def streamlit_stub(request):
        return PlainTextResponse('streamlit shell')

    app = Starlette(
        routes=[
            Route('/robots.txt', server.robots_txt),
            Route('/sitemap_stocks.xml', server.sitemap_stocks_xml),
            Route('/stock_picker/{ticker}', server.stock_page),
            Route('/stock_picker', streamlit_stub),
        ],
        middleware=[Middleware(server.StockCanonicalMiddleware)],
    )
    return TestClient(app)


class TestHelpers:

    @pytest.mark.parametrize('raw, expected', [('bbca', 'BBCA.JK'), ('BBCA.JK', 'BBCA.JK'), ('bbca.jk', 'BBCA.JK')])
    def test_slug_to_symbol(self, raw, expected):
        assert seo.slug_to_symbol(raw) == expected

    def test_ticker_slug(self):
        assert seo.ticker_slug('BBCA.JK') == 'bbca'

    def test_formatting_uses_indonesian_separators(self):
        assert seo.fmt_idr(6325) == 'Rp6.325'
        assert seo.fmt_dividend(227.5) == 'Rp227,50'
        assert seo.fmt_dividend(110.99999999) == 'Rp111'
        assert seo.fmt_pct(5.3122) == '5,31%'
        assert seo.fmt_market_cap(7.7e14) == 'Rp770,00 triliun'
        assert seo.fmt_idr(float('nan')) == '-'


class TestStockPage:

    def test_page_has_unique_seo_metadata_and_content(self):
        html = seo.render_stock_page('BBCA.JK')
        assert '<title>Dividen BBCA: Yield 5,31%, Jadwal &amp; Riwayat | Panen Dividen</title>' in html
        assert '<link rel="canonical" href="https://panendividen.com/stock_picker/bbca">' in html
        assert 'Rp336 per saham' in html
        assert '<td>2025</td><td class="num">Rp336</td>' in html
        assert 'href="/stock_picker?stock=BBCA.JK"' in html
        assert '"@type": "BreadcrumbList"' in html

    def test_peers_link_to_other_dividend_payers_in_sector(self):
        context = seo.build_stock_context('BBCA.JK')
        assert [p['ticker'] for p in context['peers']] == ['PURI']

    def test_lapsed_payer_summary(self):
        context = seo.build_stock_context('PURI.JK')
        assert 'tidak membagikan dividen untuk tahun fiskal terakhir' in context['summary']
        assert 'Yield' not in context['title']
        assert ('Saham syariah', 'Tidak') in context['stats']

    def test_unknown_ticker_returns_none(self):
        assert seo.render_stock_page('ZZZZ.JK') is None

    def test_sitemap_lists_only_dividend_payers(self):
        xml = seo.render_stock_sitemap()
        assert '<loc>https://panendividen.com/stock_picker/bbca</loc>' in xml
        assert 'puri' in xml
        assert 'dpum' not in xml


class TestRoutes:

    def test_stock_page_ok(self, client):
        response = client.get('/stock_picker/bbca')
        assert response.status_code == 200
        assert 'Dividen Saham BBCA' in response.text

    @pytest.mark.parametrize('path', ['/stock_picker/BBCA', '/stock_picker/bbca.jk'])
    def test_non_canonical_ticker_redirects(self, client, path):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 301
        assert response.headers['location'] == '/stock_picker/bbca'

    def test_redirect_keeps_utm_parameters(self, client):
        response = client.get('/stock_picker/BBCA?utm_source=instagram&utm_campaign=sept', follow_redirects=False)
        assert response.headers['location'] == '/stock_picker/bbca?utm_source=instagram&utm_campaign=sept'

    @pytest.mark.parametrize('path', ['/stock_picker/zzzz', '/stock_picker/%3Cb%3E'])
    def test_unknown_ticker_is_404_noindex(self, client, path):
        response = client.get(path)
        assert response.status_code == 404
        assert 'noindex' in response.text
        assert '<b>' not in response.text

    def test_data_failure_returns_503(self, client):
        with patch.object(seo, 'render_stock_page', side_effect=ConnectionError):
            response = client.get('/stock_picker/bbca')
        assert response.status_code == 503

    def test_robots_lists_stock_sitemap(self, client):
        assert 'sitemap_stocks.xml' in client.get('/robots.txt').text

    def test_interactive_url_gets_canonical_link_header(self, client):
        response = client.get('/stock_picker?stock=BBCA.JK')
        assert response.text == 'streamlit shell'
        assert response.headers['link'] == '<https://panendividen.com/stock_picker/bbca>; rel="canonical"'

    @pytest.mark.parametrize('query', ['', '?stock=AAPL&market=US', '?stock=D05.SI'])
    def test_no_canonical_header_without_idx_stock(self, client, query):
        assert 'link' not in client.get(f'/stock_picker{query}').headers
