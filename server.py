"""ASGI entry point: Streamlit app plus crawlable SEO routes on the same port.

Run with ``streamlit run server.py`` (Streamlit detects the ``st.App``).
"""
import re
import logging
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit

import streamlit as st
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from starlette.routing import Route

from harvest import seo, webhooks
from harvest.utils import setup_logging


logger = logging.getLogger('seo')
visit_logger = setup_logging('tracking')
STATIC_DIR = Path(__file__).resolve().parent / 'static'

# Pages the edge/browser may cache briefly; data only changes daily.
_CACHE_HEADERS = {'Cache-Control': 'public, max-age=600'}

# Crawlers, link-preview fetchers, AI agents and scripted clients; these pages draw far more of them than people.
_BOT_UA = re.compile(
    r'bot|crawl|spider|slurp|scrap|fetch|preview|headless|lighthouse|inspectiontool|externalhit|externalagent'
    r'|whatsapp|chatgpt|anthropic|python|curl|wget|httpx|go-http|okhttp|java/|axios|node|libwww|uptime|monitor',
    re.IGNORECASE,
)


def is_probable_bot(user_agent):
    """Whether a request looks automated; an empty user agent counts, since browsers always send one."""
    return not user_agent or bool(_BOT_UA.search(user_agent))


async def robots_txt(request):
    return FileResponse(STATIC_DIR / 'robots.txt', media_type='text/plain', headers=_CACHE_HEADERS)


async def sitemap_xml(request):
    return FileResponse(STATIC_DIR / 'sitemap.xml', media_type='application/xml', headers=_CACHE_HEADERS)


async def sitemap_stocks_xml(request):
    try:
        body = await run_in_threadpool(seo.render_stock_sitemap)
    except Exception:
        logger.exception('failed to build stock sitemap')
        return Response(status_code=503, headers={'Retry-After': '300'})
    return Response(body, media_type='application/xml', headers=_CACHE_HEADERS)


async def stock_page(request):
    slug = request.path_params['ticker']
    if not slug.replace('.', '').isalnum() or not slug.isascii():
        return HTMLResponse(seo.render_not_found(slug), status_code=404)
    symbol = seo.slug_to_symbol(slug)
    canonical_slug = seo.ticker_slug(symbol)
    if slug != canonical_slug:
        # Keep the query string so UTM parameters survive the redirect.
        query = f'?{request.url.query}' if request.url.query else ''
        return RedirectResponse(f'/stock_picker/{canonical_slug}{query}', status_code=301)

    try:
        html = await run_in_threadpool(seo.render_stock_page, symbol)
    except Exception:
        logger.exception('failed to render stock page for %s', symbol)
        # 503 tells crawlers to retry later instead of dropping the URL.
        return HTMLResponse('Layanan sedang sibuk, coba lagi nanti.', status_code=503, headers={'Retry-After': '300'})

    if html is None:
        return HTMLResponse(seo.render_not_found(slug), status_code=404)

    # The VISIT log comes from the page's beacon (visit_beacon), so HTML-only scrapers never reach it.
    return HTMLResponse(html, headers=_CACHE_HEADERS)


async def visit_beacon(request):
    """Log a stock page view, posted by the page's script once it runs in a browser.

    The body is the page's own path and query, e.g. ``/stock_picker/bbca?utm_source=ig``.
    Anyone can post here, so the input is checked and re-encoded before it reaches the log.
    """
    user_agent = request.headers.get('user-agent', '')
    body = (await request.body())[:1024].decode('utf-8', 'replace')
    url = urlsplit(body)
    prefix = '/stock_picker/'
    slug = url.path[len(prefix):] if url.path.startswith(prefix) else ''
    if is_probable_bot(user_agent) or not (0 < len(slug) <= 12 and slug.isascii() and slug.isalnum()):
        return Response(status_code=204)

    # Same shape as main.py's VISIT log; no session here, so log the user agent to tell crawlers apart.
    log_msg = f'VISIT | page=stock_page/{seo.slug_to_symbol(slug)}'
    query = urlencode(parse_qsl(url.query))
    if query:
        log_msg += f' | {query.replace("&", " | ")}'
    log_msg += f' | ua={user_agent}'
    visit_logger.info(log_msg)
    return Response(status_code=204)


class StockCanonicalMiddleware:
    """Point ``/stock_picker?stock=BBCA.JK`` at ``/stock_picker/bbca`` via a Link header.

    The interactive Streamlit URL serves the same generic HTML shell for every
    stock, so without this Google may index it as a thin duplicate.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'].rstrip('/') != '/stock_picker':
            return await self.app(scope, receive, send)

        params = parse_qs(scope.get('query_string', b'').decode('latin-1'))
        stock = (params.get('stock') or [''])[0].strip()
        market = (params.get('market') or ['JKSE'])[0].upper()
        is_idx = stock and market in ('JKSE', 'IDX') and ('.' not in stock or stock.upper().endswith('.JK'))
        if not is_idx:
            return await self.app(scope, receive, send)

        link = f'<{seo.stock_page_url(seo.slug_to_symbol(stock))}>; rel="canonical"'.encode()

        async def send_with_link(message):
            if message['type'] == 'http.response.start':
                message['headers'] = [*message.get('headers', []), (b'link', link)]
            await send(message)

        return await self.app(scope, receive, send_with_link)


app = st.App(
    'main.py',
    routes=[
        Route('/robots.txt', robots_txt),
        Route('/sitemap.xml', sitemap_xml),
        Route('/sitemap_stocks.xml', sitemap_stocks_xml),
        # Old static sitemap URL, possibly already submitted to Search Console.
        Route('/sitemap_full.xml', sitemap_stocks_xml),
        Route('/stock_picker/{ticker}', stock_page),
        Route('/visit/stock_page', visit_beacon, methods=['POST']),
        Route('/webhooks/bmc', webhooks.bmc_webhook, methods=['POST']),
        Route('/webhooks/trakteer', webhooks.trakteer_webhook, methods=['POST']),
        Route('/webhooks/telegram', webhooks.telegram_webhook, methods=['POST']),
    ],
    middleware=[Middleware(StockCanonicalMiddleware)],
)
