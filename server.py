"""ASGI entry point: Streamlit app plus crawlable SEO routes on the same port.

Run with ``streamlit run server.py`` (Streamlit detects the ``st.App``).
"""
import re
import logging
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit

import streamlit as st
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from starlette.routing import Route

from harvest import memwatch, seo, share, webhooks
from harvest.utils import setup_logging


logger = logging.getLogger('seo')
visit_logger = setup_logging('tracking')
memwatch.start()  # Timed MEMORY lines in the logs to track down the slow RSS growth in production.
STATIC_DIR = Path(__file__).resolve().parent / 'static'

# Pages the edge/browser may cache briefly; data only changes daily.
_CACHE_HEADERS = {'Cache-Control': 'public, max-age=600'}

# Crawlers, link-preview fetchers, AI agents and scripted clients; these pages draw far more of them than people.
_BOT_UA = re.compile(
    r'bot|crawl|spider|slurp|scrap|fetch|preview|headless|lighthouse|inspectiontool|externalhit|externalagent'
    r'|whatsapp|chatgpt|anthropic|python|curl|wget|httpx|go-http|okhttp|java/|axios|node|libwww|uptime|monitor',
    re.IGNORECASE,
)


# A plain Chrome UA, with nothing between "Gecko)" and "Chrome/" or after "Safari/", so Samsung Internet, Edge, Opera
# and in-app WebViews, whose Chromium can lag by design, don't match.
_PLAIN_CHROME_UA = re.compile(r'\(KHTML, like Gecko\) Chrome/(\d+)\.[\d.]+ (?:Mobile )?Safari/537\.36$')

# Chrome auto-updates every four weeks, so one this many versions behind is almost always a scraper pinned to an old build.
_STALE_CHROME_VERSIONS = 8


def current_chrome_major(today=None):
    """Estimate of the stable Chrome major version: 140 shipped on 2025-09-02, then one release every four weeks."""
    return 140 + ((today or date.today()) - date(2025, 9, 2)).days // 28


def is_probable_bot(user_agent, today=None):
    """Whether a request looks automated; an empty user agent counts, since browsers always send one."""
    if not user_agent or _BOT_UA.search(user_agent):
        return True
    chrome = _PLAIN_CHROME_UA.search(user_agent)
    return bool(chrome) and int(chrome.group(1)) < current_chrome_major(today) - _STALE_CHROME_VERSIONS


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


async def calendar_share_png(request):
    """The Full Year calendar image for a share id, e.g. /share/calendar/jkse-2025.png."""
    share_id = request.path_params['share_id']
    if share.parse_share_id(share_id) is None:
        return Response('Not found', status_code=404, media_type='text/plain')
    try:
        png = await run_in_threadpool(share.calendar_share_png, share_id)
    except Exception:
        logger.exception('failed to render calendar share image %s', share_id)
        return Response(status_code=503, headers={'Retry-After': '300'})
    if png is None:
        return Response('Not found', status_code=404, media_type='text/plain')
    return Response(png, media_type='image/png', headers=_CACHE_HEADERS)


async def calendar_share_page(request):
    """Link-preview page for a shared calendar: Open Graph tags for unfurlers, a redirect for people."""
    calendar_share = share.parse_share_id(request.path_params['share_id'])
    if calendar_share is None:
        return HTMLResponse('Not found', status_code=404)

    # Unfurlers fetch the image next, so render it now: a year with no data gets a 404 instead of a
    # preview with a broken image, and the image is cached by the time they ask. People are sent on
    # to the live calendar right away, so they don't wait for the render.
    if is_probable_bot(request.headers.get('user-agent', '')):
        try:
            png = await run_in_threadpool(share.calendar_share_png, calendar_share.share_id)
        except Exception:
            logger.exception('failed to render calendar share image %s', calendar_share.share_id)
            return HTMLResponse('Layanan sedang sibuk, coba lagi nanti.', status_code=503, headers={'Retry-After': '300'})
        if png is None:
            return HTMLResponse('Not found', status_code=404)

    return HTMLResponse(share.render_share_page(calendar_share), headers=_CACHE_HEADERS)


# The .png route comes first: {share_id} alone would also match 'jkse-2025.png'
SHARE_ROUTES = [
    Route('/share/calendar/{share_id}.png', calendar_share_png),
    Route('/share/calendar/{share_id}', calendar_share_page),
]


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
        *SHARE_ROUTES,
        Route('/webhooks/bmc', webhooks.bmc_webhook, methods=['POST']),
        Route('/webhooks/trakteer', webhooks.trakteer_webhook, methods=['POST']),
        Route('/webhooks/telegram', webhooks.telegram_webhook, methods=['POST']),
    ],
    middleware=[Middleware(StockCanonicalMiddleware)],
)
