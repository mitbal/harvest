"""Server-rendered, crawlable pages that sit in front of the Streamlit app.

Streamlit renders everything client-side over a websocket, so search engines
see an empty shell. These helpers build plain HTML stock pages
(``/stock_picker/<ticker>``) and sitemaps from the same Redis data the
screener uses. Nothing here imports Streamlit, so it is safe to call from
Starlette route handlers.
"""
import io
import os
import json
import time
import logging
import threading
from pathlib import Path
from xml.sax.saxutils import escape

import redis
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

import harvest.data as hd


logger = logging.getLogger(__name__)

SITE_URL = os.getenv('SITE_URL', 'https://panendividen.com').rstrip('/')
SCORE_TABLE_KEY = 'div_score_jkse'  # same key the screener reads

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / 'templates'
_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape(['html']),
)


class _TTLCache:
    """Tiny thread-safe TTL cache; route handlers run in a thread pool."""

    def __init__(self, ttl_seconds):
        self.ttl = ttl_seconds
        self._data = {}
        self._lock = threading.Lock()

    def get_or_set(self, key, loader):
        now = time.monotonic()
        with self._lock:
            hit = self._data.get(key)
            if hit and now - hit[0] < self.ttl:
                return hit[1]
        value = loader()
        with self._lock:
            self._data[key] = (now, value)
        return value


_table_cache = _TTLCache(60 * 10)
_dividend_cache = _TTLCache(60 * 60 * 6)
_profile_cache = _TTLCache(60 * 60 * 24)


# ---------------------------------------------------------------------------
# Ticker helpers
# ---------------------------------------------------------------------------

def ticker_slug(symbol):
    """'BBCA.JK' -> 'bbca'."""
    return symbol.upper().removesuffix('.JK').lower()


def slug_to_symbol(slug):
    """'bbca' / 'BBCA.JK' -> 'BBCA.JK'."""
    return slug.upper().removesuffix('.JK') + '.JK'


def stock_page_url(symbol):
    return f'{SITE_URL}/stock_picker/{ticker_slug(symbol)}'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_score_table(raw):
    """Mirror the formats accepted by get_div_score_table in stock_picker.py."""
    updated = None
    if isinstance(raw, bytes) and raw.startswith(b'PAR1'):
        df = pd.read_parquet(io.BytesIO(raw))
    else:
        payload = json.loads(raw)
        if isinstance(payload, dict) and 'date' in payload:
            updated = payload['date']
            df = pd.DataFrame(json.loads(payload['content']))
        else:
            df = pd.DataFrame(payload)
    return df.set_index('symbol'), updated


def load_score_table():
    """Return (DataFrame indexed by symbol, last-updated string or None)."""
    def _load():
        r = redis.from_url(os.environ['REDIS_URL'], socket_connect_timeout=10, socket_timeout=30)
        raw = r.get(SCORE_TABLE_KEY)
        if raw is None:
            raise LookupError(f'Redis key {SCORE_TABLE_KEY!r} is empty')
        return _parse_score_table(raw)

    return _table_cache.get_or_set(SCORE_TABLE_KEY, _load)


def load_dividend_history(symbol):
    def _load():
        try:
            return hd.get_dividend_history_single_stock_dag(symbol)
        except Exception:
            logger.exception('dividend history failed for %s', symbol)
            return None

    return _dividend_cache.get_or_set(symbol, _load)


def load_company_profile(symbol):
    """Company name/description are nice-to-have; never fail the page on them."""
    def _load():
        try:
            cp = hd.get_company_profile([symbol])
            return cp.loc[symbol].to_dict() if symbol in cp.index else {}
        except Exception:
            logger.exception('company profile failed for %s', symbol)
            return {}

    return _profile_cache.get_or_set(symbol, _load)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _num(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(value) else value


def _text(value):
    return value.strip() if isinstance(value, str) else ''


def fmt_idr(value, decimals=0):
    value = _num(value)
    if value is None:
        return '-'
    text = f'{value:,.{decimals}f}'.replace(',', '_').replace('.', ',').replace('_', '.')
    return f'Rp{text}'


def fmt_dividend(value):
    """Per-share dividends: show cents only when there are any (Rp227,50 vs Rp250)."""
    value = _num(value)
    if value is None:
        return '-'
    value = round(value, 2)
    return fmt_idr(value, 2 if value % 1 else 0)


def is_dividend_payer(row):
    return (_num(row.get('numDividendYear')) or 0) > 0


def fmt_pct(value, decimals=2):
    value = _num(value)
    if value is None:
        return '-'
    return f'{value:.{decimals}f}'.replace('.', ',') + '%'


def fmt_market_cap(value):
    value = _num(value)
    if value is None:
        return '-'
    for threshold, unit in ((1e15, 'kuadriliun'), (1e12, 'triliun'), (1e9, 'miliar')):
        if value >= threshold:
            return f'Rp{value / threshold:.2f}'.replace('.', ',') + f' {unit}'
    return fmt_idr(value)


def fmt_date(value):
    try:
        return pd.to_datetime(value).strftime('%d-%m-%Y')
    except (TypeError, ValueError):
        return '-'


# ---------------------------------------------------------------------------
# Page building
# ---------------------------------------------------------------------------

def _dividend_rows(div_df, limit=12):
    if div_df is None or div_df.empty:
        return [], []

    df = div_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date', ascending=False)

    recent = [
        {
            'ex_date': fmt_date(row['date']),
            'payment_date': fmt_date(row.get('payment_date')),
            'amount': fmt_dividend(row['adjDividend']),
            'type': str(row.get('dividend_type') or '-').capitalize(),
        }
        for _, row in df.head(limit).iterrows()
    ]

    year_col = 'fiscal_year' if 'fiscal_year' in df.columns else None
    years = df[year_col] if year_col else df['date'].dt.year
    annual = (
        df.assign(year=years)
        .groupby('year')['adjDividend']
        .agg(['sum', 'count'])
        .sort_index(ascending=False)
        .head(10)
    )
    yearly = [
        {'year': int(year), 'total': fmt_dividend(row['sum']), 'count': int(row['count'])}
        for year, row in annual.iterrows()
    ]
    return recent, yearly


def _peers(table, symbol, sector, limit=8):
    if not sector or 'sector' not in table.columns:
        return []
    peers = table[(table['sector'] == sector) & (table.index != symbol)]
    peers = peers[peers.apply(is_dividend_payer, axis=1)]
    if 'mktCap' in peers.columns:
        peers = peers.sort_values('mktCap', ascending=False)
    return [
        {'ticker': ticker_slug(sym).upper(), 'url': f'/stock_picker/{ticker_slug(sym)}', 'yield': fmt_pct(row.get('yield'))}
        for sym, row in peers.head(limit).iterrows()
    ]


def build_stock_context(symbol):
    """Return the template context for a stock, or None if the ticker is unknown."""
    table, updated = load_score_table()
    if symbol not in table.index:
        return None

    row = table.loc[symbol]
    profile = load_company_profile(symbol)
    recent, yearly = _dividend_rows(load_dividend_history(symbol))

    ticker = ticker_slug(symbol).upper()
    company = _text(profile.get('companyName')) or ticker
    sector = _text(row.get('sector')) or _text(profile.get('sector'))
    industry = _text(row.get('industry')) or _text(profile.get('industry'))
    dividend_yield = fmt_pct(row.get('yield'))
    last_div = fmt_dividend(row.get('lastDiv'))
    num_div_years = int(_num(row.get('numDividendYear')) or 0)

    if num_div_years and not _num(row.get('lastDiv')):
        summary = (
            f'{company} ({ticker}) tidak membagikan dividen untuk tahun fiskal terakhir, '
            f'namun tercatat pernah membagikan dividen selama {num_div_years} tahun.'
        )
    elif num_div_years:
        summary = (
            f'{company} ({ticker}) membagikan dividen {last_div} per saham untuk tahun fiskal terakhir, '
            f'setara dividend yield {dividend_yield} pada harga {fmt_idr(row.get("price"))}. '
            f'{ticker} tercatat membagikan dividen selama {num_div_years} tahun.'
        )
    else:
        summary = f'{company} ({ticker}) belum memiliki riwayat pembagian dividen yang tercatat.'

    stats = [
        ('Harga saham', fmt_idr(row.get('price'))),
        ('Dividend yield', dividend_yield),
        ('Dividen per saham (tahun fiskal terakhir)', last_div),
        ('Jumlah tahun membagi dividen', str(num_div_years) if num_div_years else '-'),
        ('Rata-rata kenaikan dividen per tahun', fmt_pct(row.get('avgFlatAnnualDivIncrease'), 1)),
        ('Pertumbuhan pendapatan (median 5 tahun)', fmt_pct(row.get('revenueGrowth'), 1)),
        ('Pertumbuhan laba bersih (median 5 tahun)', fmt_pct(row.get('netIncomeGrowth'), 1)),
        ('Kapitalisasi pasar', fmt_market_cap(row.get('mktCap'))),
        ('Sektor', sector or '-'),
        ('Industri', industry or '-'),
        ('Saham syariah', 'Ya' if _num(row.get('is_syariah')) == 1 else 'Tidak'),
    ]

    canonical = stock_page_url(symbol)
    if _num(row.get('lastDiv')):
        title = f'Dividen {ticker}: Yield {dividend_yield}, Jadwal & Riwayat | Panen Dividen'
    else:
        title = f'Dividen {ticker}: Riwayat & Profil Saham | Panen Dividen'
    meta_description = (
        f'Riwayat dividen saham {ticker} ({company}): dividen {last_div} per saham, yield {dividend_yield}, '
        f'jadwal ex-date dan pembayaran, serta pertumbuhan pendapatan dan laba.'
    )
    json_ld = {
        '@context': 'https://schema.org',
        '@graph': [
            {
                '@type': 'WebPage',
                'name': title,
                'url': canonical,
                'description': meta_description,
                'inLanguage': 'id-ID',
                **({'dateModified': updated} if updated else {}),
                'about': {'@type': 'Corporation', 'name': company, 'tickerSymbol': f'IDX:{ticker}'},
            },
            {
                '@type': 'BreadcrumbList',
                'itemListElement': [
                    {'@type': 'ListItem', 'position': 1, 'name': 'Panen Dividen', 'item': SITE_URL},
                    {'@type': 'ListItem', 'position': 2, 'name': 'Ranking Screener', 'item': f'{SITE_URL}/stock_picker'},
                    {'@type': 'ListItem', 'position': 3, 'name': ticker, 'item': canonical},
                ],
            },
        ],
    }

    return {
        'ticker': ticker,
        'company': company,
        'title': title,
        'meta_description': meta_description,
        'canonical': canonical,
        'summary': summary,
        'stats': stats,
        'recent_dividends': recent,
        'yearly_dividends': yearly,
        'description': _text(profile.get('description')),
        'sector': sector,
        'peers': _peers(table, symbol, sector),
        'app_url': f'/stock_picker?stock={symbol}',
        'updated': updated,
        'json_ld': json.dumps(json_ld, ensure_ascii=False).replace('</', '<\\/'),
    }


def render_stock_page(symbol):
    context = build_stock_context(symbol)
    if context is None:
        return None
    return _env.get_template('stock_page.html').render(**context)


def render_not_found(slug):
    return _env.get_template('stock_not_found.html').render(ticker=slug.upper())


def render_stock_sitemap():
    table, updated = load_score_table()
    lastmod = ''
    if updated:
        try:
            lastmod = f'<lastmod>{pd.to_datetime(updated).strftime("%Y-%m-%d")}</lastmod>'
        except (TypeError, ValueError):
            pass

    # Non-payers still get a page, but aren't worth advertising to crawlers.
    table = table[table.apply(is_dividend_payer, axis=1)]
    if 'mktCap' in table.columns:
        table = table.sort_values('mktCap', ascending=False)
    urls = '\n'.join(
        f'<url><loc>{escape(stock_page_url(symbol))}</loc>{lastmod}</url>'
        for symbol in table.index
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f'{urls}\n'
        '</urlset>\n'
    )
