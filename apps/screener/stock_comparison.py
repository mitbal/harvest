import io
import os
import time
import logging
import mimetypes

# Streamlit's Tornado static server uses mimetypes.guess_type() at request time.
# Register SVG so logos are served as image/svg+xml (not text/plain).
mimetypes.add_type('image/svg+xml', '.svg')

import redis
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from datetime import datetime

import harvest.plot as hp
import harvest.data as hd
from harvest.utils import setup_logging


st.title('Stock Comparison ⚖️')
st.set_page_config(page_title='Stock Comparison - Panen Dividen')

# ── Constants ────────────────────────────────────────────────────────────────
DIV_MATURITY_HALFLIFE = 25
PROJECTION_HORIZON_YRS = 5

# Colour palette for up to 10 selected stocks
_STOCK_PALETTE = [
    '#e41a1c', '#377eb8', '#ff7f00', '#984ea3',
    '#4daf4a', '#a65628', '#f781bf', '#999999',
    '#17becf', '#bcbd22',
]


# ── Logging ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_logger(name, level=logging.INFO):
    return setup_logging(name, level)


logger = get_logger('comparison')


# ── Redis connection ─────────────────────────────────────────────────────────
import os

@st.cache_resource
def connect_redis(redis_url):
    return redis.from_url(
        redis_url,
        socket_connect_timeout=10,
        socket_timeout=30,
        socket_keepalive=True,
        retry_on_timeout=True,
    )


# ── Data loading (mirrors stock_picker.py) ───────────────────────────────────
@st.cache_data(ttl=60 * 10, show_spinner='Downloading stock universe…')
def get_div_score_table(key='jkse_div_score'):
    redis_url = os.environ['REDIS_URL']
    r = connect_redis(redis_url)
    rjson = r.get(key)

    if rjson is not None:
        if isinstance(rjson, bytes) and rjson.startswith(b'PAR1'):
            final_df = pd.read_parquet(io.BytesIO(rjson))
        else:
            import json
            div_score_json = json.loads(rjson)
            if 'date' in div_score_json:
                final_df = pd.DataFrame(json.loads(div_score_json['content']))
            else:
                final_df = pd.DataFrame(div_score_json)
    else:
        final_df = pd.read_csv('dividend_historical.csv')

    final_df.rename(columns={'symbol': 'stock'}, inplace=True)
    cp_df = hd.get_company_profile(final_df['stock'].to_list())
    final_df.drop(columns=['price'], inplace=True)
    final_df = final_df.merge(cp_df[['price', 'changes', 'beta']], left_on='stock', right_on='symbol')
    return final_df.set_index('stock')


_KEEP_COLS = [
    'price', 'changes', 'sector', 'industry', 'mktCap', 'ipoDate',
    'yield', 'lastDiv', 'avgFlatAnnualDivIncrease', 'numDividendYear',
    'positiveYear', 'numOfYear', 'maximumCutPct', 'max10CutPct',
    'peRatio', 'psRatio', 'revenueGrowth', 'netIncomeGrowth',
    'medianProfitMargin', 'earningTTM', 'revenueTTM',
    'revenueGrowthTTM', 'netIncomeGrowthTTM', 'beta',
    'return_7d', 'return_1m', 'return_1y', 'return_10y',
    'total_return_1y', 'total_return_10y',
    'is_syariah',
]


@st.cache_data(show_spinner=False)
def get_processed_df(df):
    df = df.copy()
    df['marginTTM'] = df['earningTTM'] / df['revenueTTM'] * 100
    df['mc_penalty'] = df['mktCap'].apply(lambda x: 1 / (1 + np.exp(-2 * (x / 3_000_000_000_000 - 1))))
    df['maximumCutPct'] = df['maximumCutPct'].apply(lambda x: min(x, 0) * -1)
    df['max10CutPct'] = df['max10CutPct'].apply(lambda x: min(x, 0) * -1)
    df['maxDivIncrease'] = df.apply(lambda x: min(x['avgFlatAnnualDivIncrease'], x['lastDiv'] * 0.05), axis=1)
    df['maxRevGrowthDecrease'] = df.apply(lambda x: min(x['revenueGrowthTTM'], 0), axis=1)
    df['maxIncGrowthDecrease'] = df.apply(lambda x: min(x['netIncomeGrowthTTM'], 0), axis=1)

    for col in ['return_7d', 'return_1m', 'return_1y', 'return_10y', 'total_return_1y', 'total_return_10y']:
        if col in df.columns:
            df[col] = df[col] * 100

    df['DScore'] = (
        (df['lastDiv'] + df['maxDivIncrease'] * PROJECTION_HORIZON_YRS * (df['positiveYear'] / df['numOfYear'])) / df['price']
    ) * 100 \
      * (df['numDividendYear'] / (df['numDividendYear'] + DIV_MATURITY_HALFLIFE)) \
      * (1 - np.exp(-df['numDividendYear'] / 5)) \
      * (100 - df['max10CutPct']) / 100 \
      * df['mc_penalty'] \
      * (1 + df['maxRevGrowthDecrease'] / 100) \
      * (1 + df['maxIncGrowthDecrease'] / 100)

    df = df.fillna(0).sort_values('DScore', ascending=False)
    df.insert(0, 'Rank', range(1, len(df) + 1))
    return df


# ── Metric definitions ────────────────────────────────────────────────────────
METRIC_OPTIONS = {
    # ── Dividend ──
    'Dividend Yield (%)':        ('yield',                  'higher_better', '{:.2f}%'),
    'Last Dividend':             ('lastDiv',                'higher_better', '{:,.2f}'),
    'Div Growth (Annual)':       ('avgFlatAnnualDivIncrease','higher_better','{:,.2f}'),
    'Years Paying Dividend':     ('numDividendYear',        'higher_better', '{:.0f}'),
    'Years Raised Dividend':     ('positiveYear',           'higher_better', '{:.0f}'),
    'Dividend Score':            ('DScore',                 'higher_better', '{:.2f}'),
    # ── Valuation ──
    'PE Ratio':                  ('peRatio',                'lower_better',  '{:.1f}x'),
    'PS Ratio':                  ('psRatio',                'lower_better',  '{:.2f}x'),
    # ── Growth ──
    'Revenue Growth (5Y)':       ('revenueGrowth',          'higher_better', '{:.1f}%'),
    'Net Income Growth (5Y)':    ('netIncomeGrowth',        'higher_better', '{:.1f}%'),
    'Revenue Growth (TTM)':      ('revenueGrowthTTM',       'higher_better', '{:.1f}%'),
    'Net Income Growth (TTM)':   ('netIncomeGrowthTTM',     'higher_better', '{:.1f}%'),
    # ── Profitability ──
    'Profit Margin (Median)':    ('medianProfitMargin',     'higher_better', '{:.1f}%'),
    'Profit Margin (TTM)':       ('marginTTM',              'higher_better', '{:.1f}%'),
    # ── Price Return ──
    '1M Return':                 ('return_1m',              'higher_better', '{:+.1f}%'),
    '1Y Return':                 ('return_1y',              'higher_better', '{:+.1f}%'),
    'Total 1Y Return':           ('total_return_1y',        'higher_better', '{:+.1f}%'),
    '10Y Return':                ('return_10y',             'higher_better', '{:+.1f}%'),
    # ── General ──
    'Price':                     ('price',                  'neutral',       '{:,.2f}'),
    'Market Cap':                ('mktCap',                 'neutral',       '{:.2e}'),
    'Beta':                      ('beta',                   'neutral',       '{:.2f}'),
}

# Subset shown by default in the Table tab
TABLE_DEFAULT_METRICS = [
    'Dividend Yield (%)', 'Last Dividend', 'Years Paying Dividend',
    'Years Raised Dividend', 'Dividend Score',
    'PE Ratio', 'PS Ratio',
    'Revenue Growth (5Y)', 'Net Income Growth (5Y)',
    'Profit Margin (Median)', 'Profit Margin (TTM)',
    '1Y Return', 'Total 1Y Return',
]

# Metrics available in the distribution / scatter selectors
CHART_METRIC_OPTIONS = {k: v[0] for k, v in METRIC_OPTIONS.items() if v[1] != 'neutral' or k in ('PE Ratio', 'PS Ratio', 'Beta', 'Market Cap')}


# ══════════════════════════════════════════════════════════════════════════════
# ── URL query-param helpers ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
_qp = st.query_params

# Supported params:
#   market       → "jkse" | "sp500"          (default: jkse)
#   stocks       → comma-separated codes      (e.g. "BBCA,BMRI")
#   tab          → "table" | "dist" | "scatter" (default: table)
#   dist_metric  → label key in METRIC_OPTIONS
#   dist_sector  → sector name or "All"
#   sc_x         → label key in METRIC_OPTIONS
#   sc_y         → label key in METRIC_OPTIONS
#   sc_size      → label key in METRIC_OPTIONS

_MARKET_QP_MAP = {'jkse': 'Indonesian Stock', 'sp500': 'S&P 500 (US and World Stock)'}
_MARKET_QP_REV = {v: k for k, v in _MARKET_QP_MAP.items()}
_TAB_QP = ['table', 'dist', 'scatter']


def _qp_get(key, default=None):
    return _qp.get(key, default)


# ── Resolve initial market from URL ──────────────────────────────────────────
_market_qp = _qp_get('market', 'jkse').lower()
if _market_qp not in _MARKET_QP_MAP:
    _market_qp = 'jkse'
_market_label_default = _MARKET_QP_MAP[_market_qp]
_market_index_default = list(_MARKET_QP_MAP.values()).index(_market_label_default)


# ══════════════════════════════════════════════════════════════════════════════
# ── Market selection sidebar (same as stock_picker.py) ───────────────────────
# ══════════════════════════════════════════════════════════════════════════════
stock_select = st.sidebar.radio(
    'Stock List Selection',
    ['Indonesian Stock', 'S&P 500 (US and World Stock)'],
    horizontal=False,
    key='comp_sl',
    index=_market_index_default,
)

sl = 'JKSE' if stock_select == 'Indonesian Stock' else 'S&P500'

# Sync market → URL
_market_qp_val = _MARKET_QP_REV.get(stock_select, 'jkse')
if _qp_get('market') != _market_qp_val:
    _qp['market'] = _market_qp_val

# Log market selection changes
_prev_market = st.session_state.get('_comp_logged_market')
if _prev_market != sl:
    logger.info('COMPARISON | event=market_select | market=%s', sl)
    st.session_state['_comp_logged_market'] = sl

if sl == 'JKSE':
    key = 'div_score_jkse'
    currency = 'IDR'
    divisor = 1_000_000_000_000
    mcap_suffix = 'T IDR'
else:
    key = 'div_score_sp500'
    currency = 'USD'
    divisor = 1_000_000_000
    mcap_suffix = 'B USD'

final_df = get_div_score_table(key)
if sl != 'JKSE':
    try:
        final_df = final_df.drop('GOOGL')
    except KeyError:
        pass

if sl == 'JKSE' and 'is_syariah' in final_df.columns:
    is_syariah = st.sidebar.toggle('Syariah Only?', key='comp_syariah')
    if is_syariah:
        final_df = final_df[final_df['is_syariah'] == True]

_keep = [c for c in _KEEP_COLS if c in final_df.columns]
final_df = final_df[_keep]

filtered_df = get_processed_df(final_df)

stock_options = sorted(filtered_df.index.tolist())

# ── Resolve initial stocks from URL ──────────────────────────────────────────
_stocks_qp_raw = _qp_get('stocks', '')
_stocks_default = [
    s.strip().upper()
    for s in _stocks_qp_raw.split(',')
    if s.strip().upper() in stock_options
] if _stocks_qp_raw else []


# ── Stock multiselect ────────────────────────────────────────────────────────
st.markdown('### 🔍 Select Stocks to Compare')
selected_stocks = st.multiselect(
    label='Choose 2 or more stocks from the universe',
    options=stock_options,
    default=_stocks_default,
    placeholder='Type or pick stock codes…',
    key='comp_stocks',
)

# Sync stocks → URL
_stocks_qp_val = ','.join(selected_stocks)
if _qp_get('stocks', '') != _stocks_qp_val:
    if _stocks_qp_val:
        _qp['stocks'] = _stocks_qp_val
    elif 'stocks' in _qp:
        del _qp['stocks']

if len(selected_stocks) < 2:
    st.info('👆 Select at least **2 stocks** above to start comparing.')
    st.stop()

# Log stock selection when it changes
_prev_selection = st.session_state.get('_comp_logged_stocks')
_cur_selection = tuple(sorted(selected_stocks))
if _prev_selection != _cur_selection:
    logger.info(
        'COMPARISON | event=stock_select | market=%s | stocks=%s | count=%d',
        sl, ','.join(selected_stocks), len(selected_stocks),
    )
    st.session_state['_comp_logged_stocks'] = _cur_selection

# Build colour mapping for selected stocks
stock_colors = {s: _STOCK_PALETTE[i % len(_STOCK_PALETTE)] for i, s in enumerate(selected_stocks)}
comp_df = filtered_df.loc[selected_stocks].copy()


# ══════════════════════════════════════════════════════════════════════════════
# ── Tabs ─────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Resolve active tab from URL
_tab_qp = _qp_get('tab', 'table').lower()
_tab_default_idx = _TAB_QP.index(_tab_qp) if _tab_qp in _TAB_QP else 0

tab_table, tab_dist, tab_scatter = st.tabs(['📋 Table', '📊 Distribution', '🔵 Scatter Plot'])

# Log which tab is being viewed (tracks tab switches across reruns)
_active_tab_key = 'comp_active_tab'
_tab_map = {0: 'Table', 1: 'Distribution', 2: 'Scatter'}
# Each tab block syncs `tab` URL param when it renders, enabling shareability.


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Table
# ─────────────────────────────────────────────────────────────────────────────
with tab_table:
    _prev_tab = st.session_state.get('_comp_logged_tab')
    if _prev_tab != 'Table':
        logger.info(
            'COMPARISON | event=tab_view | tab=Table | stocks=%s',
            ','.join(selected_stocks),
        )
        st.session_state['_comp_logged_tab'] = 'Table'
    # Sync tab → URL (only update when tab actually changes to avoid extra reruns)
    if _qp_get('tab') != 'table':
        _qp['tab'] = 'table'
    st.markdown('#### Select metrics to compare')
    selected_metrics = st.multiselect(
        'Metrics',
        options=list(METRIC_OPTIONS.keys()),
        default=TABLE_DEFAULT_METRICS,
        key='comp_table_metrics',
        label_visibility='collapsed',
    )

    if not selected_metrics:
        st.info('Pick at least one metric above.')
    else:
        # Build a rows = metrics, cols = stocks DataFrame
        rows = []
        for label in selected_metrics:
            col_name, direction, fmt = METRIC_OPTIONS[label]
            if col_name not in comp_df.columns:
                continue
            row = {'Metric': label}
            vals = {}
            for stock in selected_stocks:
                v = comp_df.loc[stock, col_name]
                vals[stock] = v
                try:
                    row[stock] = fmt.format(v)
                except (ValueError, TypeError):
                    row[stock] = str(v)

            # Determine best / worst for highlighting
            numeric_vals = {k: v for k, v in vals.items() if isinstance(v, (int, float)) and not np.isnan(v)}
            if numeric_vals and direction != 'neutral':
                best = max(numeric_vals, key=numeric_vals.get) if direction == 'higher_better' else min(numeric_vals, key=numeric_vals.get)
                worst = min(numeric_vals, key=numeric_vals.get) if direction == 'higher_better' else max(numeric_vals, key=numeric_vals.get)
                row['_best'] = best
                row['_worst'] = worst
            else:
                row['_best'] = None
                row['_worst'] = None

            rows.append(row)

        table_df = pd.DataFrame(rows).set_index('Metric')

        # Render with color styling using HTML
        def _cell(val, stock, row_meta):
            best = row_meta.get('_best')
            worst = row_meta.get('_worst')
            stock_color = stock_colors[stock]
            if stock == best:
                bg = 'rgba(46, 125, 50, 0.18)'
                border = '2px solid #2e7d32'
            elif stock == worst:
                bg = 'rgba(183, 28, 28, 0.13)'
                border = '2px solid #b71c1c'
            else:
                bg = 'transparent'
                border = f'2px solid {stock_color}20'
            return f'<td style="text-align:center;background:{bg};border:{border};padding:6px 12px;font-weight:500">{val}</td>'

        # Build the HTML table
        html_rows_meta = {r['Metric']: r for r in rows}
        
        header_cols = ''.join(
            f'<th style="text-align:center;padding:8px 14px;background:{stock_colors[s]}22;'
            f'border-bottom:3px solid {stock_colors[s]};color:{stock_colors[s]};font-size:1.05rem">{s}</th>'
            for s in selected_stocks
        )
        header = f'<tr><th style="text-align:left;padding:8px 14px;min-width:200px">Metric</th>{header_cols}</tr>'

        body_html = ''
        for i, metric_label in enumerate(selected_metrics):
            col_name, direction, fmt = METRIC_OPTIONS[metric_label]
            if col_name not in comp_df.columns:
                continue
            meta = html_rows_meta.get(metric_label, {})
            row_bg = '#f8f9fa' if i % 2 == 0 else 'white'
            cells = ''.join(_cell(table_df.loc[metric_label, s], s, meta) for s in selected_stocks if s in table_df.columns)
            body_html += (
                f'<tr style="background:{row_bg}">'
                f'<td style="padding:6px 14px;font-weight:600;color:#374151">{metric_label}</td>'
                f'{cells}</tr>'
            )

        table_html = f"""
        <style>
            .comp-table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; border-radius: 8px; overflow: hidden; }}
            .comp-table th {{ font-weight: 700; white-space: nowrap; }}
            .comp-table td, .comp-table th {{ border: 1px solid #e5e7eb; }}
        </style>
        <div style="overflow-x:auto;border-radius:10px;border:1px solid #e5e7eb;box-shadow:0 1px 6px rgba(0,0,0,0.06)">
            <table class="comp-table">
                <thead style="background:#f3f4f6">{header}</thead>
                <tbody>{body_html}</tbody>
            </table>
        </div>
        <p style="font-size:0.78rem;color:#6b7280;margin-top:6px">
            🟢 <b>Green border</b> = best value for that metric &nbsp;|&nbsp;
            🔴 <b>Red border</b> = worst value for that metric
        </p>
        """
        st.markdown(table_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Distribution
# ─────────────────────────────────────────────────────────────────────────────
with tab_dist:
    _prev_tab = st.session_state.get('_comp_logged_tab')
    if _prev_tab != 'Distribution':
        logger.info(
            'COMPARISON | event=tab_view | tab=Distribution | stocks=%s',
            ','.join(selected_stocks),
        )
        st.session_state['_comp_logged_tab'] = 'Distribution'
    # Sync tab → URL
    if _qp_get('tab') != 'dist':
        _qp['tab'] = 'dist'

    dist_label_options = {k: v[0] for k, v in METRIC_OPTIONS.items()}
    _dist_metric_keys = list(dist_label_options.keys())

    # Resolve dist_metric from URL
    _dist_metric_qp = _qp_get('dist_metric', 'Dividend Yield (%)')
    _dist_metric_default_idx = (
        _dist_metric_keys.index(_dist_metric_qp)
        if _dist_metric_qp in _dist_metric_keys
        else _dist_metric_keys.index('Dividend Yield (%)')
    )

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])
    selected_dist_label = ctrl1.selectbox(
        'Metric', options=_dist_metric_keys,
        index=_dist_metric_default_idx,
        key='comp_dist_metric',
    )
    dist_col = dist_label_options[selected_dist_label]

    # Sync dist_metric → URL
    if _qp_get('dist_metric') != selected_dist_label:
        _qp['dist_metric'] = selected_dist_label
        if 'tab' not in _qp or _qp.get('tab') != 'dist':
            _qp['tab'] = 'dist'

    sector_opts = ['All'] + sorted(filtered_df['sector'].dropna().unique().tolist())

    # Resolve dist_sector from URL
    _dist_sector_qp = _qp_get('dist_sector', 'All')
    _dist_sector_default_idx = (
        sector_opts.index(_dist_sector_qp)
        if _dist_sector_qp in sector_opts
        else 0
    )
    selected_sector = ctrl2.selectbox(
        'Filter by Sector', options=sector_opts,
        index=_dist_sector_default_idx,
        key='comp_dist_sector',
    )

    # Sync dist_sector → URL
    if _qp_get('dist_sector') != selected_sector:
        _qp['dist_sector'] = selected_sector

    show_universe = ctrl3.toggle('Show Full Universe', value=True, key='comp_dist_universe',
                                  help='Overlay the full population distribution behind the selected stocks')

    # Dividend Yield-specific toggle: exclude stocks that never paid a dividend
    is_yield_metric = (selected_dist_label == 'Dividend Yield (%)')
    exclude_zero_yield = ctrl4.toggle(
        'Exclude 0% Yield',
        value=False,
        key='comp_dist_excl_zero',
        disabled=not is_yield_metric,
        help='Only visible for Dividend Yield — removes non-dividend-paying stocks from the distribution',
    )

    plot_universe_df = filtered_df.copy()
    if selected_sector != 'All':
        plot_universe_df = plot_universe_df[plot_universe_df['sector'] == selected_sector]

    # Apply zero-yield filter and report how many were removed
    n_removed = 0
    if is_yield_metric and exclude_zero_yield:
        before = len(plot_universe_df)
        plot_universe_df = plot_universe_df[plot_universe_df['yield'] > 0]
        n_removed = before - len(plot_universe_df)

    # Build comparison_vals dict for highlighting
    comparison_vals = {}
    for s in selected_stocks:
        if s in plot_universe_df.index and dist_col in plot_universe_df.columns:
            comparison_vals[s] = float(plot_universe_df.loc[s, dist_col])
        elif dist_col in comp_df.columns:
            comparison_vals[s] = float(comp_df.loc[s, dist_col])

    if dist_col not in plot_universe_df.columns or plot_universe_df.empty:
        st.warning('Metric not available for the current filter.')
    else:
        # Show removed-stock banner when zero-yield filter is active
        if is_yield_metric and exclude_zero_yield and n_removed > 0:
            total_before = n_removed + len(plot_universe_df)
            st.caption(
                f'🚫 **{n_removed} stocks with 0% dividend yield** removed from the distribution '
                f'({n_removed} of {total_before} · {n_removed/total_before*100:.1f}% of universe). '
                f'Showing **{len(plot_universe_df)} dividend-paying stocks** only.'
            )

        # KPIs
        valid_vals = plot_universe_df[dist_col].replace([float('inf'), -float('inf')], float('nan')).dropna()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric('Universe Mean',   f'{valid_vals.mean():.2f}')
        k2.metric('Universe Median', f'{valid_vals.median():.2f}')
        k3.metric('25th Pct',        f'{valid_vals.quantile(0.25):.2f}')
        k4.metric('75th Pct',        f'{valid_vals.quantile(0.75):.2f}')

        st.markdown(f'#### 📊 Distribution of {selected_dist_label}')

        q05 = float(valid_vals.quantile(0.05))
        q95 = float(valid_vals.quantile(0.95))

        if show_universe:
            # Full distribution with selected stocks as vertical rules
            dist_chart = hp.plot_card_distribution(
                plot_universe_df,
                dist_col,
                current_val=None,
                color='#6366f1',
                height=420,
                show_axis=True,
                comparison_vals=comparison_vals if comparison_vals else None,
                x_range=(q05, q95),
                fill_opacity=0.25,
                show_median=True,
            )
            st.altair_chart(dist_chart, width='stretch')
        else:
            # Only selected stocks — bar chart showing their individual values
            bar_data = pd.DataFrame({
                'Stock': list(comparison_vals.keys()),
                'Value': list(comparison_vals.values()),
            })
            _, direction, fmt = METRIC_OPTIONS[selected_dist_label]

            color_scale = alt.Scale(
                domain=list(stock_colors.keys()),
                range=list(stock_colors.values()),
            )

            bars = alt.Chart(bar_data).mark_bar(
                cornerRadiusTopLeft=6,
                cornerRadiusTopRight=6,
            ).encode(
                x=alt.X('Stock:N', sort=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
                y=alt.Y('Value:Q', title=selected_dist_label),
                color=alt.Color('Stock:N', scale=color_scale, legend=None),
                tooltip=[
                    alt.Tooltip('Stock:N', title='Stock'),
                    alt.Tooltip('Value:Q', title=selected_dist_label, format='.2f'),
                ],
            ).properties(height=380)

            # Value labels on bars
            labels = alt.Chart(bar_data).mark_text(
                dy=-8, fontSize=13, fontWeight='bold',
            ).encode(
                x=alt.X('Stock:N', sort=None),
                y=alt.Y('Value:Q'),
                text=alt.Text('Value:Q', format='.2f'),
                color=alt.Color('Stock:N', scale=color_scale, legend=None),
            )

            st.altair_chart((bars + labels).properties(height=380), width='stretch')

        # Stock-by-stock value summary below chart
        if comparison_vals:
            st.markdown('##### Selected Stock Values')
            cols = st.columns(min(len(comparison_vals), 5))
            for i, (stock, val) in enumerate(comparison_vals.items()):
                _, direction, fmt = METRIC_OPTIONS[selected_dist_label]
                pct_rank = (valid_vals < val).mean() * 100
                direction_text = 'higher' if direction == 'higher_better' else 'lower'
                cols[i % len(cols)].metric(
                    label=stock,
                    value=fmt.format(val),
                    delta=f'Top {100 - pct_rank:.0f}% ({direction_text} is better)' if direction != 'neutral' else f'{pct_rank:.0f}th percentile',
                    delta_color='normal' if (direction == 'higher_better' and pct_rank >= 50) or
                                           (direction == 'lower_better' and pct_rank <= 50) else 'inverse',
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Scatter Plot
# ─────────────────────────────────────────────────────────────────────────────
with tab_scatter:
    _prev_tab = st.session_state.get('_comp_logged_tab')
    if _prev_tab != 'Scatter':
        logger.info(
            'COMPARISON | event=tab_view | tab=Scatter | stocks=%s',
            ','.join(selected_stocks),
        )
        st.session_state['_comp_logged_tab'] = 'Scatter'
    # Sync tab → URL
    if _qp_get('tab') != 'scatter':
        _qp['tab'] = 'scatter'

    scatter_options = {k: v[0] for k, v in METRIC_OPTIONS.items() if v[0] in filtered_df.columns or v[0] == 'DScore'}
    scatter_keys = list(scatter_options.keys())

    # Special sentinel values for bubble-size selector
    _SIZE_NONE   = '— None (uniform size) —'
    _SIZE_SYMBOL = '★ Symbol (company logo)'
    _size_special = [_SIZE_NONE, _SIZE_SYMBOL]
    size_keys_extended = _size_special + scatter_keys

    # Resolve scatter axes from URL
    _sc_x_qp    = _qp_get('sc_x',    'Dividend Yield (%)')
    _sc_y_qp    = _qp_get('sc_y',    'PE Ratio')
    _sc_size_qp = _qp_get('sc_size', 'Market Cap')

    def _sc_idx(label, default):
        if label in scatter_keys:
            return scatter_keys.index(label)
        return scatter_keys.index(default) if default in scatter_keys else 0

    def _sc_size_idx(label, default):
        if label in size_keys_extended:
            return size_keys_extended.index(label)
        if default in size_keys_extended:
            return size_keys_extended.index(default)
        return size_keys_extended.index(_SIZE_NONE)

    sc1, sc2, sc3, sc4 = st.columns(4)
    x_label    = sc1.selectbox('X Axis',      options=scatter_keys,
                               index=_sc_idx(_sc_x_qp, 'Dividend Yield (%)'),
                               key='comp_sc_x')
    y_label    = sc2.selectbox('Y Axis',      options=scatter_keys,
                               index=_sc_idx(_sc_y_qp, 'PE Ratio'),
                               key='comp_sc_y')
    size_label = sc3.selectbox('Bubble Size', options=size_keys_extended,
                               index=_sc_size_idx(_sc_size_qp, 'Market Cap'),
                               key='comp_sc_size')

    # Sync scatter axes → URL
    _sc_updates = {}
    if _qp_get('sc_x')    != x_label:    _sc_updates['sc_x']    = x_label
    if _qp_get('sc_y')    != y_label:    _sc_updates['sc_y']    = y_label
    if _qp_get('sc_size') != size_label: _sc_updates['sc_size'] = size_label
    if _sc_updates:
        _qp.update(_sc_updates)
        if _qp.get('tab') != 'scatter':
            _qp['tab'] = 'scatter'

    show_universe_scatter = sc4.toggle(
        'Show Universe', value=False,
        key='comp_sc_universe',
        help='Show all stocks as grey background dots',
    )

    # Determine size mode
    _size_is_none   = (size_label == _SIZE_NONE)
    _size_is_symbol = (size_label == _SIZE_SYMBOL)
    size_col = scatter_options.get(size_label) if not _size_is_none and not _size_is_symbol else None

    x_col = scatter_options[x_label]
    y_col = scatter_options[y_label]

    # ── Helper: return a logo URL for the company ────────────────────────── #
    _JKSE_LOGO_BASE  = 'https://raw.githubusercontent.com/mitbal/daguerreo-data/refs/heads/main/jkse/logos'
    _SP500_LOGO_BASE = 'https://raw.githubusercontent.com/mitbal/daguerreo-data/refs/heads/main/sp500/logos'

    @st.cache_data(show_spinner=False)
    def _load_logo_b64(stock: str, market: str) -> str | None:
        """Return a direct SVG URL for the stock's logo.

        JKSE  → GitHub raw CDN under jkse/logos/{ticker}.svg
        Other → GitHub raw CDN under sp500/logos/{ticker}.svg
        """
        ticker = stock.split('.')[0]   # strip .JK / exchange suffix

        if market == 'JKSE':
            return f'{_JKSE_LOGO_BASE}/{ticker}.svg'

        return f'{_SP500_LOGO_BASE}/{ticker}.svg'


    # Validate columns exist
    missing = [c for c in [x_col, y_col] if c not in filtered_df.columns]
    if size_col and size_col not in filtered_df.columns:
        missing.append(size_col)
    if missing:
        st.warning(f'Some selected metrics are not available in the data: {missing}')
    elif x_col == y_col:
        st.warning(
            f'⚠️ **X Axis** and **Y Axis** are both set to **{x_label}**. '
            'Please choose two different metrics to plot a meaningful scatter chart.'
        )
    else:
        # ── Columns needed for selected-stocks frame ───────────────────────── #
        _cols_needed = [c for c in [x_col, y_col, size_col, 'sector'] if c]
        sc_comp = comp_df[_cols_needed].copy().dropna(subset=[x_col, y_col])
        sc_comp = sc_comp.reset_index()  # index → 'stock' column

        # ══════════════════════════════════════════════════════════════════════
        # MODE A — Symbol (company logos via Altair mark_image)
        # JKSE: logos are fetched directly from GitHub raw-content CDN (SVG).
        # Other markets: PNG data-URLs converted from local SVG via ImageMagick.
        # ══════════════════════════════════════════════════════════════════════
        if _size_is_symbol:
            # Attach logo URL to each row
            sc_comp['_logo_url'] = sc_comp['stock'].apply(
                lambda s: _load_logo_b64(s, sl) or ''
            )
            sc_with_logo    = sc_comp[sc_comp['_logo_url'] != ''].copy()
            sc_without_logo = sc_comp[sc_comp['_logo_url'] == ''].copy()

            color_scale_sel_sym = alt.Scale(
                domain=selected_stocks,
                range=[stock_colors[s] for s in selected_stocks],
            )

            # y-offset for labels in data-unit space
            _logo_px = 28
            _y_vals_sym  = sc_comp[y_col].values
            _y_range_sym = float(_y_vals_sym.max() - _y_vals_sym.min()) or abs(float(_y_vals_sym.mean())) * 0.1 or 1.0
            _dpp_sym     = (_y_range_sym * 1.1) / 520
            sc_comp = sc_comp.copy()
            sc_comp['_label_y'] = sc_comp[y_col] + (_logo_px + 6) * _dpp_sym

            sym_layers = []

            # Universe background dots (if toggled)
            if show_universe_scatter:
                _uni_cols_sym = [c for c in [x_col, y_col, 'sector'] if c]
                sc_universe_sym = filtered_df[_uni_cols_sym].copy().dropna().reset_index()
                q95_x = sc_universe_sym[x_col].quantile(0.95)
                q05_x = sc_universe_sym[x_col].quantile(0.05)
                q95_y = sc_universe_sym[y_col].quantile(0.95)
                q05_y = sc_universe_sym[y_col].quantile(0.05)
                sc_universe_sym = sc_universe_sym[
                    (sc_universe_sym[x_col] <= q95_x) & (sc_universe_sym[x_col] >= q05_x) &
                    (sc_universe_sym[y_col] <= q95_y) & (sc_universe_sym[y_col] >= q05_y)
                ]
                x_med_sym = float(sc_universe_sym[x_col].median())
                y_med_sym = float(sc_universe_sym[y_col].median())
                bg_sym = alt.Chart(sc_universe_sym).mark_circle(
                    opacity=0.18, color='#9ca3af', size=60,
                ).encode(
                    x=alt.X(f'{x_col}:Q', title=x_label, scale=alt.Scale(zero=False)),
                    y=alt.Y(f'{y_col}:Q', title=y_label, scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip('stock:N',  title='Stock'),
                        alt.Tooltip('sector:N', title='Sector'),
                        alt.Tooltip(f'{x_col}:Q', title=x_label, format='.2f'),
                        alt.Tooltip(f'{y_col}:Q', title=y_label, format='.2f'),
                    ],
                )
                vline_sym = alt.Chart(pd.DataFrame({x_col: [x_med_sym]})).mark_rule(
                    color='#d1d5db', strokeDash=[5, 5], opacity=0.8, strokeWidth=1.5,
                ).encode(x=f'{x_col}:Q')
                hline_sym = alt.Chart(pd.DataFrame({y_col: [y_med_sym]})).mark_rule(
                    color='#d1d5db', strokeDash=[5, 5], opacity=0.8, strokeWidth=1.5,
                ).encode(y=f'{y_col}:Q')
                sym_layers += [bg_sym, vline_sym, hline_sym]

            # Logo layer — mark_image with static-served SVG URLs
            if not sc_with_logo.empty:
                logo_layer = alt.Chart(sc_with_logo).mark_image(
                    width=44, height=44,
                ).encode(
                    x=alt.X(f'{x_col}:Q', title=x_label, scale=alt.Scale(zero=False)),
                    y=alt.Y(f'{y_col}:Q', title=y_label, scale=alt.Scale(zero=False)),
                    url='_logo_url:N',
                    tooltip=[
                        alt.Tooltip('stock:N',  title='Stock'),
                        alt.Tooltip('sector:N', title='Sector'),
                        alt.Tooltip(f'{x_col}:Q', title=x_label, format='.2f'),
                        alt.Tooltip(f'{y_col}:Q', title=y_label, format='.2f'),
                    ],
                )
                sym_layers.append(logo_layer)

            # Fallback circles for stocks with no logo
            if not sc_without_logo.empty:
                fallback_layer = alt.Chart(sc_without_logo).mark_circle(
                    size=800, opacity=0.85, stroke='white', strokeWidth=1.5,
                ).encode(
                    x=alt.X(f'{x_col}:Q', title=x_label, scale=alt.Scale(zero=False)),
                    y=alt.Y(f'{y_col}:Q', title=y_label, scale=alt.Scale(zero=False)),
                    color=alt.Color('stock:N', scale=color_scale_sel_sym, legend=None),
                    tooltip=[
                        alt.Tooltip('stock:N',  title='Stock'),
                        alt.Tooltip('sector:N', title='Sector'),
                        alt.Tooltip(f'{x_col}:Q', title=x_label, format='.2f'),
                        alt.Tooltip(f'{y_col}:Q', title=y_label, format='.2f'),
                    ],
                )
                sym_layers.append(fallback_layer)

            # Ticker labels floating above each logo
            sym_label_layer = alt.Chart(sc_comp).mark_text(
                align='center', baseline='bottom',
                fontSize=11, fontWeight='bold',
            ).encode(
                x=alt.X(f'{x_col}:Q', title=x_label, scale=alt.Scale(zero=False)),
                y=alt.Y('_label_y:Q', title=y_label, scale=alt.Scale(zero=False)),
                text=alt.Text('stock:N'),
                color=alt.Color('stock:N', scale=color_scale_sel_sym,
                                legend=alt.Legend(orient='right')),
            )
            sym_layers.append(sym_label_layer)

            sym_chart = alt.layer(*sym_layers).properties(height=520).interactive()
            st.altair_chart(sym_chart, width='stretch')



        # ══════════════════════════════════════════════════════════════════════
        # MODE B — Altair (None or metric-based bubble size)
        # ══════════════════════════════════════════════════════════════════════
        else:
            color_scale_sel = alt.Scale(
                domain=selected_stocks,
                range=[stock_colors[s] for s in selected_stocks],
            )

            # ── Build size encoding ──────────────────────────────────────────
            if _size_is_none:
                # Uniform size — no encoding, fixed pixel area
                _fixed_area = 600
                size_encoding = alt.value(_fixed_area)
                _px_radius_arr = np.full(len(sc_comp), np.sqrt(_fixed_area) / 2)
            else:
                # Metric-based bubble size
                size_encoding = alt.Size(f'{size_col}:Q', title=size_label,
                                         scale=alt.Scale(range=[200, 1800]))
                _sc_size_vals = sc_comp[size_col].values
                _sc_min, _sc_max = _sc_size_vals.min(), _sc_size_vals.max()
                if _sc_max > _sc_min:
                    _sc_norm = (_sc_size_vals - _sc_min) / (_sc_max - _sc_min)
                else:
                    _sc_norm = np.full(len(_sc_size_vals), 0.5)
                _px_area_arr   = 200 + _sc_norm * (1800 - 200)
                _px_radius_arr = np.sqrt(_px_area_arr) / 2

            # ── Tooltip list ─────────────────────────────────────────────────
            _tooltip = [
                alt.Tooltip('stock:N',    title='Stock'),
                alt.Tooltip('sector:N',   title='Sector'),
                alt.Tooltip(f'{x_col}:Q', title=x_label, format='.2f'),
                alt.Tooltip(f'{y_col}:Q', title=y_label, format='.2f'),
            ]
            if not _size_is_none:
                _tooltip.append(alt.Tooltip(f'{size_col}:Q', title=size_label, format='.2f'))

            # ── Base scatter for selected stocks ─────────────────────────── #
            selected_scatter = alt.Chart(sc_comp).mark_circle(
                opacity=1.0, stroke='white', strokeWidth=1.5,
            ).encode(
                x=alt.X(f'{x_col}:Q', title=x_label, scale=alt.Scale(zero=False)),
                y=alt.Y(f'{y_col}:Q', title=y_label, scale=alt.Scale(zero=False)),
                size=size_encoding,
                color=alt.Color('stock:N', title='Stock',
                                scale=color_scale_sel,
                                legend=alt.Legend(orient='right')),
                tooltip=_tooltip,
            )

            # ── Stock name labels ─────────────────────────────────────────── #
            _y_vals   = sc_comp[y_col].values
            _y_range  = float(_y_vals.max() - _y_vals.min()) or abs(float(_y_vals.mean())) * 0.1 or 1.0
            _data_per_px = (_y_range * 1.1) / 520

            sc_comp = sc_comp.copy()
            sc_comp['_label_y'] = sc_comp[y_col] + (_px_radius_arr + 6) * _data_per_px

            stock_labels = alt.Chart(sc_comp).mark_text(
                align='center',
                baseline='bottom',
                fontSize=12,
                fontWeight='bold',
            ).encode(
                x=alt.X(f'{x_col}:Q', scale=alt.Scale(zero=False)),
                y=alt.Y('_label_y:Q', scale=alt.Scale(zero=False)),
                text=alt.Text('stock:N'),
                color=alt.Color('stock:N', scale=color_scale_sel, legend=None),
            )

            layers = []

            if show_universe_scatter:
                _uni_cols_b = [c for c in [x_col, y_col, size_col, 'sector'] if c]
                sc_universe = filtered_df[_uni_cols_b].copy().dropna(subset=[x_col, y_col]).reset_index()
                q95_x = sc_universe[x_col].quantile(0.95)
                q05_x = sc_universe[x_col].quantile(0.05)
                q95_y = sc_universe[y_col].quantile(0.95)
                q05_y = sc_universe[y_col].quantile(0.05)
                sc_universe = sc_universe[
                    (sc_universe[x_col] <= q95_x) & (sc_universe[x_col] >= q05_x) &
                    (sc_universe[y_col] <= q95_y) & (sc_universe[y_col] >= q05_y)
                ]

                background = alt.Chart(sc_universe).mark_circle(
                    opacity=0.18, color='#9ca3af', size=60,
                ).encode(
                    x=alt.X(f'{x_col}:Q', title=x_label, scale=alt.Scale(zero=False)),
                    y=alt.Y(f'{y_col}:Q', title=y_label, scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip('stock:N',    title='Stock'),
                        alt.Tooltip('sector:N',   title='Sector'),
                        alt.Tooltip(f'{x_col}:Q', title=x_label, format='.2f'),
                        alt.Tooltip(f'{y_col}:Q', title=y_label, format='.2f'),
                    ],
                )

                # Median lines from full universe
                x_med = float(sc_universe[x_col].median())
                y_med = float(sc_universe[y_col].median())
                vline = alt.Chart(pd.DataFrame({x_col: [x_med]})).mark_rule(
                    color='#d1d5db', strokeDash=[5, 5], opacity=0.8, strokeWidth=1.5
                ).encode(x=f'{x_col}:Q')
                hline = alt.Chart(pd.DataFrame({y_col: [y_med]})).mark_rule(
                    color='#d1d5db', strokeDash=[5, 5], opacity=0.8, strokeWidth=1.5
                ).encode(y=f'{y_col}:Q')

                layers = [background, vline, hline, selected_scatter, stock_labels]
            else:
                layers = [selected_scatter, stock_labels]

            chart = alt.layer(*layers).properties(height=520).interactive()
            st.altair_chart(chart, width='stretch')

        # ── Quick stats table below scatter ──────────────────────────────── #
        def _safe_isnan(v):
            """Return True when v is a missing / non-finite numeric value."""
            try:
                return np.isnan(float(v))
            except (TypeError, ValueError):
                return v is None or v == ''

        def _fmt_val(fmt, v):
            if _safe_isnan(v):
                return 'N/A'
            try:
                return fmt.format(v)
            except (TypeError, ValueError):
                return str(v)

        st.markdown('##### Selected Stocks — Key Values')
        quick_cols = st.columns(len(selected_stocks))
        for i, stock in enumerate(selected_stocks):
            if stock not in comp_df.index:
                continue
            try:
                row = comp_df.loc[stock]
                x_val = row[x_col] if x_col in row.index else float('nan')
                y_val = row[y_col] if y_col in row.index else float('nan')

                _, _, x_fmt = METRIC_OPTIONS[x_label]
                _, _, y_fmt = METRIC_OPTIONS[y_label]

                # Size value line (only for metric modes)
                if not _size_is_none and not _size_is_symbol and size_col in row.index:
                    sz_val  = row[size_col]
                    _, _, sz_fmt = METRIC_OPTIONS[size_label]
                    size_line = (
                        f'<div style="font-size:0.8rem;color:#6b7280">'
                        f'{size_label}: <b>{_fmt_val(sz_fmt, sz_val)}</b></div>'
                    )
                else:
                    size_line = ''

                # Logo thumbnail (for Symbol mode)
                if _size_is_symbol:
                    logo_url = _load_logo_b64(stock, sl)
                    logo_html = (
                        f'<img src="{logo_url}" style="height:36px;margin-bottom:6px;object-fit:contain"/>'
                        if logo_url else ''
                    )
                else:
                    logo_html = ''

                card_html = f"""
<div style="border:2px solid {stock_colors[stock]};border-radius:10px;padding:12px 14px;text-align:center;">
    {logo_html}
    <div style="font-size:1.1rem;font-weight:800;color:{stock_colors[stock]}">{stock}</div>
    <div style="font-size:0.8rem;color:#6b7280;margin-top:4px">{x_label}: <b>{_fmt_val(x_fmt, x_val)}</b></div>
    <div style="font-size:0.8rem;color:#6b7280">{y_label}: <b>{_fmt_val(y_fmt, y_val)}</b></div>
    {size_line}
</div>
""".strip()
                quick_cols[i].html(card_html)
            except Exception as e:
                logger.warning('COMPARISON | event=key_values_error | stock=%s | error=%s', stock, e)
                quick_cols[i].warning(f'Could not render key values for {stock}.')
