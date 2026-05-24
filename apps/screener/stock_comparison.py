import io
import time
import logging

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
# ── Market selection sidebar (same as stock_picker.py) ───────────────────────
# ══════════════════════════════════════════════════════════════════════════════
stock_select = st.sidebar.radio(
    'Stock List Selection',
    ['Indonesian Stock', 'S&P 500 (US and World Stock)'],
    horizontal=False,
    key='comp_sl',
)

sl = 'JKSE' if stock_select == 'Indonesian Stock' else 'S&P500'

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


# ── Stock multiselect ────────────────────────────────────────────────────────
st.markdown('### 🔍 Select Stocks to Compare')
selected_stocks = st.multiselect(
    label='Choose 2 or more stocks from the universe',
    options=stock_options,
    placeholder='Type or pick stock codes…',
    key='comp_stocks',
)

if len(selected_stocks) < 2:
    st.info('👆 Select at least **2 stocks** above to start comparing.')
    st.stop()

# Build colour mapping for selected stocks
stock_colors = {s: _STOCK_PALETTE[i % len(_STOCK_PALETTE)] for i, s in enumerate(selected_stocks)}
comp_df = filtered_df.loc[selected_stocks].copy()


# ══════════════════════════════════════════════════════════════════════════════
# ── Tabs ─────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
tab_table, tab_dist, tab_scatter = st.tabs(['📋 Table', '📊 Distribution', '🔵 Scatter Plot'])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Table
# ─────────────────────────────────────────────────────────────────────────────
with tab_table:
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

    dist_label_options = {k: v[0] for k, v in METRIC_OPTIONS.items()}

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])
    selected_dist_label = ctrl1.selectbox(
        'Metric', options=list(dist_label_options.keys()),
        index=list(dist_label_options.keys()).index('Dividend Yield (%)'),
        key='comp_dist_metric',
    )
    dist_col = dist_label_options[selected_dist_label]

    sector_opts = ['All'] + sorted(filtered_df['sector'].dropna().unique().tolist())
    selected_sector = ctrl2.selectbox('Filter by Sector', options=sector_opts, key='comp_dist_sector')
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

    scatter_options = {k: v[0] for k, v in METRIC_OPTIONS.items() if v[0] in filtered_df.columns or v[0] == 'DScore'}
    scatter_keys = list(scatter_options.keys())

    sc1, sc2, sc3, sc4 = st.columns(4)
    x_label    = sc1.selectbox('X Axis',  options=scatter_keys,
                               index=scatter_keys.index('PE Ratio') if 'PE Ratio' in scatter_keys else 0,
                               key='comp_sc_x')
    y_label    = sc2.selectbox('Y Axis',  options=scatter_keys,
                               index=scatter_keys.index('Dividend Yield (%)') if 'Dividend Yield (%)' in scatter_keys else 1,
                               key='comp_sc_y')
    size_label = sc3.selectbox('Bubble Size', options=scatter_keys,
                               index=scatter_keys.index('Market Cap') if 'Market Cap' in scatter_keys else 2,
                               key='comp_sc_size')

    show_universe_scatter = sc4.toggle(
        'Show Universe', value=False,
        key='comp_sc_universe',
        help='Show all stocks as grey background dots',
    )

    x_col    = scatter_options[x_label]
    y_col    = scatter_options[y_label]
    size_col = scatter_options[size_label]

    # Validate columns exist
    missing = [c for c in [x_col, y_col, size_col] if c not in filtered_df.columns]
    if missing:
        st.warning(f'Some selected metrics are not available in the data: {missing}')
    else:
        # Prepare selected stocks DataFrame
        sc_comp = comp_df[[x_col, y_col, size_col, 'sector']].copy().dropna()
        sc_comp = sc_comp.reset_index()  # index = stock name

        # ── Remove extreme outliers from selected stocks ─────────────────── #
        # (No outlier removal for selected stocks — we always want to show them)

        color_scale_sel = alt.Scale(
            domain=selected_stocks,
            range=[stock_colors[s] for s in selected_stocks],
        )

        # ── Base scatter for selected stocks ─────────────────────────────── #
        selected_scatter = alt.Chart(sc_comp).mark_circle(
            opacity=1.0, stroke='white', strokeWidth=1.5,
        ).encode(
            x=alt.X(f'{x_col}:Q', title=x_label, scale=alt.Scale(zero=False)),
            y=alt.Y(f'{y_col}:Q', title=y_label, scale=alt.Scale(zero=False)),
            size=alt.Size(f'{size_col}:Q', title=size_label,
                          scale=alt.Scale(range=[200, 1800])),
            color=alt.Color('stock:N', title='Stock',
                            scale=color_scale_sel,
                            legend=alt.Legend(orient='right')),
            tooltip=[
                alt.Tooltip('stock:N',      title='Stock'),
                alt.Tooltip('sector:N',     title='Sector'),
                alt.Tooltip(f'{x_col}:Q',   title=x_label,    format='.2f'),
                alt.Tooltip(f'{y_col}:Q',   title=y_label,    format='.2f'),
                alt.Tooltip(f'{size_col}:Q', title=size_label, format='.2f'),
            ],
        )

        # ── Stock name labels ─────────────────────────────────────────────── #
        stock_labels = alt.Chart(sc_comp).mark_text(
            align='left',
            baseline='middle',
            dx=14,
            fontSize=12,
            fontWeight='bold',
        ).encode(
            x=alt.X(f'{x_col}:Q', scale=alt.Scale(zero=False)),
            y=alt.Y(f'{y_col}:Q', scale=alt.Scale(zero=False)),
            text=alt.Text('stock:N'),
            color=alt.Color('stock:N', scale=color_scale_sel, legend=None),
        )

        layers = []

        if show_universe_scatter:
            # Background grey dots — full universe (outliers removed)
            sc_universe = filtered_df[[x_col, y_col, size_col, 'sector']].copy().dropna().reset_index()
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
        st.markdown('##### Selected Stocks — Key Values')
        quick_cols = st.columns(len(selected_stocks))
        for i, stock in enumerate(selected_stocks):
            if stock not in comp_df.index:
                continue
            row = comp_df.loc[stock]
            x_val = row.get(x_col, float('nan'))
            y_val = row.get(y_col, float('nan'))
            sz_val = row.get(size_col, float('nan'))

            _, _, x_fmt = METRIC_OPTIONS[x_label]
            _, _, y_fmt = METRIC_OPTIONS[y_label]
            _, _, sz_fmt = METRIC_OPTIONS[size_label]

            quick_cols[i].markdown(
                f"""
                <div style="border:2px solid {stock_colors[stock]};border-radius:10px;padding:12px 14px;text-align:center;">
                    <div style="font-size:1.1rem;font-weight:800;color:{stock_colors[stock]}">{stock}</div>
                    <div style="font-size:0.8rem;color:#6b7280;margin-top:4px">{x_label}: <b>{x_fmt.format(x_val) if not np.isnan(x_val) else 'N/A'}</b></div>
                    <div style="font-size:0.8rem;color:#6b7280">{y_label}: <b>{y_fmt.format(y_val) if not np.isnan(y_val) else 'N/A'}</b></div>
                    <div style="font-size:0.8rem;color:#6b7280">{size_label}: <b>{sz_fmt.format(sz_val) if not np.isnan(sz_val) else 'N/A'}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
