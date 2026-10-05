import itertools
import hashlib
import concurrent.futures
import os
import pickle
import streamlit as st
import datetime
import pandas as pd
import numpy as np
from st_supabase_connection import SupabaseConnection

import harvest.data as hd
import harvest.plot as hp
import harvest.prices as hprices
import harvest.swing as sw
from harvest.premium import require_premium
from harvest.swing import (
    STRATEGY_ENGINE_VERSION,
    DEFAULT_PROFIT_TARGET,
    DEFAULT_STOP_LOSS,
    DEFAULT_MAX_HOLD,
    DEFAULT_SLIPPAGE_TICKS,
    MIN_PROFIT_FACTOR,
    MIN_T_STAT,
    MIN_STOCK_VALIDATION_TRADES,
    STRATEGY_REGIME_RSI,
    STRATEGY_CROSS_MOMENTUM,
    STRATEGY_SUPERTREND_PULLBACK,
    STRATEGY_BREADTH_RSI2,
    STRATEGY_TYPES,
    STRATEGY_PARAMETER_LABELS,
    STRATEGY_PARAMETER_INPUTS,
    SEARCH_PRESETS,
)

st.set_page_config(page_title='Short-Term Swing Trading - Panen Dividen')
require_premium()
hp.enable_chart_theme()
st.title('Short-Term Swing Trading Strategy Lab')

@st.cache_data(max_entries=64)
def df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


@st.cache_data(ttl=6 * 60 * 60, max_entries=16, show_spinner=False)
def load_price_volatility(symbols: tuple[str, ...], lookback: int = 60) -> pd.DataFrame:
    """Calculate recent daily volatility from stored close history."""
    columns = ['stock', 'Daily Volatility', 'Volatility Observations']
    try:
        # Two calendar days per session leaves room for weekends and IDX holidays.
        start = (datetime.date.today() - datetime.timedelta(days=int(lookback) * 2 + 14)).isoformat()
        prices = fetch_close_history(tuple(symbols), start)
        if prices.empty:
            return pd.DataFrame(columns=columns)
        prices = prices.sort_values(['symbol', 'date'])
        prices = prices.groupby('symbol', group_keys=False).tail(int(lookback) + 1)
        prices['return'] = prices.groupby('symbol')['close'].pct_change()

        # Ignore likely corporate-action/data errors while retaining genuinely
        # volatile IDX sessions. They would otherwise dominate the ranking.
        prices.loc[prices['return'].abs() > 0.35, 'return'] = np.nan
        volatility = prices.groupby('symbol')['return'].agg(['std', 'count']).reset_index()
        volatility['Daily Volatility'] = volatility['std'] * 100
        volatility = volatility.rename(columns={
            'symbol': 'stock',
            'count': 'Volatility Observations',
        })
        return volatility[columns]
    except Exception:
        return pd.DataFrame(columns=columns)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def load_live_daily_prices(stock: str, start_from: str) -> pd.DataFrame:
    """Load fresh daily closes for the live scanner, cached briefly."""
    try:
        prices = hd.get_daily_stock_price(stock, start_from=start_from)
        if prices is None or prices.empty:
            return pd.DataFrame(columns=['date', 'close'])
        prices = prices[['date', 'close']].copy()
        prices['date'] = pd.to_datetime(prices['date'], errors='coerce')
        prices['close'] = pd.to_numeric(prices['close'], errors='coerce')
        return prices.dropna().drop_duplicates('date', keep='last').sort_values('date')
    except Exception:
        return pd.DataFrame(columns=['date', 'close'])

st.sidebar.markdown(
    """
    **Short-term swing trading** uses completed daily closes to research 1–10
    session mean-reversion, pullback, relative-strength, and breakout setups.

    * **Mean reversion:** regime RSI, Bollinger lower-band, RSI recovery, and
      market-wide breadth capitulation with RSI(2).
    * **Trend and momentum:** Donchian, squeeze, EMA pullback, relative strength,
      cross-sectional momentum, and close-based Supertrend pullback.
    * **Execution:** a completed closing setup enters on the next session in the
      backtest; live setups remain provisional until today's close.

    Sweep compact parameter grids, validate on recent held-out data, inspect an
    individual ticker, and scan the current shortlist for actionable setups.
    """
)

# Initialize Supabase explicitly inside Streamlit
@st.cache_resource(show_spinner=False)
def get_db_connection() -> SupabaseConnection:
    conn = st.connection("supabase", type=SupabaseConnection)
    try:
         conn.auth.sign_in_with_password({
             "email": st.secrets["connections"]["supabase"]["EMAIL_ADDRESS"],
             "password": st.secrets["connections"]["supabase"]["PASSWORD"],
         })
    except Exception:
         pass
    return conn

conn = get_db_connection()


@st.cache_data(ttl=6 * 60 * 60, max_entries=32, show_spinner=False)
def fetch_close_history(symbols: tuple[str, ...], start: str) -> pd.DataFrame:
    """Daily closes for ``symbols`` since ``start`` from the Parquet price history."""
    return hprices.load_prices('jkse', symbols=symbols, start=start)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_dividend_events() -> dict[str, pd.DataFrame]:
    """Load ex-dividend dates and amounts per symbol for price adjustment.

    The pipeline publishes this pickle to Supabase storage; the local copy is a
    fallback for development.
    """
    path = 'data/jkse/dividends.pkl'
    raw = None
    try:
        raw = pickle.loads(conn.client.storage.from_('harvest_dividend').download(path))
    except Exception:
        try:
            raw = pd.read_pickle(path)
        except Exception:
            return {}

    events = {}
    for symbol, records in (raw or {}).items():
        frame = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records
        if frame.empty or 'date' not in frame.columns:
            continue
        # adjDividend is split-adjusted, matching the split-adjusted closes.
        amount_col = 'adjDividend' if 'adjDividend' in frame.columns else 'dividend'
        if amount_col not in frame.columns:
            continue
        frame = pd.DataFrame({
            'date': pd.to_datetime(frame['date'], errors='coerce'),
            'dividend': pd.to_numeric(frame[amount_col], errors='coerce'),
        }).dropna()
        frame = frame[frame['dividend'] > 0]
        if not frame.empty:
            events[symbol] = frame.groupby('date', as_index=False)['dividend'].sum()
    return events


@st.cache_data(ttl=6 * 60 * 60, max_entries=8, show_spinner=False)
def load_close_history(symbols: tuple[str, ...], start: str) -> pd.DataFrame:
    """Stored daily closes, used for market breadth when no fetched prices exist."""
    try:
        return fetch_close_history(symbols, start)
    except Exception:
        return pd.DataFrame(columns=['symbol', 'date', 'close'])


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=6 * 60 * 60, show_spinner="Loading JKSE index data…")
def load_jkse_index() -> pd.DataFrame:
    """Load JKSE/IHSG index data. Tries live API first, falls back to CSV."""
    try:
        df = hd.get_daily_stock_price('^JKSE', start_from='2000-01-01')
        if df is not None and not df.empty:
            df = df.rename(columns={'date': 'date', 'close': 'close'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            return df[['close']]
    except Exception:
        pass
    try:
        df = pd.read_csv('data/jkse.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df[['close']]
    except Exception:
        return pd.DataFrame(columns=['close'])


# ═══════════════════════════════════════════════════════════════════════════════
# TOP RANKED VALUATION SHORTLIST
# ═══════════════════════════════════════════════════════════════════════════════

with st.container():
    st.subheader("Top Ranked Shortlist", divider='blue')
    filter_cols = st.columns(4)
    min_mcap_trill = filter_cols[0].number_input("Min Market Cap (Trillion IDR)", value=1.0, step=1.0, key="dt_mcap")
    min_pe = filter_cols[1].number_input("Minimum PE", value=1.0, step=1.0, key="dt_minpe")
    max_pe = filter_cols[2].number_input("Maximum PE", value=20.0, step=1.0, key="dt_maxpe")
    max_stocks = filter_cols[3].number_input("Max Stocks Output", min_value=10, max_value=200, value=200, step=10, key="dt_maxs")
    rank_cols = st.columns(3)
    volatility_weight = rank_cols[0].slider(
        "Volatility Weight", min_value=0, max_value=100, value=60, step=5,
        key="dt_volatility_weight",
        help="Higher values favor stocks with larger recent daily price moves",
    )
    volatility_lookback = rank_cols[1].number_input(
        "Volatility Lookback", min_value=20, max_value=252, value=60, step=10,
        key="dt_volatility_lookback",
    )
    min_volatility = rank_cols[2].number_input(
        "Minimum Daily Volatility %", min_value=0.0, max_value=20.0, value=1.0, step=0.25,
        key="dt_min_volatility",
        help="Exclude stocks whose daily returns are too quiet for short-term trading",
    )
    min_value_traded_bn = st.number_input(
        "Min Avg Daily Value Traded (Billion IDR)", min_value=0.0, max_value=500.0,
        value=5.0, step=1.0, key="dt_min_value_traded",
        help="Average volume x price. Thin stocks have wide spreads and cannot be exited at the modelled close.",
    )
    include_blue_chips = st.checkbox("Pin Major Blue-chips", value=True, help="Always show major blue-chip stocks at the top of the list", key="dt_blue")

    final_df = pd.DataFrame()
    try:
        val_df = pd.read_csv('data/jkse/valuation.csv')
        try:
            cp_df = pd.read_csv('data/jkse/company_profiles.csv')
            val_df = val_df.merge(
                cp_df[['symbol', 'mktCap', 'price', 'volAvg']],
                left_on='stock', right_on='symbol', how='inner',
            )
        except Exception:
            pass

        if 'last_1y_mean' in val_df.columns and 'last_5y_mean' in val_df.columns:
            val_df['avg_pe'] = val_df[['last_1y_mean', 'last_3y_mean', 'last_5y_mean']].mean(axis=1)
        else:
            val_df['avg_pe'] = val_df[['last_2y_mean', 'last_3y_mean', 'last_10y_mean']].mean(axis=1)

        val_df['Discount'] = (1 - (val_df['current_pe'] / val_df['avg_pe'])) * 100

        mask = pd.Series(True, index=val_df.index)
        if 'mktCap' in val_df.columns:
            mask &= (val_df['mktCap'] >= (min_mcap_trill * 1_000_000_000_000))

        mask &= (val_df['current_pe'] >= min_pe) & (val_df['current_pe'] <= max_pe)
        mask &= (val_df['avg_pe'] > 0) & (val_df['avg_pe'] < 100)

        if {'volAvg', 'price'}.issubset(val_df.columns):
            val_df['Avg Value Traded'] = val_df['volAvg'] * val_df['price']
            mask &= val_df['Avg Value Traded'] >= float(min_value_traded_bn) * 1_000_000_000

        eligible_df = val_df[mask].copy()
        volatility_df = load_price_volatility(
            tuple(sorted(eligible_df['stock'].dropna().unique())),
            int(volatility_lookback),
        )
        eligible_df = eligible_df.merge(volatility_df, on='stock', how='left')
        min_observations = max(15, int(volatility_lookback) // 2)
        eligible_df = eligible_df[
            (eligible_df['Volatility Observations'] >= min_observations)
            & (eligible_df['Daily Volatility'] >= float(min_volatility))
        ].copy()

        eligible_df['Discount Score'] = eligible_df['Discount'].rank(pct=True) * 100
        eligible_df['Volatility Score'] = eligible_df['Daily Volatility'].rank(pct=True) * 100
        vol_weight = float(volatility_weight) / 100
        eligible_df['Trading Rank'] = (
            eligible_df['Discount Score'] * (1 - vol_weight)
            + eligible_df['Volatility Score'] * vol_weight
        )
        filtered_df = eligible_df.sort_values(
            ['Trading Rank', 'Daily Volatility', 'Discount'], ascending=False,
        )

        special_stocks_df = pd.DataFrame()
        if include_blue_chips:
            blue_chips = [
                'BBCA.JK', 'BMRI.JK', 'BBRI.JK', 'BBNI.JK', 'INDF.JK', 'ICBP.JK',
                'UNVR.JK', 'AMRT.JK', 'TLKM.JK', 'EXCL.JK', 'ISAT.JK', 'ASII.JK',
                'UNTR.JK', 'ADRO.JK', 'PTBA.JK', 'ITMG.JK', 'BRIS.JK', 'KLBF.JK', 'GOTO.JK'
            ]
            special_stocks_df = eligible_df[eligible_df['stock'].isin(blue_chips)].copy()

        if not special_stocks_df.empty:
            special_stocks_df = special_stocks_df.drop_duplicates(subset=['stock']).sort_values(
                ['Trading Rank', 'Daily Volatility'], ascending=False,
            )
            filtered_df = filtered_df[~filtered_df['stock'].isin(special_stocks_df['stock'])]
            final_df = pd.concat([special_stocks_df, filtered_df]).head(int(max_stocks))
        else:
            final_df = filtered_df.head(int(max_stocks))

        top_stocks = final_df[[
            'stock', 'current_pe', 'avg_pe', 'Discount', 'Daily Volatility',
            'Discount Score', 'Volatility Score', 'Trading Rank',
        ]].rename(
            columns={'stock': 'Symbol', 'current_pe': 'Current PE', 'avg_pe': 'Historical Avg PE'}
        )

        st.dataframe(
            top_stocks,
            column_config={
                "Discount": st.column_config.NumberColumn("Discount vs Avg PE", format="%.2f%%"),
                "Current PE": st.column_config.NumberColumn("Current PE", format="%.2f"),
                "Historical Avg PE": st.column_config.NumberColumn("Historical Avg PE", format="%.2f"),
                "Daily Volatility": st.column_config.NumberColumn("Daily Volatility", format="%.2f%%"),
                "Discount Score": st.column_config.NumberColumn("Discount Score", format="%.1f"),
                "Volatility Score": st.column_config.NumberColumn("Volatility Score", format="%.1f"),
                "Trading Rank": st.column_config.ProgressColumn(
                    "Trading Rank", format="%.1f", min_value=0, max_value=100,
                ),
            },
            hide_index=True,
            width='stretch'
        )
    except Exception as e:
        st.warning(f"Could not load valuation data: {e}")

shortlist_tickers = final_df['stock'].tolist() if not final_df.empty else []


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET REGIME STATUS BAR
# ═══════════════════════════════════════════════════════════════════════════════

jkse_df = load_jkse_index()
# Widgets further down own these values; read them from session state so the
# regime shown here matches what the optimizer, backtest, and scanner use.
_regime_fast = int(st.session_state.get('dt_reg_fast', 50))
_regime_slow = int(st.session_state.get('dt_reg_slow', 200))
full_regime_series = sw.compute_regime_series(jkse_df, fast_sma=_regime_fast, slow_sma=_regime_slow)
current_regime = sw.get_current_regime(full_regime_series)

if current_regime:
    regime_emoji = "🐂" if current_regime == 'bull' else "🐻"
    st.info(
        f"{regime_emoji} **Current Market Regime: {current_regime.upper()}**  "
        f"(JKSE {_regime_fast}/{_regime_slow} SMA crossover — "
        f"last regime change: {full_regime_series[full_regime_series != full_regime_series.shift(1)].index[-1].strftime('%d %b %Y') if len(full_regime_series) > 1 else 'N/A'})"
    )
else:
    st.warning("⚠️ Could not determine market regime. JKSE index data unavailable.")


# ═══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

OPTIMIZER_WORKERS = min(8, max(2, os.cpu_count() or 2))


st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Find the Best Short-Term Swing Strategy", divider='violet')
st.caption(
    "Test compact strategy-specific parameter sets without varying irrelevant inputs. "
    "The most recent data is held out, then each strategy is ranked by how "
    "consistently it performs across the full stock shortlist. Cross-sectional "
    "momentum is ranked against the same shortlist on every date."
)

search_cols = st.columns(3)
search_depth = search_cols[0].selectbox(
    "Search Depth", options=list(SEARCH_PRESETS), index=1, key="dt_search_depth",
    help="Quick for iteration, Balanced for normal use, Deep for final research",
)
strategy_types = search_cols[1].multiselect(
    "Strategies", options=STRATEGY_TYPES,
    default=STRATEGY_TYPES, key="dt_strategy_types",
    help="Compare mean-reversion, trend, breakout, and relative-strength setups",
)
opt_validation_pct = search_cols[2].slider(
    "Holdout Data %", min_value=20, max_value=40, value=30, step=5,
    key="dt_validation_pct",
)

date_cols = st.columns(2)
opt_start = date_cols[0].date_input("Start Date", value=datetime.date(2022, 1, 1), key="dt_start")
opt_end = date_cols[1].date_input("End Date", value=datetime.date.today(), key="dt_end")

with st.expander("Advanced: market regime, trading costs, and price data"):
    advanced_cols = st.columns(5)
    regime_fast = advanced_cols[0].number_input(
        "Regime Fast SMA", min_value=10, max_value=100, value=50, step=10, key="dt_reg_fast",
    )
    regime_slow = advanced_cols[1].number_input(
        "Regime Slow SMA", min_value=50, max_value=300, value=200, step=10, key="dt_reg_slow",
    )
    opt_buy_fee = advanced_cols[2].number_input(
        "Buy Fee %", min_value=0.0, max_value=5.0, value=0.15, step=0.01, key="dt_buy_fee",
    )
    opt_sell_fee = advanced_cols[3].number_input(
        "Sell Fee %", min_value=0.0, max_value=5.0, value=0.15, step=0.01, key="dt_sell_fee",
    )
    opt_sell_tax = advanced_cols[4].number_input(
        "Sell Tax %", min_value=0.0, max_value=5.0, value=0.1, step=0.01, key="dt_sell_tax",
    )
    data_cols = st.columns(2)
    opt_slippage_ticks = data_cols[0].number_input(
        "Slippage (ticks per side)", min_value=0.0, max_value=5.0,
        value=DEFAULT_SLIPPAGE_TICKS, step=0.5, key="dt_slippage_ticks",
        help="IDX price steps lost on each fill; 0.5 per side is one full spread per round trip. One tick is ~0.5-1% for stocks under Rp 500.",
    )
    adjust_dividends = data_cols[1].checkbox(
        "Adjust prices for dividends", value=True, key="dt_adjust_dividends",
        help="Removes ex-dividend drops, which otherwise look like oversold dips.",
    )

dividend_events = load_dividend_events() if adjust_dividends else {}
if adjust_dividends and not dividend_events:
    st.caption("Dividend data unavailable; strategies are using unadjusted closes.")


def _prepare_close(symbol: str, close: pd.Series) -> pd.Series:
    """Apply the page's price-adjustment setting to one close series."""
    return sw.dividend_adjust(close, dividend_events.get(symbol))

search_space = SEARCH_PRESETS[search_depth]
hold_days = search_space['hold_days']
exit_rules = search_space['exit_rules']
param_combos = []
for strategy_type in strategy_types:
    param_combos.extend(
        (strategy_type, *strategy_params, hold_d, profit_target, stop_loss)
        for strategy_params, hold_d, (profit_target, stop_loss)
        in itertools.product(search_space['strategy_params'][strategy_type], hold_days, exit_rules)
    )
total_combos = len(param_combos) * len(shortlist_tickers)

info_cols = st.columns(5)
info_cols[0].metric("Strategies", f"{len(param_combos):,}")
info_cols[1].metric("Stocks", f"{len(shortlist_tickers):,}")
info_cols[2].metric("Backtest Runs", f"{total_combos:,}")
info_cols[3].metric("Holdout", f"Latest {opt_validation_pct}%")
info_cols[4].metric("Parallel Workers", OPTIMIZER_WORKERS)

if total_combos > 3000:
    st.warning(
        f"{total_combos:,} combinations across {len(shortlist_tickers)} stocks "
        "may take several minutes. Use Quick search while iterating."
    )

run_optimizer = st.button("Find Best Strategy", type="primary", disabled=(total_combos == 0 or len(shortlist_tickers) == 0))

price_cache_key = hashlib.md5(
    f"{sorted(shortlist_tickers)}{opt_start}{opt_end}".encode()
).hexdigest()
results_cache_key = hashlib.md5(
    f"{STRATEGY_ENGINE_VERSION}{param_combos}{regime_fast}{regime_slow}{opt_start}{opt_end}{shortlist_tickers}{opt_buy_fee}{opt_sell_fee}{opt_sell_tax}{opt_validation_pct}{opt_slippage_ticks}{adjust_dividends}".encode()
).hexdigest()

if run_optimizer:
    st.session_state.pop('dt_optimizer_results', None)
    st.session_state.pop('dt_optimizer_results_cache_key', None)

    regime_series = sw.compute_regime_series(jkse_df, fast_sma=int(regime_fast), slow_sma=int(regime_slow))

    if st.session_state.get('dt_price_cache_key') == price_cache_key and \
       'dt_prices' in st.session_state:
        all_prices_df = st.session_state['dt_prices']
        st.info(f"Using cached price data ({len(all_prices_df):,} rows). Only grid params changed.")
    else:
        try:
            start_str = opt_start.strftime('%Y-%m-%d')
            with st.spinner("Loading price history..."):
                all_prices_df = fetch_close_history(tuple(sorted(shortlist_tickers)), start_str)
            if not all_prices_df.empty:
                all_prices_df = all_prices_df[all_prices_df['date'] <= pd.to_datetime(opt_end)]
                all_prices_df = all_prices_df.sort_values(['symbol', 'date'])

            st.session_state['dt_prices'] = all_prices_df
            st.session_state['dt_price_cache_key'] = price_cache_key

        except Exception as e:
            st.error(f"Failed to fetch price data: {e}")
            all_prices_df = pd.DataFrame()

    if all_prices_df.empty:
        st.error("No price data returned. Make sure the shortlist is non-empty and price history is available.")
    else:
        fetch_cols = st.columns(3)
        fetch_cols[0].metric("Rows Fetched", f"{len(all_prices_df):,}")
        fetch_cols[1].metric("Stocks with Data", f"{all_prices_df['symbol'].nunique()}")
        fetch_cols[2].metric("Expected Stocks", f"{len(shortlist_tickers)}")

        results = []
        progress_bar = st.progress(0, text="Testing strategies...")
        grouped = {
            sym: _prepare_close(sym, grp.set_index('date')['close'])
            for sym, grp in all_prices_df.groupby('symbol')
        }
        market_breadth = sw.build_market_breadth(grouped)
        momentum_lookbacks = {
            int(combo[1]) for combo in param_combos
            if combo[0] == STRATEGY_CROSS_MOMENTUM
        }
        momentum_frames = sw.build_momentum_percentiles(grouped, momentum_lookbacks)
        benchmark_close = jkse_df['close'] if not jkse_df.empty else pd.Series(dtype=float)
        status_counts = {
            'ok': 0, 'no_signals': 0, 'missing_data': 0,
            'insufficient_history': 0, 'error': 0,
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=OPTIMIZER_WORKERS) as executor:
            futures = {
                executor.submit(
                    sw.evaluate_stock_strategies,
                    ticker,
                    grouped.get(ticker),
                    regime_series,
                    param_combos,
                    sw.holdout_split_date(opt_start, opt_end, opt_validation_pct),
                    float(opt_buy_fee),
                    float(opt_sell_fee),
                    float(opt_sell_tax),
                    benchmark_close,
                    {
                        lookback: frame[ticker]
                        for lookback, frame in momentum_frames.items()
                        if ticker in frame.columns
                    },
                    market_breadth,
                    float(opt_slippage_ticks),
                ): ticker
                for ticker in shortlist_tickers
            }
            for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                try:
                    stock_rows, status = future.result()
                    results.extend(stock_rows)
                    status_counts[status] += 1
                except Exception:
                    status_counts['error'] += 1
                processed = completed * len(param_combos)
                progress_bar.progress(
                    completed / len(futures),
                    text=f"Testing strategies... {processed:,}/{total_combos:,}",
                )

        progress_bar.empty()

        if results:
            results_df = pd.DataFrame(results)
            st.session_state['dt_optimizer_results'] = results_df
            st.session_state['dt_optimizer_results_cache_key'] = results_cache_key
            st.success(
                f"Strategy search complete: {len(param_combos):,} strategies tested "
                f"across {results_df['Symbol'].nunique()} stocks."
            )
        else:
            st.warning(
                "No usable backtests were produced. "
                f"No signals: {status_counts['no_signals']}; "
                f"insufficient history: {status_counts['insufficient_history']}; "
                f"missing data: {status_counts['missing_data']}; "
                f"worker errors: {status_counts['error']}."
            )


# ── Display Results ────────────────────────────────────────────────────────────
recommended_strategy = None
stock_strategy_map = {}
strategy_was_applied = False
results_are_current = (
    'dt_optimizer_results' in st.session_state
    and st.session_state.get('dt_optimizer_results_cache_key') == results_cache_key
)
if 'dt_optimizer_results' in st.session_state and not results_are_current:
    st.session_state.pop('dt_active_strategy', None)
    st.session_state.pop('dt_stock_strategies', None)
    st.session_state.pop('dt_applied_strategy_signature', None)
    st.session_state.pop('dt_single_strategy_signature', None)
    st.info("Search settings changed. Run **Find Best Strategy** to refresh the results.")

if results_are_current:
    results_df = st.session_state['dt_optimizer_results'].copy()
    strategy_df = sw.summarize_strategies(results_df)
    tested_stocks = max(1, results_df['Symbol'].nunique())
    min_validated_stocks = max(3, int(np.ceil(tested_stocks * 0.20)))
    min_validation_trades = max(20, tested_stocks * 2)
    qualified_df = strategy_df[
        (strategy_df['Stocks Validated'] >= min_validated_stocks)
        & (strategy_df['Validation Trades'] >= min_validation_trades)
        & strategy_df['Passes Gates']
    ].copy()
    gate_description = (
        f"positive after-cost average trade in both train and holdout, holdout profit "
        f"factor ≥ {MIN_PROFIT_FACTOR:.1f}, t-stat ≥ {MIN_T_STAT:.1f}, and at least "
        f"{min_validation_trades} holdout trades across {min_validated_stocks} stocks"
    )

    if qualified_df.empty:
        # Never promote a losing strategy just because it lost the least.
        st.session_state.pop('dt_active_strategy', None)
        st.session_state.pop('dt_applied_strategy_signature', None)
        st.warning(
            "No strategy passed the profitability gates "
            f"({gate_description}). Nothing was applied to the backtest or scanner. "
            "Review **All Strategies**, try a longer date range, or widen the search."
        )

    col_cfg = {
        "Symbol":        st.column_config.TextColumn("Symbol"),
        "Strategy":      st.column_config.TextColumn("Strategy"),
        "Parameters":    st.column_config.TextColumn("Strategy Parameters"),
        "Param A":       st.column_config.NumberColumn("Param A", format="%.2f"),
        "Param B":       st.column_config.NumberColumn("Param B", format="%.2f"),
        "Param C":       st.column_config.NumberColumn("Param C", format="%.2f"),
        "Param D":       st.column_config.NumberColumn("Param D", format="%.2f"),
        "Max Hold":      st.column_config.NumberColumn("Max Hold Days", format="%d"),
        "Profit Target": st.column_config.NumberColumn("Profit Target", format="%.1f%%"),
        "Stop Loss":     st.column_config.NumberColumn("Stop Loss", format="%.1f%%"),
        "Win Rate":      st.column_config.ProgressColumn("Win Rate", help="Fraction of profitable trades", format="%.1f%%", min_value=0, max_value=100),
        "Total Return":  st.column_config.NumberColumn("Total Return", format="%.2f%%"),
        "Avg Trade":     st.column_config.NumberColumn("Avg Trade", format="%.2f%%"),
        "Max Drawdown":  st.column_config.NumberColumn("Max Drawdown", format="%.2f%%"),
        "# Trades":      st.column_config.NumberColumn("# Trades", format="%d"),
        "Bear Entries":  st.column_config.NumberColumn("Bear Entries", format="%d"),
        "Bull Entries":  st.column_config.NumberColumn("Bull Entries", format="%d"),
        "Validation Return": st.column_config.NumberColumn("Validation Return", format="%.2f%%"),
        "Validation Win Rate": st.column_config.NumberColumn("Validation Win Rate", format="%.1f%%"),
        "Validation Drawdown": st.column_config.NumberColumn("Validation Drawdown", format="%.2f%%"),
        "Validation Trades": st.column_config.NumberColumn("Validation Trades", format="%d"),
        "Validation Avg Trade": st.column_config.NumberColumn("Validation Avg Trade", format="%.2f%%"),
        "Validation Profit Factor": st.column_config.NumberColumn("Profit Factor", help="Gross holdout gains / gross holdout losses", format="%.2f"),
        "Validation t-stat": st.column_config.NumberColumn("t-stat", help="Mean holdout trade divided by its standard error", format="%.2f"),
        "Passes Gates": st.column_config.CheckboxColumn("Passes Gates"),
        "Stocks Tested": st.column_config.NumberColumn("Stocks Tested", format="%d"),
        "Stocks Validated": st.column_config.NumberColumn("Stocks Validated", format="%d"),
        "Train Trades": st.column_config.NumberColumn("Train Trades", format="%d"),
        "Train Avg Trade": st.column_config.NumberColumn("Train Avg Trade", format="%.2f%%"),
        "Median Validation Return": st.column_config.NumberColumn("Median Validation Return", format="%.2f%%"),
        "Positive Stocks %": st.column_config.ProgressColumn("Positive Stocks", format="%.1f%%", min_value=0, max_value=100),
        "Validation Coverage %": st.column_config.ProgressColumn("Stock Coverage", format="%.1f%%", min_value=0, max_value=100),
        "Generalization Gap": st.column_config.NumberColumn("Validation - Train", format="%.2f%%"),
        "Strategy Score": st.column_config.ProgressColumn("Strategy Score", format="%.1f", min_value=0, max_value=100),
        "Stock Strategy Score": st.column_config.ProgressColumn("Stock Strategy Score", format="%.1f", min_value=0, max_value=100),
    }

    tab_strategy, tab_stocks, tab_all = st.tabs([
        "Recommended Strategy", "Stock Breakdown", "All Strategies",
    ])
    st.session_state['dt_stock_strategies'] = {}
    st.session_state['dt_stock_strategies_cache_key'] = results_cache_key

    if not qualified_df.empty:
        recommended = qualified_df.iloc[0]
        recommended_strategy = {
            'strategy_type': recommended['Strategy'],
            'param_a': float(recommended['Param A']),
            'param_b': float(recommended['Param B']),
            'param_c': float(recommended['Param C']),
            'param_d': float(recommended['Param D']),
            'max_hold': int(recommended['Max Hold']),
            'profit_target': float(recommended['Profit Target']),
            'stop_loss': float(recommended['Stop Loss']),
            'score': float(recommended['Strategy Score']),
            'validation_avg_trade': float(recommended['Validation Avg Trade']),
        }
        stock_strategy_map = sw.select_stock_strategies(results_df, recommended_strategy)
        st.session_state['dt_stock_strategies'] = stock_strategy_map
        st.session_state['dt_stock_strategies_cache_key'] = results_cache_key
        strategy_signature = (
            results_cache_key,
            recommended_strategy['strategy_type'],
            recommended_strategy['param_a'],
            recommended_strategy['param_b'],
            recommended_strategy['param_c'],
            recommended_strategy['param_d'],
            recommended_strategy['max_hold'],
            recommended_strategy['profit_target'],
            recommended_strategy['stop_loss'],
        )
        if st.session_state.get('dt_applied_strategy_signature') != strategy_signature:
            downstream_defaults = {
                'bt_strategy_type': recommended_strategy['strategy_type'],
                'bt_parameter_strategy': recommended_strategy['strategy_type'],
                'bt_param_a': recommended_strategy['param_a'],
                'bt_param_b': recommended_strategy['param_b'],
                'bt_param_c': recommended_strategy['param_c'],
                'bt_param_d': recommended_strategy['param_d'],
                'bt_max_hold': recommended_strategy['max_hold'],
                'bt_pt': recommended_strategy['profit_target'],
                'bt_sl': recommended_strategy['stop_loss'],
                'bt_buy_fee': float(opt_buy_fee),
                'bt_sell_fee': float(opt_sell_fee),
                'bt_sell_tax': float(opt_sell_tax),
                'bt_slippage': float(opt_slippage_ticks),
            }
            for key, value in downstream_defaults.items():
                st.session_state[key] = value
            st.session_state['dt_active_strategy'] = recommended_strategy
            st.session_state['dt_applied_strategy_signature'] = strategy_signature
            strategy_was_applied = True

        strategy_params = [
            'Strategy', 'Param A', 'Param B', 'Param C', 'Param D',
            'Max Hold', 'Profit Target', 'Stop Loss',
        ]
        recommended_runs = results_df.copy()
        for parameter in strategy_params:
            recommended_runs = recommended_runs[recommended_runs[parameter] == recommended[parameter]]

        with tab_strategy:
            st.caption(
                "Only strategies that pass the absolute profitability gates are eligible; "
                "among those, the recommendation ranks holdout quality, consistency, and "
                "drawdown control. It is not the largest in-sample winner."
            )
            if strategy_was_applied:
                st.success("Applied this strategy to the single-stock backtest and live scanner.")
            metric_cols = st.columns(5)
            metric_cols[0].metric("Strategy Score", f"{recommended['Strategy Score']:.1f}/100")
            metric_cols[1].metric("Validation Avg Trade", f"{recommended['Validation Avg Trade']:.2f}%")
            metric_cols[2].metric("Validation Win Rate", f"{recommended['Validation Win Rate']:.1f}%")
            metric_cols[3].metric("Positive Stocks", f"{recommended['Positive Stocks %']:.1f}%")
            metric_cols[4].metric("Validation Trades", f"{int(recommended['Validation Trades']):,}")
            st.caption(f"Stock-specific parameters available for {len(stock_strategy_map)} tickers.")
            st.dataframe(
                qualified_df.head(10), column_config=col_cfg,
                column_order=[
                    'Strategy Score', 'Strategy', 'Parameters',
                    'Max Hold', 'Profit Target', 'Stop Loss',
                    'Validation Avg Trade', 'Validation Profit Factor',
                    'Validation t-stat', 'Validation Win Rate',
                    'Median Validation Return', 'Positive Stocks %',
                    'Validation Coverage %', 'Validation Drawdown',
                    'Validation Trades', 'Stocks Validated',
                    'Generalization Gap',
                ],
                hide_index=True, width='stretch',
            )

        with tab_stocks:
            use_best_stock_strategy = st.checkbox(
                "Use best strategy for each stock",
                value=False,
                key="dt_use_best_stock_strategy_breakdown",
                help=(
                    "Choose each stock's highest-scoring configuration using holdout "
                    "average trade, win rate, return, trade count, and drawdown."
                ),
            )
            if use_best_stock_strategy:
                stock_breakdown_df = sw.best_stock_strategy_rows(results_df)
                st.caption(
                    "Each row uses that stock's highest-scoring strategy and parameter "
                    f"set on held-out data. Only configurations with at least "
                    f"{MIN_STOCK_VALIDATION_TRADES} holdout trades and a positive average "
                    "trade in both train and holdout are considered."
                )
                stock_breakdown_order = [
                    'Symbol', 'Stock Strategy Score', 'Strategy', 'Parameters',
                    'Max Hold', 'Profit Target', 'Stop Loss',
                    'Validation Avg Trade', 'Validation Win Rate',
                    'Validation Return', 'Validation Drawdown', 'Validation Trades',
                    'Avg Trade', 'Win Rate', '# Trades',
                ]
            else:
                stock_breakdown_df = recommended_runs.sort_values(
                    ['Validation Avg Trade', 'Validation Trades'], ascending=False,
                )
                st.caption("Per-stock results for the single recommended parameter set.")
                stock_breakdown_order = [
                    'Symbol', 'Validation Avg Trade', 'Validation Win Rate',
                    'Validation Return', 'Validation Drawdown', 'Validation Trades',
                    'Avg Trade', 'Win Rate', '# Trades',
                ]
            st.dataframe(
                stock_breakdown_df,
                column_config=col_cfg,
                column_order=stock_breakdown_order,
                hide_index=True, width='stretch',
            )

    else:
        with tab_strategy:
            st.info("No strategy passed the profitability gates, so none is recommended.")

    with tab_all:
        st.caption(f"To be recommended, a strategy needs {gate_description}.")
        st.dataframe(
            strategy_df, column_config=col_cfg,
            column_order=[
                'Passes Gates', 'Strategy Score', 'Strategy', 'Parameters',
                'Max Hold', 'Profit Target', 'Stop Loss',
                'Validation Avg Trade', 'Validation Profit Factor',
                'Validation t-stat', 'Validation Win Rate',
                'Median Validation Return', 'Positive Stocks %',
                'Validation Coverage %', 'Validation Drawdown',
                'Validation Trades', 'Stocks Validated',
                'Train Avg Trade', 'Train Trades', 'Generalization Gap',
            ],
            hide_index=True, width='stretch',
        )

    st.download_button(
        "Download Strategy Rankings CSV",
        data=df_to_csv(strategy_df),
        file_name="short_term_swing_strategy_rankings.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE STOCK BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Run a Short-Term Swing Backtest on a Specific Ticker", divider='red')

active_strategy = st.session_state.get('dt_active_strategy')
stock_strategy_map = st.session_state.get('dt_stock_strategies', {})

main_col1, main_col2 = st.columns([1, 2])
with main_col1:
    try:
        stock_list = sorted(pd.read_csv('data/jkse/valuation.csv')['stock'].unique().tolist())
    except Exception:
        stock_list = ['BBCA.JK']
    default_idx = stock_list.index('BBCA.JK') if 'BBCA.JK' in stock_list else 0
    stock = st.selectbox('Stock Ticker', options=stock_list, index=default_idx, key="dt_single_stock")
with main_col2:
    start_date = st.date_input('Start Date', value=datetime.date(2023, 1, 1), key="dt_single_start")
    end_date = st.date_input('End Date', value=datetime.date.today(), key="dt_single_end")

stock_specific_strategy = stock_strategy_map.get(stock)
single_strategy = stock_specific_strategy or active_strategy
if (
    'bt_pt' not in st.session_state
    or st.session_state.get('dt_profit_target_version') != STRATEGY_ENGINE_VERSION
):
    st.session_state['bt_pt'] = DEFAULT_PROFIT_TARGET
    st.session_state['dt_profit_target_version'] = STRATEGY_ENGINE_VERSION
st.session_state.setdefault('bt_max_hold', DEFAULT_MAX_HOLD)
st.session_state.setdefault('bt_sl', DEFAULT_STOP_LOSS)
st.session_state.setdefault('bt_buy_fee', 0.15)
st.session_state.setdefault('bt_sell_fee', 0.15)
st.session_state.setdefault('bt_sell_tax', 0.1)
st.session_state.setdefault('bt_slippage', DEFAULT_SLIPPAGE_TICKS)
if single_strategy:
    single_strategy_signature = (
        stock, st.session_state.get('dt_stock_strategies_cache_key'),
        single_strategy['strategy_type'], single_strategy['param_a'],
        single_strategy['param_b'], single_strategy['param_c'], single_strategy['param_d'],
        single_strategy['max_hold'], single_strategy['profit_target'], single_strategy['stop_loss'],
    )
    if st.session_state.get('dt_single_strategy_signature') != single_strategy_signature:
        for key, value in {
            'bt_strategy_type': single_strategy['strategy_type'],
            'bt_parameter_strategy': single_strategy['strategy_type'],
            'bt_param_a': single_strategy['param_a'],
            'bt_param_b': single_strategy['param_b'],
            'bt_param_c': single_strategy['param_c'],
            'bt_param_d': single_strategy['param_d'],
            'bt_max_hold': single_strategy['max_hold'],
            'bt_pt': single_strategy['profit_target'],
            'bt_sl': single_strategy['stop_loss'],
        }.items():
            st.session_state[key] = value
        st.session_state['dt_single_strategy_signature'] = single_strategy_signature

with st.expander("Strategy Parameters & Costs", expanded=False):
    if st.session_state.get('bt_strategy_type') not in STRATEGY_TYPES:
        st.session_state['bt_strategy_type'] = STRATEGY_REGIME_RSI
        st.session_state.pop('bt_parameter_strategy', None)
    bt_strategy_type = st.selectbox("Entry Strategy", STRATEGY_TYPES, key="bt_strategy_type")
    if st.session_state.get('bt_parameter_strategy') != bt_strategy_type:
        defaults = SEARCH_PRESETS['Quick']['strategy_params'][bt_strategy_type][0]
        for key, value in zip(('bt_param_a', 'bt_param_b', 'bt_param_c', 'bt_param_d'), defaults):
            st.session_state[key] = value
        st.session_state['bt_parameter_strategy'] = bt_strategy_type

    parameter_labels = STRATEGY_PARAMETER_LABELS[bt_strategy_type]
    parameter_inputs = STRATEGY_PARAMETER_INPUTS[bt_strategy_type]
    parameter_cols = st.columns(4)
    bt_params = [
        column.number_input(
            label, min_value=bounds[0], max_value=bounds[1], step=bounds[2], key=f'bt_param_{letter}',
        )
        for column, label, bounds, letter in zip(
            parameter_cols, parameter_labels, parameter_inputs, ('a', 'b', 'c', 'd'),
        )
    ]
    bt_param_a, bt_param_b, bt_param_c, bt_param_d = bt_params

    if single_strategy:
        parameter_source = "stock-specific holdout result" if stock_specific_strategy else "global recommendation"
        st.info(
            f"Using the {parameter_source} for **{stock}**: {single_strategy['strategy_type']} "
            f"({sw.format_strategy_params(single_strategy['strategy_type'], single_strategy['param_a'], single_strategy['param_b'], single_strategy['param_c'], single_strategy['param_d'])}); "
            f"target {single_strategy['profit_target']:.1f}%, stop {single_strategy['stop_loss']:.1f}%, "
            f"hold up to {single_strategy['max_hold']} sessions."
        )

    st.markdown("#### Exit Rules & Costs")
    exit_cols = st.columns(3)
    bt_max_hold = exit_cols[0].number_input(
        "Max Hold Sessions", min_value=1, max_value=20, step=1, key="bt_max_hold",
    )
    bt_profit_target = exit_cols[1].number_input(
        "Profit Target %", min_value=0.0, max_value=20.0,
        step=0.5, key="bt_pt",
    )
    bt_stop_loss = exit_cols[2].number_input(
        "Stop Loss %", min_value=0.0, max_value=20.0, step=0.5, key="bt_sl",
    )
    cost_cols = st.columns(4)
    bt_buy_fee = cost_cols[0].number_input(
        "Buy Fee %", min_value=0.0, max_value=5.0, step=0.01, key="bt_buy_fee",
    )
    bt_sell_fee = cost_cols[1].number_input(
        "Sell Fee %", min_value=0.0, max_value=5.0, step=0.01, key="bt_sell_fee",
    )
    bt_sell_tax = cost_cols[2].number_input(
        "Sell Tax %", min_value=0.0, max_value=5.0, step=0.01, key="bt_sell_tax",
    )
    bt_slippage = cost_cols[3].number_input(
        "Slippage (ticks/side)", min_value=0.0, max_value=5.0, step=0.5, key="bt_slippage",
    )

st.divider()

if st.button('Run Short-Term Swing Backtest', key="dt_run_single") and stock:
    regime_series = sw.compute_regime_series(jkse_df, fast_sma=int(regime_fast), slow_sma=int(regime_slow))
    with st.spinner("Loading price history..."):
        try:
            price_df = fetch_close_history((stock,), start_date.strftime('%Y-%m-%d'))[['date', 'close']]
        except Exception as e:
            st.error(f"Error loading price history: {e}")
            price_df = pd.DataFrame()

    if price_df.empty:
        st.error("No historical data found for the given stock and timeframe.")
    else:
        price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
        price_df['close'] = pd.to_numeric(price_df['close'], errors='coerce')
        close = price_df.dropna().drop_duplicates('date', keep='last').sort_values('date').set_index('date')['close']
        close = _prepare_close(stock, close[close.index <= pd.to_datetime(end_date)])
        cached_prices = st.session_state.get('dt_prices', pd.DataFrame())
        momentum_percentile = None
        single_breadth = None
        if bt_strategy_type == STRATEGY_CROSS_MOMENTUM and not cached_prices.empty:
            grouped_single = {
                symbol: group.assign(date=pd.to_datetime(group['date'])).set_index('date')['close']
                for symbol, group in cached_prices.groupby('symbol')
            }
            frame = sw.build_momentum_percentiles(grouped_single, {int(bt_param_a)}).get(int(bt_param_a))
            if frame is not None and stock in frame:
                momentum_percentile = frame[stock]
        if bt_strategy_type == STRATEGY_BREADTH_RSI2:
            # Breadth needs the whole universe: prefer the optimizer's fetched
            # prices, otherwise load the shortlist's stored history.
            universe_prices = cached_prices
            if universe_prices.empty and shortlist_tickers:
                universe_prices = load_close_history(
                    tuple(sorted(shortlist_tickers)), start_date.strftime('%Y-%m-%d'),
                )
            if not universe_prices.empty:
                single_breadth = sw.build_market_breadth({
                    symbol: group.assign(date=pd.to_datetime(group['date'])).set_index('date')['close']
                    for symbol, group in universe_prices.groupby('symbol')
                })

        signals = sw.build_regime_signals(
            close, regime_series, bt_strategy_type, bt_param_a, bt_param_b,
            bt_param_c, bt_param_d, bt_max_hold, bt_profit_target, bt_stop_loss,
            benchmark_close=jkse_df['close'] if not jkse_df.empty else None,
            momentum_percentile=momentum_percentile,
            breadth=single_breadth,
        )
        if signals is None:
            st.warning("Not enough data to calculate the configured indicators.")
        else:
            close, entries, exits, regime_aligned = signals
            if not entries.any():
                detail = ""
                if bt_strategy_type == STRATEGY_CROSS_MOMENTUM and momentum_percentile is None:
                    detail = " Run the optimizer first to populate universe ranks."
                elif bt_strategy_type == STRATEGY_BREADTH_RSI2 and (single_breadth is None or single_breadth.empty):
                    detail = " Market breadth is unavailable; run the optimizer first to load universe prices."
                st.warning(f"No entry signals generated. Try loosening thresholds.{detail}")
            else:
                fees = sw.build_trade_fees(
                    close, entries, exits, bt_buy_fee, bt_sell_fee, bt_sell_tax, bt_slippage,
                )
                vbt = sw.load_vectorbt()
                pf = vbt.Portfolio.from_signals(close, entries, exits, freq='1D', fees=fees)
                st.subheader('Performance Outline')
                metrics = st.columns(4)
                tot_ret = pf.total_return() * 100 if pf.total_return() is not None else 0.0
                bh_ret = ((close.iloc[-1] / close.iloc[0]) - 1) * 100
                win_rate = pf.trades.win_rate() * 100 if len(pf.trades) else 0.0
                max_dd = pf.max_drawdown() * 100 if pf.max_drawdown() is not None else 0.0
                avg_trade = pf.trades.returns.mean() * 100 if len(pf.trades) else 0.0
                metrics[0].metric("Total Return", f"{tot_ret:.2f}%", delta=f"vs B&H {bh_ret:.2f}%")
                metrics[1].metric("Win Rate", f"{win_rate:.2f}%")
                metrics[2].metric("Avg Trade", f"{avg_trade:.3f}%")
                metrics[3].metric("Max Drawdown", f"{max_dd:.2f}%")
                bear_n = int((entries & (regime_aligned == 'bear')).sum())
                bull_n = int((entries & (regime_aligned == 'bull')).sum())
                st.caption(f"Bear entries: {bear_n} | Bull entries: {bull_n} | Total trades: {len(pf.trades)}")
                st.plotly_chart(
                    hp.style_plotly_chart(pf.plot()),
                    width='stretch',
                    theme=None,
                )
                with st.expander('View Trade Log'):
                    st.dataframe(pf.trades.records_readable)


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE SHORT-TERM SWING SETUP SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Live Short-Term Swing Setup Scanner", divider='orange')
st.caption(
    "Evaluate every supported strategy for every stock in today's shortlist. "
    "A setup confirmed at today's close is modeled as an entry at the next session's close; "
    "during market hours it remains provisional."
)

curr_regime = sw.get_current_regime(full_regime_series)
if curr_regime:
    st.info(f"Current JKSE regime: **{curr_regime.upper()}**. Strategies apply their own regime filters where relevant.")

scanner_cols = st.columns(2)
alert_lookback = scanner_cols[0].number_input("Calendar Days of History", min_value=90, max_value=730, value=365, step=30, key="dt_alert_lookback")
alert_watch_band = scanner_cols[1].number_input("Watch Proximity", min_value=1.0, max_value=20.0, value=5.0, step=1.0, key="dt_alert_watch")

if stock_strategy_map:
    st.caption(
        f"All {len(STRATEGY_TYPES)} strategies are evaluated. Stock-specific optimized parameters apply to "
        f"{len(stock_strategy_map)} tickers; other strategies use the global recommendation or baseline presets."
    )
elif active_strategy:
    st.caption(f"All {len(STRATEGY_TYPES)} strategies are evaluated using the global recommendation where applicable and baseline presets otherwise.")
else:
    st.caption(f"All {len(STRATEGY_TYPES)} strategies are evaluated with baseline presets. Run the optimizer to apply validated parameters.")
st.caption(
    "Setups from optimized parameters that passed the holdout gates are listed separately "
    "from baseline presets, which have not been validated and are for research only."
)

scan_button = st.button("Scan Today's Closing Setups", type="primary", disabled=len(shortlist_tickers) == 0)



if scan_button:
    scanner_plans = {
        ticker: sw.scanner_strategy_plan(stock_strategy_map.get(ticker), active_strategy)
        for ticker in shortlist_tickers
    }
    strategy_params_for_warmup = [
        (item['strategy_type'], *item['params'])
        for plan in scanner_plans.values()
        for item in plan
    ]
    required_rows = max(
        sw.strategy_warmup(strategy_type, *params)
        for strategy_type, *params in strategy_params_for_warmup
    )
    alert_start = (datetime.date.today() - datetime.timedelta(days=max(int(alert_lookback), required_rows * 2))).strftime('%Y-%m-%d')

    try:
        cp_df = hd.get_company_profile(shortlist_tickers)
    except Exception as e:
        st.warning(f"Could not fetch current profile prices: {e}")
        cp_df = pd.DataFrame()

    cached_prices = st.session_state.get('dt_prices', pd.DataFrame())
    grouped_alert = {}
    if not cached_prices.empty:
        cached_prices = cached_prices.copy()
        cached_prices['date'] = pd.to_datetime(cached_prices['date'], errors='coerce')
        grouped_alert = {
            symbol: group.dropna(subset=['date']).set_index('date')['close'].sort_index()
            for symbol, group in cached_prices.groupby('symbol')
        }

    def _series_is_current(series: pd.Series) -> bool:
        return len(series) >= required_rows and series.index[-1].date() >= datetime.date.today() - datetime.timedelta(days=7)

    scan_progress = st.progress(0, text="Loading current daily histories...")
    stored_prices = fetch_close_history(tuple(sorted(shortlist_tickers)), alert_start)
    grouped_stored = {
        symbol: group.set_index('date')['close'] for symbol, group in stored_prices.groupby('symbol')
    }
    current_series = {}
    price_sources = {}
    stale_skipped = 0
    for index, ticker in enumerate(shortlist_tickers, start=1):
        scan_progress.progress(index / len(shortlist_tickers), text=f"Loading {ticker}...")
        series = sw.clean_price_series(grouped_alert.get(ticker))
        source = "Optimizer cache"
        if not _series_is_current(series):
            if ticker in grouped_stored:
                series = sw.clean_price_series(grouped_stored[ticker])
            source = "Stored history"
        if not _series_is_current(series):
            live_history = load_live_daily_prices(ticker, alert_start)
            if not live_history.empty:
                series = sw.clean_price_series(live_history.set_index('date')['close'])
                source = "Live daily API"
        if not _series_is_current(series):
            stale_skipped += 1
            continue
        series = _prepare_close(ticker, series)
        if ticker in cp_df.index:
            live_price = pd.to_numeric(cp_df.loc[ticker, 'price'], errors='coerce')
            if pd.notna(live_price) and live_price > 0 and 0.65 <= live_price / float(series.iloc[-1]) <= 1.35:
                series.loc[pd.Timestamp(datetime.date.today()).normalize()] = float(live_price)
                series = series.sort_index()
        current_series[ticker] = series
        price_sources[ticker] = source

    momentum_lookbacks = {
        int(item['params'][0])
        for plan in scanner_plans.values()
        for item in plan
        if item['strategy_type'] == STRATEGY_CROSS_MOMENTUM
    }
    momentum_frames = sw.build_momentum_percentiles(current_series, momentum_lookbacks)
    scan_breadth = sw.build_market_breadth(current_series)
    benchmark_close = jkse_df['close'] if not jkse_df.empty else None
    scan_rows = []
    for index, (ticker, series) in enumerate(current_series.items(), start=1):
        scan_progress.progress(index / len(current_series), text=f"Evaluating {ticker}...")
        aligned_regime = full_regime_series.reindex(series.index, method='ffill').fillna('bear') if not full_regime_series.empty else pd.Series('bear', index=series.index)
        for item in scanner_plans[ticker]:
            strategy_type = item['strategy_type']
            params = item['params']
            momentum_percentile = None
            frame = momentum_frames.get(int(params[0]))
            if strategy_type == STRATEGY_CROSS_MOMENTUM and frame is not None and ticker in frame:
                momentum_percentile = frame[ticker]
            setup, _, indicator_series = sw.strategy_components(
                series, aligned_regime, benchmark_close, momentum_percentile,
                strategy_type, *params, breadth=scan_breadth,
            )
            indicator = float(indicator_series.iloc[-1]) if pd.notna(indicator_series.iloc[-1]) else np.nan
            fires = bool(setup.iloc[-1])
            trend_window = {
                STRATEGY_REGIME_RSI: int(params[2]),
                STRATEGY_BREADTH_RSI2: int(params[3]),
            }.get(strategy_type, 0)
            trend_ok = True
            if trend_window > 0:
                trend_value = series.rolling(trend_window, min_periods=trend_window).mean().iloc[-1]
                trend_ok = bool(pd.notna(trend_value) and series.iloc[-1] > trend_value)
            scan_rows.append({
                'Symbol': ticker,
                'Current Price': float(series.iloc[-1]),
                'History Through': series.index[-1].date(),
                'Price Source': price_sources[ticker],
                'Parameter Source': item['parameter_source'],
                'Strategy': strategy_type,
                'Stock Trend': (
                    "Bullish" if indicator >= 0 else "Bearish"
                ) if strategy_type == STRATEGY_SUPERTREND_PULLBACK and np.isfinite(indicator) else "",
                'Parameters': sw.format_strategy_params(strategy_type, *params),
                'Indicator': indicator,
                'Validated': item['validated'],
                'Expected Trade': item['expected_trade'],
                'Strategy Score': item['score'],
                'Return Uplift': item['return_uplift'],
                'Profit Target': item['profit_target'],
                'Stop Loss': item['stop_loss'],
                'Max Hold': item['max_hold'],
                'Setup Fires': fires,
                'Near Setup': sw.is_near_setup(
                    strategy_type, indicator, params, float(alert_watch_band), trend_ok,
                    regime=curr_regime,
                ),
            })
    scan_progress.empty()

    if stale_skipped:
        st.warning(f"Skipped {stale_skipped} stock(s) without sufficiently recent daily history.")
    if not scan_rows:
        st.warning("Could not evaluate any stock. Increase the history window or verify the price sources.")
    else:
        alert_df = pd.DataFrame(scan_rows)
        alert_df['Signal'] = np.select(
            [alert_df['Setup Fires'], alert_df['Near Setup']],
            ["Setup Confirmed", "Watch"],
            default="Neutral",
        )
        # Rank by expected value per trade (holdout average), then strategy quality.
        alert_df = alert_df.sort_values(
            ['Validated', 'Expected Trade', 'Strategy Score', 'Indicator'],
            ascending=[False, False, False, False], na_position='last',
        )
        confirmed_df = alert_df[alert_df['Signal'] == "Setup Confirmed"]
        validated_confirmed_df = confirmed_df[confirmed_df['Validated']]
        research_confirmed_df = confirmed_df[~confirmed_df['Validated']]
        watch_df = alert_df[alert_df['Signal'] == "Watch"]
        neutral_df = alert_df[alert_df['Signal'] == "Neutral"]
        summary_cols = st.columns(4)
        summary_cols[0].metric("Validated Setups", len(validated_confirmed_df), help="Confirmed setups from optimized parameters that passed the holdout gates")
        summary_cols[1].metric("Unvalidated Setups", len(research_confirmed_df), help="Confirmed setups from baseline presets")
        summary_cols[2].metric("Watch", len(watch_df))
        summary_cols[3].metric("Strategy Checks", len(alert_df), help=f"{len(current_series)} stocks x {len(STRATEGY_TYPES)} strategies")
        if not scan_breadth.empty:
            st.caption(f"Market breadth: {scan_breadth.iloc[-1]:.1f}% of the shortlist closed above its 20-day average.")
        if not validated_confirmed_df.empty:
            st.success(f"{len(validated_confirmed_df)} validated setup(s) confirmed at the latest close for next-session execution research.")

        scanner_cfg = {
            "Current Price": st.column_config.NumberColumn("Price", format="%.0f"),
            "History Through": st.column_config.DateColumn("History Through", format="YYYY-MM-DD"),
            "Indicator": st.column_config.NumberColumn("Current Indicator", format="%.2f"),
            "Validated": st.column_config.CheckboxColumn("Validated"),
            "Expected Trade": st.column_config.NumberColumn("Holdout Avg Trade", help="Average after-cost holdout trade for these parameters", format="%.2f%%"),
            "Strategy Score": st.column_config.ProgressColumn("Strategy Score", format="%.1f", min_value=0, max_value=100),
            "Return Uplift": st.column_config.NumberColumn("Return vs Global", format="%+.2f%%"),
            "Profit Target": st.column_config.NumberColumn("Target", format="%.1f%%"),
            "Stop Loss": st.column_config.NumberColumn("Stop", format="%.1f%%"),
            "Max Hold": st.column_config.NumberColumn("Max Hold", format="%d sessions"),
        }
        if not validated_confirmed_df.empty:
            st.markdown("#### Validated Setups Confirmed at Latest Close")
            st.dataframe(validated_confirmed_df, column_config=scanner_cfg, hide_index=True, width='stretch')
        if not research_confirmed_df.empty:
            with st.expander(f"Unvalidated baseline setups ({len(research_confirmed_df)}) — research only"):
                st.dataframe(research_confirmed_df, column_config=scanner_cfg, hide_index=True, width='stretch')
        if not watch_df.empty:
            st.markdown("#### Approaching Setup")
            st.dataframe(watch_df, column_config=scanner_cfg, hide_index=True, width='stretch')
        with st.expander(f"Neutral stocks ({len(neutral_df)})"):
            st.dataframe(neutral_df, column_config=scanner_cfg, hide_index=True, width='stretch')
