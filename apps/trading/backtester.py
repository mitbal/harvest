import itertools
import hashlib
import streamlit as st
import datetime
import pandas as pd
import numpy as np
import vectorbt as vbt
from st_supabase_connection import SupabaseConnection

import harvest.data as hd
import harvest.plot as hp

st.title('Strategy Backtester')
st.sidebar.markdown(
    """
    Explore specific trading strategies across different time horizons.
    Choose a strategy, enter a ticker from the exchange, and set the timeframe
    to view cumulative performance, win margins, and recent trade logs.
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

# --- Top Ranked Valuation Shortlist ---
with st.container():
    st.subheader("Top Ranked Shortlist", divider='blue')
    filter_cols = st.columns(4)
    min_mcap_trill = filter_cols[0].number_input("Min Market Cap (Trillion IDR)", value=1.0, step=1.0)
    min_pe = filter_cols[1].number_input("Minimum PE", value=5.0, step=1.0)
    max_pe = filter_cols[2].number_input("Maximum PE", value=50.0, step=1.0)
    max_stocks = filter_cols[3].number_input("Max Stocks Output", min_value=10, max_value=200, value=50, step=10)
    include_big_banks = st.checkbox("Pin Big 4 Banks (BBCA, BMRI, BBRI, BBNI)", value=True, help="Always show Big 4 banks at the top of the list, regardless of filters")

    final_df = pd.DataFrame()
    try:
        val_df = pd.read_csv('data/jkse/valuation.csv')
        
        try:
            cp_df = pd.read_csv('data/jkse/company_profiles.csv')
            val_df = val_df.merge(cp_df[['symbol', 'mktCap']], left_on='stock', right_on='symbol', how='inner')
        except Exception:
            pass
        
        # 1. Calculate avg_pe and Discount for all stocks before filtering
        if 'last_1y_mean' in val_df.columns and 'last_5y_mean' in val_df.columns:
            val_df['avg_pe'] = val_df[['last_1y_mean', 'last_3y_mean', 'last_5y_mean']].mean(axis=1)
        else:
            val_df['avg_pe'] = val_df[['last_2y_mean', 'last_3y_mean', 'last_10y_mean']].mean(axis=1)
            
        val_df['Discount'] = (1 - (val_df['current_pe'] / val_df['avg_pe'])) * 100
        
        # 2. Apply dynamic filters to create the base shortlist
        mask = pd.Series(True, index=val_df.index)
        if 'mktCap' in val_df.columns:
            mask &= (val_df['mktCap'] >= (min_mcap_trill * 1_000_000_000_000))
        
        mask &= (val_df['current_pe'] >= min_pe) & (val_df['current_pe'] <= max_pe)
        mask &= (val_df['avg_pe'] > 0) & (val_df['avg_pe'] < 100)
        
        filtered_df = val_df[mask].copy().sort_values(by='Discount', ascending=False)
        
        # 3. Handle Big Banks pinning & trim to max_stocks
        if include_big_banks:
            big_banks = ['BBCA.JK', 'BMRI.JK', 'BBRI.JK', 'BBNI.JK']
            banks_df = val_df[val_df['stock'].isin(big_banks)].copy()
            banks_df = banks_df.sort_values(by='Discount', ascending=False)
            filtered_df = filtered_df[~filtered_df['stock'].isin(big_banks)]
            final_df = pd.concat([banks_df, filtered_df]).head(int(max_stocks))
        else:
            final_df = filtered_df.head(int(max_stocks))
            
        top_stocks = final_df[['stock', 'current_pe', 'avg_pe', 'Discount']].rename(
            columns={
                'stock': 'Symbol',
                'current_pe': 'Current PE',
                'avg_pe': 'Historical Avg PE',
            }
        )
        
        st.dataframe(
            top_stocks,
            column_config={
                "Discount": st.column_config.NumberColumn(
                    "Discount vs Avg PE",
                    format="%.2f%%",
                    help="Calculated as (1 - (Current PE / Historical Average PE)) * 100"
                ),
                "Current PE": st.column_config.NumberColumn("Current PE", format="%.2f"),
                "Historical Avg PE": st.column_config.NumberColumn("Historical Avg PE", format="%.2f")
            },
            hide_index=True,
            use_container_width=True
        )
    except Exception as e:
        st.warning(f"Could not load valuation data: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# RSI PARAMETER GRID SEARCH OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

def _run_rsi_backtest(price_series: pd.Series, period: int, buy_thresh: int, sell_thresh: int, max_hold: int):
    """Run RSI day-trading backtest and return (total_return, win_rate, max_drawdown, num_trades)."""
    if len(price_series) < period + 5:
        return None

    # Wilder's RSI calculation (matches TradingView RMA)
    # Using alpha=1/period and adjust=False in ewm replicates Wilder's recursive smoothing
    delta = price_series.diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    ema_up   = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    ema_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs  = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))

    raw_entries = (rsi < buy_thresh).to_numpy()
    raw_exits   = (rsi > sell_thresh).to_numpy()

    clean_entries = np.zeros_like(raw_entries, dtype=bool)
    clean_exits   = np.zeros_like(raw_exits, dtype=bool)

    in_pos = False
    days_held = 0
    for i in range(len(raw_entries)):
        if not in_pos:
            if raw_entries[i]:
                clean_entries[i] = True
                in_pos = True
                days_held = 0
        else:
            days_held += 1
            if raw_exits[i] or days_held >= max_hold:
                clean_exits[i] = True
                in_pos = False

    entries = pd.Series(clean_entries, index=price_series.index)
    exits   = pd.Series(clean_exits, index=price_series.index)

    try:
        pf = vbt.Portfolio.from_signals(price_series, entries, exits, freq='1D')
        num_trades = len(pf.trades)
        if num_trades == 0:
            return None
        tot_ret  = float(pf.total_return() * 100)
        win_rate = float(pf.trades.win_rate() * 100)
        max_dd   = float(pf.max_drawdown() * 100)
        return tot_ret, win_rate, max_dd, num_trades
    except Exception:
        return None


st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🔍 RSI Grid Search Optimizer", divider='violet')
st.caption(
    "Sweep multiple RSI parameter combinations across the entire shortlist. "
    "All price data is fetched in a single batch; backtesting runs fully in-memory."
)

with st.expander("⚙️ Configure Grid Parameters", expanded=True):
    g_col1, g_col2, g_col3, g_col4 = st.columns(4)

    with g_col1:
        st.markdown("**RSI Period**")
        rsi_period_min  = st.number_input("Period Min",  min_value=3,  max_value=50, value=7,  step=1,  key="opt_period_min")
        rsi_period_max  = st.number_input("Period Max",  min_value=3,  max_value=50, value=21, step=1,  key="opt_period_max")
        rsi_period_step = st.number_input("Period Step", min_value=1,  max_value=20, value=7,  step=1,  key="opt_period_step")

    with g_col2:
        st.markdown("**Buy if RSI <**")
        buy_min  = st.number_input("Buy Min",  min_value=5,  max_value=49, value=20, step=1, key="opt_buy_min")
        buy_max  = st.number_input("Buy Max",  min_value=5,  max_value=49, value=35, step=1, key="opt_buy_max")
        buy_step = st.number_input("Buy Step", min_value=1,  max_value=20, value=5,  step=1, key="opt_buy_step")

    with g_col3:
        st.markdown("**Sell if RSI >**")
        sell_min  = st.number_input("Sell Min",  min_value=51, max_value=95, value=60, step=1, key="opt_sell_min")
        sell_max  = st.number_input("Sell Max",  min_value=51, max_value=95, value=80, step=1, key="opt_sell_max")
        sell_step = st.number_input("Sell Step", min_value=1,  max_value=20, value=5,  step=1, key="opt_sell_step")

    with g_col4:
        st.markdown("**Other Settings**")
        opt_max_hold = st.number_input("Max Hold Days", min_value=1, max_value=20, value=3, step=1, key="opt_max_hold")
        opt_rank_by  = st.selectbox(
            "Rank By",
            options=["Win Rate", "Total Return", "Least Drawdown"],
            key="opt_rank_by"
        )
        opt_top_n = st.number_input("Show Top N Rows", min_value=5, max_value=300, value=30, step=5, key="opt_top_n")

    g_date_col1, g_date_col2 = st.columns(2)
    opt_start = g_date_col1.date_input("Start Date", value=datetime.date(2023, 1, 1), key="opt_start")
    opt_end   = g_date_col2.date_input("End Date",   value=datetime.date.today(),     key="opt_end")

# Build the parameter grid and compute size
periods        = list(range(int(rsi_period_min), int(rsi_period_max) + 1, int(rsi_period_step)))
buy_thresholds = list(range(int(buy_min),  int(buy_max)  + 1, int(buy_step)))
sell_thresholds= list(range(int(sell_min), int(sell_max) + 1, int(sell_step)))
param_combos   = list(itertools.product(periods, buy_thresholds, sell_thresholds))

shortlist_tickers = final_df['stock'].tolist() if not final_df.empty else []
total_combos = len(param_combos) * len(shortlist_tickers)

# Info line
info_cols = st.columns(4)
info_cols[0].metric("RSI Periods",    f"{periods}")
info_cols[1].metric("Buy Thresholds", f"{buy_thresholds}")
info_cols[2].metric("Sell Thresholds",f"{sell_thresholds}")
info_cols[3].metric("Total Runs",     f"{total_combos:,}")

if total_combos > 1000:
    st.warning(
        f"⚠️ **Large grid**: {total_combos:,} combinations across {len(shortlist_tickers)} stocks "
        "may take several minutes. Consider narrowing the ranges."
    )

run_optimizer = st.button("🚀 Run RSI Grid Search", type="primary", disabled=(total_combos == 0 or len(shortlist_tickers) == 0))

# Two separate cache keys:
# - price_cache_key: depends only on (tickers, dates) — changing RSI params won't re-fetch
# - results_cache_key: depends on everything — changing any param clears results
price_cache_key = hashlib.md5(
    f"{sorted(shortlist_tickers)}{opt_start}{opt_end}".encode()
).hexdigest()
results_cache_key = hashlib.md5(
    f"{periods}{buy_thresholds}{sell_thresholds}{opt_max_hold}{opt_start}{opt_end}{shortlist_tickers}".encode()
).hexdigest()

if run_optimizer:
    st.session_state.pop('optimizer_results', None)
    st.session_state.pop('optimizer_results_cache_key', None)

    # --- Price data: reuse cached copy if stocks & dates haven't changed ---
    if st.session_state.get('optimizer_price_cache_key') == price_cache_key and \
       'optimizer_prices' in st.session_state:
        all_prices_df = st.session_state['optimizer_prices']
        st.info(f"💾 Using cached price data ({len(all_prices_df):,} rows). Only RSI params changed.")
    else:
        fetch_progress = st.progress(0, text="Fetching price data from Supabase…")
        all_rows = []
        try:
            start_str = opt_start.strftime('%Y-%m-%d')

            # Supabase has a hard server-side row cap (~1000 rows per request).
            # Fetching each ticker individually guarantees we get all rows
            # (each stock has ~500 rows, well within the per-request limit).
            for i, ticker in enumerate(shortlist_tickers):
                fetch_progress.progress(
                    (i + 1) / len(shortlist_tickers),
                    text=f"Fetching {ticker}  ({i+1}/{len(shortlist_tickers)})…"
                )
                try:
                    res = (
                        conn.table("historical_prices")
                            .select("symbol,date,close")
                            .eq("symbol", ticker)
                            .gte("date", start_str)
                            .execute()
                    )
                    all_rows.extend(res.data or [])
                except Exception:
                    pass  # skip failed tickers silently

            fetch_progress.empty()

            all_prices_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
            if not all_prices_df.empty:
                all_prices_df['date']  = pd.to_datetime(all_prices_df['date'])
                all_prices_df['close'] = pd.to_numeric(all_prices_df['close'], errors='coerce')
                all_prices_df = all_prices_df[all_prices_df['date'] <= pd.to_datetime(opt_end)]
                all_prices_df = all_prices_df.sort_values(['symbol', 'date'])

            # Cache for reuse when only RSI params change
            st.session_state['optimizer_prices'] = all_prices_df
            st.session_state['optimizer_price_cache_key'] = price_cache_key

        except Exception as e:
            fetch_progress.empty()
            st.error(f"Failed to fetch price data: {e}")
            all_prices_df = pd.DataFrame()



    if all_prices_df.empty:
        st.error("No price data returned. Make sure the shortlist is non-empty and Supabase is reachable.")
    else:
        # Show fetch diagnostics so we can verify all stocks were retrieved
        fetch_cols = st.columns(3)
        fetch_cols[0].metric("Rows Fetched", f"{len(all_prices_df):,}")
        fetch_cols[1].metric("Stocks with Data", f"{all_prices_df['symbol'].nunique()}")
        fetch_cols[2].metric("Expected Stocks", f"{len(shortlist_tickers)}")

        results = []
        progress_bar  = st.progress(0, text="Running grid search…")
        status_text   = st.empty()
        processed     = 0

        grouped = {sym: grp.set_index('date')['close'] for sym, grp in all_prices_df.groupby('symbol')}

        for ticker in shortlist_tickers:
            if ticker not in grouped:
                processed += len(param_combos)
                continue
            price_series = grouped[ticker]

            for (period, buy_thresh, sell_thresh) in param_combos:
                # skip invalid combos
                if buy_thresh >= sell_thresh:
                    processed += 1
                    continue

                result = _run_rsi_backtest(price_series, period, buy_thresh, sell_thresh, int(opt_max_hold))

                if result is not None:
                    tot_ret, win_rate, max_dd, num_trades = result

                    results.append({
                        'Symbol':        ticker,
                        'RSI Period':    period,
                        'Buy <':         buy_thresh,
                        'Sell >':        sell_thresh,
                        'Total Return':  tot_ret,
                        'Win Rate':      win_rate,
                        'Max Drawdown':  max_dd,
                        '# Trades':      num_trades,
                    })

                processed += 1
                pct = processed / total_combos
                progress_bar.progress(pct, text=f"Running grid search… {processed:,}/{total_combos:,}")

        progress_bar.empty()
        status_text.empty()

        if results:
            results_df = pd.DataFrame(results)
            st.session_state['optimizer_results']   = results_df
            st.session_state['optimizer_results_cache_key'] = results_cache_key
            st.success(f"✅ Grid search complete — {len(results_df):,} valid combinations found across {results_df['Symbol'].nunique()} stocks.")
        else:
            st.warning("No trades were generated for any combination. Try loosening the RSI thresholds or broadening the date range.")


# ── Display Results ────────────────────────────────────────────────────────────
if 'optimizer_results' in st.session_state:
    results_df = st.session_state['optimizer_results'].copy()

    # Filter rows with at least 1 trade (exclude combos that never triggered)
    results_df = results_df[results_df['# Trades'] >= 1]

    # Sorting
    rank_by = st.session_state.get('opt_rank_by', 'Win Rate')
    if rank_by == 'Win Rate':
        sort_col, ascending = 'Win Rate', False
    elif rank_by == 'Total Return':
        sort_col, ascending = 'Total Return', False
    else:  # Least Drawdown
        sort_col, ascending = 'Max Drawdown', True

    top_n = int(st.session_state.get('opt_top_n', 30))

    tab_best, tab_all = st.tabs(["🥇 Best Per Stock", "📋 All Combinations"])

    with tab_best:
        st.caption("For each stock, only the single best-performing parameter combination is shown.")
        if rank_by == 'Least Drawdown':
            best_df = results_df.loc[results_df.groupby('Symbol')['Max Drawdown'].idxmin()].copy()
        elif rank_by == 'Total Return':
            best_df = results_df.loc[results_df.groupby('Symbol')['Total Return'].idxmax()].copy()
        else:
            best_df = results_df.loc[results_df.groupby('Symbol')['Win Rate'].idxmax()].copy()

        best_df = best_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
        best_df.index += 1
        best_df.index.name = 'Rank'

        st.dataframe(
            best_df.head(top_n),
            column_config={
                "Symbol":       st.column_config.TextColumn("Symbol"),
                "RSI Period":   st.column_config.NumberColumn("Period", format="%d"),
                "Buy <":        st.column_config.NumberColumn("Buy RSI <", format="%d"),
                "Sell >":       st.column_config.NumberColumn("Sell RSI >", format="%d"),
                "Win Rate":     st.column_config.ProgressColumn(
                                    "Win Rate",
                                    help="Percentage of winning trades",
                                    format="%.1f%%",
                                    min_value=0, max_value=100,
                                ),
                "Total Return": st.column_config.NumberColumn("Total Return", format="%.2f%%"),
                "Max Drawdown": st.column_config.NumberColumn("Max Drawdown", format="%.2f%%"),
                "# Trades":     st.column_config.NumberColumn("# Trades", format="%d"),
            },
            use_container_width=True,
        )

    with tab_all:
        st.caption(f"All valid (stock, params) combinations globally ranked by **{rank_by}**. Showing top {top_n}.")
        all_df = results_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
        all_df.index += 1
        all_df.index.name = 'Rank'

        st.dataframe(
            all_df.head(top_n),
            column_config={
                "Symbol":       st.column_config.TextColumn("Symbol"),
                "RSI Period":   st.column_config.NumberColumn("Period", format="%d"),
                "Buy <":        st.column_config.NumberColumn("Buy RSI <", format="%d"),
                "Sell >":       st.column_config.NumberColumn("Sell RSI >", format="%d"),
                "Win Rate":     st.column_config.ProgressColumn(
                                    "Win Rate",
                                    help="Percentage of winning trades",
                                    format="%.1f%%",
                                    min_value=0, max_value=100,
                                ),
                "Total Return": st.column_config.NumberColumn("Total Return", format="%.2f%%"),
                "Max Drawdown": st.column_config.NumberColumn("Max Drawdown", format="%.2f%%"),
                "# Trades":     st.column_config.NumberColumn("# Trades", format="%d"),
            },
            use_container_width=True,
        )

    st.download_button(
        "⬇️ Download Full Results CSV",
        data=results_df.sort_values(sort_col, ascending=ascending).to_csv(index=False),
        file_name="rsi_grid_search_results.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE STOCK BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Run Strategy on specific ticker", divider='red')

main_col1, main_col2 = st.columns([1, 2])

with main_col1:
    try:
        _val_data = pd.read_csv('data/jkse/valuation.csv')
        stock_list = sorted(_val_data['stock'].unique().tolist())
    except:
        stock_list = ['BBCA.JK']
        
    try:
        default_idx = stock_list.index('BBCA.JK')
    except ValueError:
        default_idx = 0
        
    stock = st.selectbox('Stock Ticker', options=stock_list, index=default_idx)
    strategy = st.selectbox('Strategy Time Frame & Metric', options=[
        'Short-Term: RSI Day Trading (Max 3 Days)',
        'Medium-Term: MACD Crossover (1-4 Weeks)', 
        'Medium-Term: SMA 20/50 Crossover (1-3 Months)'
    ])

with main_col2:
    start_date = st.date_input('Start Date', value=datetime.date(2023, 1, 1))
    end_date = st.date_input('End Date', value=datetime.date.today())

# RSI parameters — shown only when RSI strategy is selected
if strategy == 'Short-Term: RSI Day Trading (Max 3 Days)':
    st.markdown("**RSI Parameters**")
    rsi_cols = st.columns(4)
    bt_rsi_period   = rsi_cols[0].number_input("RSI Period",      min_value=3,  max_value=50, value=14, step=1,  key="bt_rsi_period")
    bt_buy_thresh   = rsi_cols[1].number_input("Buy if RSI <",    min_value=5,  max_value=49, value=30, step=1,  key="bt_buy_thresh")
    bt_sell_thresh  = rsi_cols[2].number_input("Sell if RSI >",   min_value=51, max_value=95, value=70, step=1,  key="bt_sell_thresh")
    bt_max_hold     = rsi_cols[3].number_input("Max Hold Days",   min_value=1,  max_value=20, value=3,  step=1,  key="bt_max_hold")

st.divider()


if st.button('Run Backtest'):
    if stock:
        with st.spinner("Fetching data from Supabase..."):
            start_date_str = start_date.strftime('%Y-%m-%d')
            try:
                # Query Supabase historical_prices table
                res = conn.table("historical_prices").select("date,close").eq("symbol", stock).gte("date", start_date_str).execute()
                
                if res.data and len(res.data) > 0:
                    price_df = pd.DataFrame(res.data).sort_values("date").reset_index(drop=True)
                    price_df['date'] = pd.to_datetime(price_df['date'])
                else:
                    st.warning("Insufficient data available in Supabase for the selected date range.")
                    price_df = pd.DataFrame() # Create empty to abort safely
            except Exception as e:
                st.error(f"Error fetching data from Supabase: {e}")
                price_df = pd.DataFrame()

            if not price_df.empty:
                # Prepare data
                price_df = price_df.sort_values('date').set_index('date')
                price_df = price_df[price_df.index <= pd.to_datetime(end_date)]
                
                if len(price_df) > 50:
                    entries = pd.Series(False, index=price_df.index)
                    exits = pd.Series(False, index=price_df.index)

                    if strategy == 'Short-Term: RSI Day Trading (Max 3 Days)':
                        # Wilder's RSI calculation (matches TradingView RMA)
                        delta = price_df['close'].diff()
                        up    = delta.clip(lower=0)
                        down  = -delta.clip(upper=0)
                        ema_up   = up.ewm(alpha=1/bt_rsi_period, adjust=False, min_periods=bt_rsi_period).mean()
                        ema_down = down.ewm(alpha=1/bt_rsi_period, adjust=False, min_periods=bt_rsi_period).mean()
                        rs = ema_up / ema_down
                        price_df['RSI'] = 100 - (100 / (1 + rs))
                        
                        raw_entries = (price_df['RSI'] < bt_buy_thresh).to_numpy()
                        raw_exits   = (price_df['RSI'] > bt_sell_thresh).to_numpy()
                        
                        clean_entries = np.zeros_like(raw_entries, dtype=bool)
                        clean_exits   = np.zeros_like(raw_exits, dtype=bool)
                        
                        in_pos = False
                        days_held = 0
                        
                        for i in range(len(raw_entries)):
                            if not in_pos:
                                if raw_entries[i]:
                                    clean_entries[i] = True
                                    in_pos = True
                                    days_held = 0
                            else:
                                days_held += 1
                                if raw_exits[i] or days_held >= bt_max_hold:
                                    clean_exits[i] = True
                                    in_pos = False
                                    
                        entries = pd.Series(clean_entries, index=price_df.index)
                        exits   = pd.Series(clean_exits,   index=price_df.index)
                        
                    elif strategy == 'Medium-Term: MACD Crossover (1-4 Weeks)':
                        # MACD Crossover using native pandas EWM
                        ema_fast = price_df['close'].ewm(span=12, adjust=False).mean()
                        ema_slow = price_df['close'].ewm(span=26, adjust=False).mean()
                        macd_line = ema_fast - ema_slow
                        signal_line = macd_line.ewm(span=9, adjust=False).mean()
                        
                        entries = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
                        exits = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
                            
                    elif strategy == 'Medium-Term: SMA 20/50 Crossover (1-3 Months)':
                        # SMA Crossover using native pandas rolling mean
                        price_df['SMA20'] = price_df['close'].rolling(window=20).mean()
                        price_df['SMA50'] = price_df['close'].rolling(window=50).mean()
                        
                        entries = (price_df['SMA20'] > price_df['SMA50']) & (price_df['SMA20'].shift(1) <= price_df['SMA50'].shift(1))
                        exits = (price_df['SMA20'] < price_df['SMA50']) & (price_df['SMA20'].shift(1) >= price_df['SMA50'].shift(1))

                    # Perform VectorBT backtesting
                    pf = vbt.Portfolio.from_signals(price_df['close'], entries, exits, freq='1D')
                    
                    st.subheader('Performance Outline')
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    # Compute metrics. Provide safe defaults if no trades occurred.
                    tot_ret = pf.total_return() * 100 if pf.total_return() is not None else 0.0
                    idx_ret = ((price_df['close'].iloc[-1] / price_df['close'].iloc[0]) - 1) * 100
                    
                    try:
                        win_r = pf.trades.win_rate() * 100 if len(pf.trades) > 0 else 0.0
                    except:
                        win_r = 0.0

                    try:
                        max_dd = pf.max_drawdown() * 100 if pf.max_drawdown() is not None else 0.0
                    except:
                        max_dd = 0.0

                    metrics_col1.metric("Total Return", f"{tot_ret:.2f}%", delta=f"vs idx {idx_ret:.2f}%")
                    metrics_col2.metric("Win Rate", f"{win_r:.2f}%")
                    metrics_col3.metric("Max Drawdown", f"{max_dd:.2f}%")

                    # Plotting chart
                    st.plotly_chart(pf.plot())
                    
                    with st.expander('View Trade Log'):
                        if len(pf.trades) > 0:
                            st.dataframe(pf.trades.records_readable)
                        else:
                            st.info("No trades executed within this time window.")
                else:
                    st.warning("Not enough data to run strategies. Try selecting a broader date range.")
            else:
                st.error("No historical data found for the given stock and timeframe.")


# ─────────────────────────────────────────────────────────────────────────────
# RSI LIVE ALERT SCANNER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🚨 RSI Buy Signal Scanner", divider='orange')
st.caption(
    "Check which shortlisted stocks are currently oversold (RSI below buy threshold) "
    "using the latest available price data."
)

alert_col1, alert_col2, alert_col3, alert_col4 = st.columns(4)
alert_period     = alert_col1.number_input("RSI Period",    min_value=3,  max_value=50, value=14, step=1,  key="alert_period")
alert_buy_thresh = alert_col2.number_input("Buy if RSI <",  min_value=5,  max_value=49, value=30, step=1,  key="alert_buy")
alert_watch_band = alert_col3.number_input("Watch Band (+)",min_value=1,  max_value=20, value=10, step=1,  key="alert_watch",
                                            help="Stocks with RSI between Buy threshold and Buy+Watch are flagged as 'Watch'")
alert_lookback   = alert_col4.number_input("Days of History",min_value=30, max_value=365, value=120, step=10, key="alert_lookback")

scan_button = st.button("🔍 Scan for Buy Signals", type="primary", disabled=len(shortlist_tickers) == 0)

def _compute_current_rsi(price_series: pd.Series, period: int) -> float | None:
    """Return the most recent RSI value for a price series."""
    if len(price_series) < period + 5:
        return None
    # Wilder's RSI calculation (matches TradingView RMA)
    delta = price_series.diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    ema_up   = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    ema_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs  = ema_up / ema_down
    rsi_val = (100 - (100 / (1 + rs))).iloc[-1]
    return float(rsi_val) if not pd.isna(rsi_val) else None

if scan_button:
    alert_start = (datetime.date.today() - datetime.timedelta(days=int(alert_lookback))).strftime('%Y-%m-%d')

    with st.spinner("Fetching live prices..."):
        try:
            # Batch fetch live prices for all shortlisted stocks
            cp_df = hd.get_company_profile(shortlist_tickers)
        except Exception as e:
            st.warning(f"Could not fetch live prices via get_company_profile: {e}")
            cp_df = pd.DataFrame()

    scan_progress = st.progress(0, text="Scanning shortlist…")
    scan_rows = []

    # Prefer cached optimizer prices if they cover the needed date range, else fetch fresh
    cached_prices = st.session_state.get('optimizer_prices', pd.DataFrame())
    use_cache = (
        not cached_prices.empty
        and cached_prices['date'].min() <= pd.to_datetime(alert_start)
    )

    if use_cache:
        grouped_alert = {
            sym: grp.set_index('date')['close']
            for sym, grp in cached_prices.groupby('symbol')
        }
        # run RSI calc on each cached ticker
        for i, ticker in enumerate(shortlist_tickers):
            scan_progress.progress((i + 1) / len(shortlist_tickers), text=f"Checking {ticker}…")
            series = grouped_alert.get(ticker)
            if series is None or series.empty:
                continue
            
            series = series[series.index >= pd.to_datetime(alert_start)].copy()
            
            # Incorporate live price from cp_df
            if ticker in cp_df.index:
                live_price = float(cp_df.loc[ticker, 'price'])
                today_ts = pd.Timestamp(datetime.date.today())
                if series.index[-1].date() == today_ts.date():
                    series.iloc[-1] = live_price
                else:
                    series[today_ts] = live_price
                series = series.sort_index()

            current_price = float(series.iloc[-1])
            rsi_val = _compute_current_rsi(series, int(alert_period))
            if rsi_val is not None:
                scan_rows.append({'Symbol': ticker, 'Current Price': current_price, 'RSI': rsi_val})
    else:
        # Fetch last N days for each ticker individually
        for i, ticker in enumerate(shortlist_tickers):
            scan_progress.progress((i + 1) / len(shortlist_tickers), text=f"Fetching {ticker}…")
            try:
                res = (
                    conn.table("historical_prices")
                        .select("symbol,date,close")
                        .eq("symbol", ticker)
                        .gte("date", alert_start)
                        .execute()
                )
                rows = res.data or []
                if rows:
                    series = (
                        pd.DataFrame(rows)
                          .assign(date=lambda d: pd.to_datetime(d['date']),
                                  close=lambda d: pd.to_numeric(d['close'], errors='coerce'))
                          .sort_values('date')
                          .set_index('date')['close']
                    )
                    
                    # Incorporate live price from cp_df
                    if ticker in cp_df.index:
                        live_price = float(cp_df.loc[ticker, 'price'])
                        today_ts = pd.Timestamp(datetime.date.today())
                        if series.index[-1].date() == today_ts.date():
                            series.iloc[-1] = live_price
                        else:
                            series[today_ts] = live_price
                        series = series.sort_index()

                    current_price = float(series.iloc[-1])
                    rsi_val = _compute_current_rsi(series, int(alert_period))
                    if rsi_val is not None:
                        scan_rows.append({'Symbol': ticker, 'Current Price': current_price, 'RSI': rsi_val})
            except Exception:
                pass

    scan_progress.empty()

    if not scan_rows:
        st.warning("Could not compute RSI for any stock. Try increasing 'Days of History'.")
    else:
        alert_df = pd.DataFrame(scan_rows)

        buy_thresh  = int(alert_buy_thresh)
        watch_upper = buy_thresh + int(alert_watch_band)

        def _classify(rsi):
            if rsi < buy_thresh:
                return "🟢 Buy Signal"
            elif rsi < watch_upper:
                return "🟡 Watch"
            else:
                return "⚪ Neutral"

        alert_df['Signal']   = alert_df['RSI'].apply(_classify)
        alert_df['RSI']      = alert_df['RSI'].round(2)
        alert_df = alert_df.sort_values('RSI')  # lowest RSI (most oversold) first

        # Summary metrics
        n_buy   = (alert_df['Signal'] == "🟢 Buy Signal").sum()
        n_watch = (alert_df['Signal'] == "🟡 Watch").sum()
        n_total = len(alert_df)

        sum_cols = st.columns(3)
        sum_cols[0].metric("🟢 Buy Signals",  n_buy,   help=f"RSI < {buy_thresh}")
        sum_cols[1].metric("🟡 Watch",        n_watch, help=f"RSI {buy_thresh}–{watch_upper}")
        sum_cols[2].metric("Stocks Scanned",  n_total)

        if n_buy > 0:
            st.success(f"**{n_buy} stock(s) currently in oversold territory** (RSI < {buy_thresh})!")

        # Show buy signals and watch stocks prominently, then neutrals collapsed
        buy_signal_df  = alert_df[alert_df['Signal'] == "🟢 Buy Signal"]
        watch_df       = alert_df[alert_df['Signal'] == "🟡 Watch"]
        neutral_df     = alert_df[alert_df['Signal'] == "⚪ Neutral"]

        col_config = {
            "Symbol":        st.column_config.TextColumn("Symbol"),
            "Current Price": st.column_config.NumberColumn("Price", format="%.0f"),
            "RSI":           st.column_config.ProgressColumn(
                                 "RSI",
                                 help="Current RSI value",
                                 format="%.1f",
                                 min_value=0, max_value=100,
                             ),
            "Signal":        st.column_config.TextColumn("Signal"),
        }

        if not buy_signal_df.empty:
            st.markdown("#### 🟢 Buy Signals")
            st.dataframe(buy_signal_df, column_config=col_config, hide_index=True, use_container_width=True)

        if not watch_df.empty:
            st.markdown("#### 🟡 Approaching Buy Zone")
            st.dataframe(watch_df, column_config=col_config, hide_index=True, use_container_width=True)

        with st.expander(f"⚪ Neutral stocks ({len(neutral_df)})"):
            st.dataframe(neutral_df, column_config=col_config, hide_index=True, use_container_width=True)

