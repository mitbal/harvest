import itertools
import hashlib
import concurrent.futures
import os
import streamlit as st
import datetime
import pandas as pd
import numpy as np
import vectorbt as vbt
from st_supabase_connection import SupabaseConnection

import harvest.data as hd

STRATEGY_ENGINE_VERSION = 4

st.set_page_config(page_title='Day Trading - Panen Dividen')
st.title('Day Trading Strategy Lab')

@st.cache_data(max_entries=64)
def df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_price_volatility(symbols: tuple[str, ...], lookback: int = 60) -> pd.DataFrame:
    """Calculate recent daily volatility from adjusted-quality local close history."""
    columns = ['stock', 'Daily Volatility', 'Volatility Observations']
    try:
        prices = pd.read_pickle('data/jkse/historical_prices.pkl')
        prices = prices[prices['symbol'].isin(symbols)][['symbol', 'date', 'close']].copy()
        prices['date'] = pd.to_datetime(prices['date'], errors='coerce')
        prices['close'] = pd.to_numeric(prices['close'], errors='coerce')
        prices = prices.dropna().sort_values(['symbol', 'date'])
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
    **Regime-aware day trading** — detect bull vs bear market using JKSE index,
    then apply the right strategy for each regime:

    * **Bear market**: buy deeply oversold stocks (mean-reversion bounce), hold 1–7 days.
    * **Bull market**: buy dips in an uptrend (pullback entry), hold 1–7 days.

    Sweep entry-condition grids, backtest individual tickers, and scan the live
    universe for today's closing entries.
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


def compute_regime_series(index_df: pd.DataFrame, fast_sma: int = 50, slow_sma: int = 200) -> pd.Series:
    """
    Return a Series (index = date) with 'bull' / 'bear' labels.
    Bull = fast SMA > slow SMA (uptrend).  Bear = fast SMA <= slow SMA (downtrend).
    """
    if index_df.empty or len(index_df) < slow_sma + 5:
        return pd.Series(dtype=str)
    close = index_df['close']
    sma_fast = close.rolling(window=fast_sma, min_periods=fast_sma).mean()
    sma_slow = close.rolling(window=slow_sma, min_periods=slow_sma).mean()
    regime = pd.Series('bear', index=close.index, dtype=str)
    regime[sma_fast > sma_slow] = 'bull'
    return regime.dropna()


def get_regime_for_date(regime_series: pd.Series, date: pd.Timestamp) -> str | None:
    """Return the regime label for a given date (most recent known value)."""
    if regime_series.empty:
        return None
    valid = regime_series[regime_series.index <= date]
    if valid.empty:
        return None
    return valid.iloc[-1]


def get_current_regime(regime_series: pd.Series) -> str | None:
    """Return the most recent regime label."""
    if regime_series.empty:
        return None
    return regime_series.iloc[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# CORE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up   = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up   = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    ema_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs  = ema_up / ema_down
    return 100 - (100 / (1 + rs))


def _build_regime_signals(
    price_series: pd.Series,
    regime_series: pd.Series,
    bear_rsi_period: int,
    bear_oversold: int,
    bull_rsi_period: int,
    bull_dip_rsi: int,
    bull_trend_sma: int,
    max_hold: int,
    profit_target_pct: float,
    stop_loss_pct: float,
    trade_start: pd.Timestamp | None = None,
):
    """Build market-on-close entries and close-based exits."""
    price_series = pd.to_numeric(price_series, errors='coerce').dropna().sort_index()
    min_len = max(bear_rsi_period, bull_rsi_period, bull_trend_sma, 50) + 5
    if len(price_series) < min_len:
        return None

    bear_rsi = _wilder_rsi(price_series, bear_rsi_period)
    bull_rsi = _wilder_rsi(price_series, bull_rsi_period)
    trend_sma = price_series.rolling(window=bull_trend_sma, min_periods=bull_trend_sma).mean()

    if regime_series.empty:
        regime_aligned = pd.Series('bear', index=price_series.index, dtype=str)
    else:
        regime_aligned = regime_series.reindex(price_series.index, method='ffill').fillna('bear')

    # Enter at the same low close that produces the oversold signal.
    bear_setup = bear_rsi < bear_oversold
    bull_setup = (
        (bull_rsi < bull_dip_rsi)
        & (price_series > trend_sma)
    )
    setup = pd.Series(
        np.where(regime_aligned == 'bear', bear_setup, bull_setup),
        index=price_series.index,
        dtype=bool,
    ).fillna(False)

    # Assume a market-on-close order can execute at the same close that produces
    # the oversold signal.
    entry_candidates = setup
    if trade_start is not None:
        entry_candidates.loc[entry_candidates.index < pd.Timestamp(trade_start)] = False
    n = len(price_series)
    clean_entries = np.zeros(n, dtype=bool)
    clean_exits = np.zeros(n, dtype=bool)
    close_arr = price_series.to_numpy(dtype=float)
    in_pos = False
    entry_price = 0.0
    days_held = 0

    for i in range(n):
        if in_pos:
            days_held += 1
            hit_target = profit_target_pct > 0 and close_arr[i] >= entry_price * (1 + profit_target_pct / 100)
            hit_stop = stop_loss_pct > 0 and close_arr[i] <= entry_price * (1 - stop_loss_pct / 100)
            if hit_target or hit_stop or days_held >= max_hold:
                clean_exits[i] = True
                in_pos = False
                continue

        if not in_pos and entry_candidates.iloc[i]:
            clean_entries[i] = True
            in_pos = True
            entry_price = close_arr[i]
            days_held = 0

    # Realize the last open position so every backtest trade is measurable.
    if in_pos:
        clean_exits[-1] = True

    return (
        price_series,
        pd.Series(clean_entries, index=price_series.index),
        pd.Series(clean_exits, index=price_series.index),
        regime_aligned,
    )


def _run_regime_backtest(
    price_series: pd.Series,
    regime_series: pd.Series,
    bear_rsi_period: int,
    bear_oversold: int,
    bull_rsi_period: int,
    bull_dip_rsi: int,
    bull_trend_sma: int,
    max_hold: int,
    profit_target_pct: float,
    stop_loss_pct: float,
    buy_fee: float,
    sell_fee: float,
    sell_tax: float,
    trade_start: pd.Timestamp | None = None,
):
    """
    Regime-aware day-trade backtest with multi-day hold.

    Bear market:  buy when RSI < bear_oversold (deep oversold bounce).
    Bull market:  buy when RSI < bull_dip_rsi AND close > SMA(trend) (dip in uptrend).

    Exit (checked each day while in position):
      - Profit target hit  → close >= entry_price * (1 + profit_target_pct/100)
      - Stop loss hit      → close <= entry_price * (1 - stop_loss_pct/100)
      - Max hold reached   → days_held >= max_hold
    """
    signals = _build_regime_signals(
        price_series, regime_series, bear_rsi_period, bear_oversold,
        bull_rsi_period, bull_dip_rsi, bull_trend_sma, max_hold,
        profit_target_pct, stop_loss_pct, trade_start,
    )
    if signals is None:
        return None
    price_series, entries, exits, regime_aligned = signals

    if not entries.any():
        return None

    fees = pd.Series(0.0, index=price_series.index)
    fees[entries] = buy_fee / 100.0
    fees[exits]   = (sell_fee + sell_tax) / 100.0

    try:
        pf = vbt.Portfolio.from_signals(price_series, entries, exits, freq='1D', fees=fees)
        num_trades = len(pf.trades)
        if num_trades == 0:
            return None
        tot_ret   = float(pf.total_return() * 100)
        win_rate  = float(pf.trades.win_rate() * 100)
        max_dd    = float(pf.max_drawdown() * 100)
        avg_trade = float(pf.trades.returns.mean() * 100) if len(pf.trades) > 0 else 0.0
        bear_entries = int((entries & (regime_aligned == 'bear')).sum())
        bull_entries = int((entries & (regime_aligned == 'bull')).sum())
        return tot_ret, win_rate, max_dd, num_trades, avg_trade, bear_entries, bull_entries
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TOP RANKED VALUATION SHORTLIST
# ═══════════════════════════════════════════════════════════════════════════════

with st.container():
    st.subheader("Top Ranked Shortlist", divider='blue')
    filter_cols = st.columns(4)
    min_mcap_trill = filter_cols[0].number_input("Min Market Cap (Trillion IDR)", value=1.0, step=1.0, key="dt_mcap")
    min_pe = filter_cols[1].number_input("Minimum PE", value=5.0, step=1.0, key="dt_minpe")
    max_pe = filter_cols[2].number_input("Maximum PE", value=50.0, step=1.0, key="dt_maxpe")
    max_stocks = filter_cols[3].number_input("Max Stocks Output", min_value=10, max_value=200, value=50, step=10, key="dt_maxs")
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
_regime_fast = 50
_regime_slow = 200
full_regime_series = compute_regime_series(jkse_df, fast_sma=_regime_fast, slow_sma=_regime_slow)
current_regime = get_current_regime(full_regime_series)

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

SEARCH_PRESETS = {
    "Quick": {
        "rsi_periods": [7, 14],
        "bear_buys": [30, 40],
        "bull_buys": [40, 50],
        "hold_days": [3, 5],
        "exit_rules": [(3.0, 2.0)],
    },
    "Balanced": {
        "rsi_periods": [7, 10, 14, 21],
        "bear_buys": [25, 30, 35, 40],
        "bull_buys": [40, 45, 50],
        "hold_days": [3, 5, 7],
        "exit_rules": [(2.0, 1.5), (3.0, 2.0)],
    },
    "Deep": {
        "rsi_periods": [5, 7, 10, 14, 21],
        "bear_buys": [25, 30, 35, 40, 45],
        "bull_buys": [35, 40, 45, 50, 55],
        "hold_days": [2, 3, 5, 7],
        "exit_rules": [(2.0, 1.0), (3.0, 1.5), (4.0, 2.0), (5.0, 3.0)],
    },
}
OPTIMIZER_WORKERS = min(8, max(2, os.cpu_count() or 2))


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    valid = frame[[value_col, weight_col]].dropna()
    valid = valid[valid[weight_col] > 0]
    if valid.empty:
        return np.nan
    return float(np.average(valid[value_col], weights=valid[weight_col]))


def _summarize_strategies(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each parameter set across stocks and score holdout robustness."""
    params = [
        'Bear Period', 'Bear Buy <', 'Bull Period', 'Bull Buy <', 'Trend SMA',
        'Max Hold', 'Profit Target', 'Stop Loss',
    ]
    rows = []
    universe_size = max(1, results_df['Symbol'].nunique())
    for values, group in results_df.groupby(params, dropna=False):
        validated = group[group['Validation Trades'] > 0]
        rows.append({
            **dict(zip(params, values)),
            'Stocks Tested': int(group['Symbol'].nunique()),
            'Stocks Validated': int(validated['Symbol'].nunique()),
            'Validation Coverage %': float(validated['Symbol'].nunique() / universe_size * 100),
            'Train Trades': int(group['# Trades'].sum()),
            'Validation Trades': int(validated['Validation Trades'].sum()),
            'Train Avg Trade': _weighted_mean(group, 'Avg Trade', '# Trades'),
            'Validation Avg Trade': _weighted_mean(validated, 'Validation Avg Trade', 'Validation Trades'),
            'Validation Win Rate': _weighted_mean(validated, 'Validation Win Rate', 'Validation Trades'),
            'Median Validation Return': float(validated['Validation Return'].median()) if not validated.empty else np.nan,
            'Positive Stocks %': float((validated['Validation Return'] > 0).mean() * 100) if not validated.empty else np.nan,
            'Validation Drawdown': _weighted_mean(validated, 'Validation Drawdown', 'Validation Trades'),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary['Generalization Gap'] = summary['Validation Avg Trade'] - summary['Train Avg Trade']
    score_inputs = {
        'Validation Avg Trade': 0.30,
        'Validation Win Rate': 0.15,
        'Median Validation Return': 0.15,
        'Positive Stocks %': 0.15,
        'Validation Coverage %': 0.15,
    }
    summary['Strategy Score'] = 0.0
    for column, weight in score_inputs.items():
        summary['Strategy Score'] += summary[column].rank(pct=True).fillna(0) * weight * 100
    drawdown_control = (-summary['Validation Drawdown'].abs()).rank(pct=True).fillna(0) * 100
    summary['Strategy Score'] += drawdown_control * 0.10
    return summary.sort_values(
        ['Strategy Score', 'Validation Avg Trade', 'Validation Trades'], ascending=False,
    ).reset_index(drop=True)


def _select_stock_strategies(
    results_df: pd.DataFrame,
    global_strategy: dict,
) -> dict[str, dict]:
    """Select stock overrides only when holdout return beats the global strategy."""
    selections = {}
    for symbol, group in results_df.groupby('Symbol'):
        global_rows = group[
            (group['Bear Period'] == global_strategy['rsi_period'])
            & (group['Bear Buy <'] == global_strategy['bear_buy'])
            & (group['Bull Period'] == global_strategy['rsi_period'])
            & (group['Bull Buy <'] == global_strategy['bull_buy'])
            & (group['Trend SMA'] == global_strategy['trend_sma'])
            & (group['Max Hold'] == global_strategy['max_hold'])
            & np.isclose(group['Profit Target'], global_strategy['profit_target'])
            & np.isclose(group['Stop Loss'], global_strategy['stop_loss'])
        ]
        if global_rows.empty:
            continue
        global_row = global_rows.iloc[0]
        global_return = float(global_row['Validation Return'])
        if global_row['Validation Trades'] <= 0 or not np.isfinite(global_return):
            continue

        candidates = group[
            (group['Validation Trades'] > 0)
            & (group['Validation Return'] > global_return + 1e-9)
        ].copy()
        if candidates.empty:
            continue

        score_weights = {
            'Validation Avg Trade': 0.35,
            'Validation Win Rate': 0.20,
            'Validation Return': 0.20,
            'Validation Trades': 0.15,
        }
        candidates['Stock Strategy Score'] = 0.0
        for column, weight in score_weights.items():
            candidates['Stock Strategy Score'] += candidates[column].rank(pct=True).fillna(0) * weight * 100
        drawdown_control = (-candidates['Validation Drawdown'].abs()).rank(pct=True).fillna(0) * 100
        candidates['Stock Strategy Score'] += drawdown_control * 0.10
        best = candidates.sort_values(
            ['Stock Strategy Score', 'Validation Trades', 'Validation Avg Trade'],
            ascending=False,
        ).iloc[0]
        selections[symbol] = {
            'rsi_period': int(best['Bear Period']),
            'bear_buy': int(best['Bear Buy <']),
            'bull_buy': int(best['Bull Buy <']),
            'trend_sma': int(best['Trend SMA']),
            'max_hold': int(best['Max Hold']),
            'profit_target': float(best['Profit Target']),
            'stop_loss': float(best['Stop Loss']),
            'score': float(best['Stock Strategy Score']),
            'validation_trades': int(best['Validation Trades']),
            'validation_avg_trade': float(best['Validation Avg Trade']),
            'validation_return': float(best['Validation Return']),
            'global_validation_return': global_return,
            'return_uplift': float(best['Validation Return'] - global_return),
        }
    return selections


def _test_stock_strategies(
    ticker: str,
    price_series: pd.Series | None,
    regime_series: pd.Series,
    param_combos: list[tuple],
    validation_pct: int,
    bull_trend_sma: int,
    buy_fee: float,
    sell_fee: float,
    sell_tax: float,
) -> tuple[list[dict], str]:
    """Test every strategy for one stock; safe to execute in a worker thread."""
    if price_series is None or price_series.empty:
        return [], 'missing_data'

    price_series = price_series[~price_series.index.duplicated(keep='last')].sort_index()
    split_idx = int(len(price_series) * (1 - validation_pct / 100))
    max_rsi_period = max(combo[0] for combo in param_combos)
    indicator_warmup = max(max_rsi_period, int(bull_trend_sma), 50) + 5
    if split_idx < indicator_warmup or len(price_series) - split_idx < indicator_warmup:
        return [], 'insufficient_history'

    train_prices = price_series.iloc[:split_idx]
    validation_start = price_series.index[split_idx]
    validation_prices = price_series.iloc[max(0, split_idx - indicator_warmup):]
    rows = []

    for rsi_p, bear_b, bull_b, hold_d, profit_target, stop_loss in param_combos:
        result = _run_regime_backtest(
            train_prices, regime_series,
            bear_rsi_period=int(rsi_p), bear_oversold=int(bear_b),
            bull_rsi_period=int(rsi_p), bull_dip_rsi=int(bull_b),
            bull_trend_sma=int(bull_trend_sma), max_hold=int(hold_d),
            profit_target_pct=float(profit_target), stop_loss_pct=float(stop_loss),
            buy_fee=buy_fee, sell_fee=sell_fee, sell_tax=sell_tax,
        )
        if result is None:
            continue

        tot_ret, win_rate, max_dd, num_trades, avg_trade, n_bear, n_bull = result
        validation_result = _run_regime_backtest(
            validation_prices, regime_series,
            bear_rsi_period=int(rsi_p), bear_oversold=int(bear_b),
            bull_rsi_period=int(rsi_p), bull_dip_rsi=int(bull_b),
            bull_trend_sma=int(bull_trend_sma), max_hold=int(hold_d),
            profit_target_pct=float(profit_target), stop_loss_pct=float(stop_loss),
            buy_fee=buy_fee, sell_fee=sell_fee, sell_tax=sell_tax,
            trade_start=validation_start,
        )
        if validation_result is None:
            val_ret, val_win, val_dd, val_trades, val_avg = np.nan, np.nan, np.nan, 0, np.nan
        else:
            val_ret, val_win, val_dd, val_trades, val_avg, _, _ = validation_result

        rows.append({
            'Symbol': ticker,
            'Bear Period': rsi_p,
            'Bear Buy <': bear_b,
            'Bull Period': rsi_p,
            'Bull Buy <': bull_b,
            'Trend SMA': bull_trend_sma,
            'Max Hold': hold_d,
            'Profit Target': profit_target,
            'Stop Loss': stop_loss,
            'Total Return': tot_ret,
            'Avg Trade': avg_trade,
            'Win Rate': win_rate,
            'Max Drawdown': max_dd,
            '# Trades': num_trades,
            'Bear Entries': n_bear,
            'Bull Entries': n_bull,
            'Validation Return': val_ret,
            'Validation Win Rate': val_win,
            'Validation Drawdown': val_dd,
            'Validation Trades': val_trades,
            'Validation Avg Trade': val_avg,
        })

    return rows, 'ok' if rows else 'no_signals'

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Find the Best Day-Trade Strategy", divider='violet')
st.caption(
    "Test a curated set of RSI, holding-period, and risk/reward combinations. "
    "The most recent data is held out, then each strategy is ranked by how "
    "consistently it performs across the full stock shortlist."
)

search_cols = st.columns(3)
search_depth = search_cols[0].selectbox(
    "Search Depth", options=list(SEARCH_PRESETS), index=1, key="dt_search_depth",
    help="Quick for iteration, Balanced for normal use, Deep for final research",
)
bull_trend_sma = search_cols[1].selectbox(
    "Bull Trend Filter", options=[20, 50, 100], index=1, key="dt_bull_sma",
    help="Bull entries require price above this moving average",
)
opt_validation_pct = search_cols[2].slider(
    "Holdout Data %", min_value=20, max_value=40, value=30, step=5,
    key="dt_validation_pct",
)

date_cols = st.columns(2)
opt_start = date_cols[0].date_input("Start Date", value=datetime.date(2022, 1, 1), key="dt_start")
opt_end = date_cols[1].date_input("End Date", value=datetime.date.today(), key="dt_end")

with st.expander("Advanced: market regime and trading costs"):
    advanced_cols = st.columns(5)
    regime_fast = advanced_cols[0].number_input("Regime Fast SMA", 10, 100, 50, 10, key="dt_reg_fast")
    regime_slow = advanced_cols[1].number_input("Regime Slow SMA", 50, 300, 200, 10, key="dt_reg_slow")
    opt_buy_fee = advanced_cols[2].number_input("Buy Fee %", 0.0, 5.0, 0.15, 0.01, key="dt_buy_fee")
    opt_sell_fee = advanced_cols[3].number_input("Sell Fee %", 0.0, 5.0, 0.15, 0.01, key="dt_sell_fee")
    opt_sell_tax = advanced_cols[4].number_input("Sell Tax %", 0.0, 5.0, 0.1, 0.01, key="dt_sell_tax")

search_space = SEARCH_PRESETS[search_depth]
rsi_periods = search_space['rsi_periods']
bear_buys = search_space['bear_buys']
bull_buys = search_space['bull_buys']
hold_days = search_space['hold_days']
exit_rules = search_space['exit_rules']
param_combos = [
    (rsi_p, bear_b, bull_b, hold_d, profit_target, stop_loss)
    for rsi_p, bear_b, bull_b, hold_d, (profit_target, stop_loss)
    in itertools.product(rsi_periods, bear_buys, bull_buys, hold_days, exit_rules)
]
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
    f"{STRATEGY_ENGINE_VERSION}{param_combos}{bull_trend_sma}{regime_fast}{regime_slow}{opt_start}{opt_end}{shortlist_tickers}{opt_buy_fee}{opt_sell_fee}{opt_sell_tax}{opt_validation_pct}".encode()
).hexdigest()

if run_optimizer:
    st.session_state.pop('dt_optimizer_results', None)
    st.session_state.pop('dt_optimizer_results_cache_key', None)

    regime_series = compute_regime_series(jkse_df, fast_sma=int(regime_fast), slow_sma=int(regime_slow))

    if st.session_state.get('dt_price_cache_key') == price_cache_key and \
       'dt_prices' in st.session_state:
        all_prices_df = st.session_state['dt_prices']
        st.info(f"Using cached price data ({len(all_prices_df):,} rows). Only grid params changed.")
    else:
        fetch_progress = st.progress(0, text="Fetching price data from Supabase...")
        all_rows = []
        try:
            start_str = opt_start.strftime('%Y-%m-%d')

            for i, ticker in enumerate(shortlist_tickers):
                fetch_progress.progress(
                    (i + 1) / len(shortlist_tickers),
                    text=f"Fetching {ticker}  ({i+1}/{len(shortlist_tickers)})..."
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
                    pass

            fetch_progress.empty()

            all_prices_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
            if not all_prices_df.empty:
                all_prices_df['date']  = pd.to_datetime(all_prices_df['date'])
                all_prices_df['close'] = pd.to_numeric(all_prices_df['close'], errors='coerce')
                all_prices_df = all_prices_df[all_prices_df['date'] <= pd.to_datetime(opt_end)]
                all_prices_df = all_prices_df.sort_values(['symbol', 'date'])

            st.session_state['dt_prices'] = all_prices_df
            st.session_state['dt_price_cache_key'] = price_cache_key

        except Exception as e:
            fetch_progress.empty()
            st.error(f"Failed to fetch price data: {e}")
            all_prices_df = pd.DataFrame()

    if all_prices_df.empty:
        st.error("No price data returned. Make sure the shortlist is non-empty and Supabase is reachable.")
    else:
        fetch_cols = st.columns(3)
        fetch_cols[0].metric("Rows Fetched", f"{len(all_prices_df):,}")
        fetch_cols[1].metric("Stocks with Data", f"{all_prices_df['symbol'].nunique()}")
        fetch_cols[2].metric("Expected Stocks", f"{len(shortlist_tickers)}")

        results = []
        progress_bar = st.progress(0, text="Testing strategies...")
        grouped = {
            sym: grp.set_index('date')['close']
            for sym, grp in all_prices_df.groupby('symbol')
        }
        status_counts = {
            'ok': 0, 'no_signals': 0, 'missing_data': 0,
            'insufficient_history': 0, 'error': 0,
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=OPTIMIZER_WORKERS) as executor:
            futures = {
                executor.submit(
                    _test_stock_strategies,
                    ticker,
                    grouped.get(ticker),
                    regime_series,
                    param_combos,
                    int(opt_validation_pct),
                    int(bull_trend_sma),
                    float(opt_buy_fee),
                    float(opt_sell_fee),
                    float(opt_sell_tax),
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
    strategy_df = _summarize_strategies(results_df)
    tested_stocks = max(1, results_df['Symbol'].nunique())
    min_validated_stocks = max(3, int(np.ceil(tested_stocks * 0.20)))
    min_validation_trades = max(20, tested_stocks * 2)
    qualified_df = strategy_df[
        (strategy_df['Stocks Validated'] >= min_validated_stocks)
        & (strategy_df['Validation Trades'] >= min_validation_trades)
    ].copy()

    if qualified_df.empty:
        qualified_df = strategy_df[strategy_df['Validation Trades'] > 0].copy()
        st.warning(
            "No strategy met the preferred validation coverage. Showing the best "
            "available results; use a longer date range for stronger evidence."
        )

    col_cfg = {
        "Symbol":        st.column_config.TextColumn("Symbol"),
        "Bear Period":   st.column_config.NumberColumn("RSI Period", format="%d"),
        "Bear Buy <":    st.column_config.NumberColumn("Bear RSI Buy <", format="%d"),
        "Bull Buy <":    st.column_config.NumberColumn("Bull RSI Buy <", format="%d"),
        "Trend SMA":     st.column_config.NumberColumn("Trend SMA", format="%d"),
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
        "Stocks Tested": st.column_config.NumberColumn("Stocks Tested", format="%d"),
        "Stocks Validated": st.column_config.NumberColumn("Stocks Validated", format="%d"),
        "Train Trades": st.column_config.NumberColumn("Train Trades", format="%d"),
        "Train Avg Trade": st.column_config.NumberColumn("Train Avg Trade", format="%.2f%%"),
        "Median Validation Return": st.column_config.NumberColumn("Median Validation Return", format="%.2f%%"),
        "Positive Stocks %": st.column_config.ProgressColumn("Positive Stocks", format="%.1f%%", min_value=0, max_value=100),
        "Validation Coverage %": st.column_config.ProgressColumn("Stock Coverage", format="%.1f%%", min_value=0, max_value=100),
        "Generalization Gap": st.column_config.NumberColumn("Validation - Train", format="%.2f%%"),
        "Strategy Score": st.column_config.ProgressColumn("Strategy Score", format="%.1f", min_value=0, max_value=100),
    }

    tab_strategy, tab_stocks, tab_all = st.tabs([
        "Recommended Strategy", "Stock Breakdown", "All Strategies",
    ])
    st.session_state['dt_stock_strategies'] = {}
    st.session_state['dt_stock_strategies_cache_key'] = results_cache_key

    if not qualified_df.empty:
        recommended = qualified_df.iloc[0]
        recommended_strategy = {
            'rsi_period': int(recommended['Bear Period']),
            'bear_buy': int(recommended['Bear Buy <']),
            'bull_buy': int(recommended['Bull Buy <']),
            'trend_sma': int(recommended['Trend SMA']),
            'max_hold': int(recommended['Max Hold']),
            'profit_target': float(recommended['Profit Target']),
            'stop_loss': float(recommended['Stop Loss']),
            'score': float(recommended['Strategy Score']),
        }
        stock_strategy_map = _select_stock_strategies(results_df, recommended_strategy)
        st.session_state['dt_stock_strategies'] = stock_strategy_map
        st.session_state['dt_stock_strategies_cache_key'] = results_cache_key
        strategy_signature = (
            results_cache_key,
            recommended_strategy['rsi_period'],
            recommended_strategy['bear_buy'],
            recommended_strategy['bull_buy'],
            recommended_strategy['trend_sma'],
            recommended_strategy['max_hold'],
            recommended_strategy['profit_target'],
            recommended_strategy['stop_loss'],
        )
        if st.session_state.get('dt_applied_strategy_signature') != strategy_signature:
            downstream_defaults = {
                'bt_bear_period': recommended_strategy['rsi_period'],
                'bt_bear_oversold': recommended_strategy['bear_buy'],
                'bt_bull_period': recommended_strategy['rsi_period'],
                'bt_bull_dip': recommended_strategy['bull_buy'],
                'bt_bull_sma': recommended_strategy['trend_sma'],
                'bt_max_hold': recommended_strategy['max_hold'],
                'bt_pt': recommended_strategy['profit_target'],
                'bt_sl': recommended_strategy['stop_loss'],
                'bt_buy_fee': float(opt_buy_fee),
                'bt_sell_fee': float(opt_sell_fee),
                'bt_sell_tax': float(opt_sell_tax),
                'dt_alert_bear_p': recommended_strategy['rsi_period'],
                'dt_alert_bear_b': recommended_strategy['bear_buy'],
                'dt_alert_bull_p': recommended_strategy['rsi_period'],
                'dt_alert_bull_b': recommended_strategy['bull_buy'],
                'dt_alert_sma': recommended_strategy['trend_sma'],
            }
            for key, value in downstream_defaults.items():
                st.session_state[key] = value
            st.session_state['dt_active_strategy'] = recommended_strategy
            st.session_state['dt_applied_strategy_signature'] = strategy_signature
            strategy_was_applied = True

        strategy_params = [
            'Bear Period', 'Bear Buy <', 'Bull Period', 'Bull Buy <', 'Trend SMA',
            'Max Hold', 'Profit Target', 'Stop Loss',
        ]
        recommended_runs = results_df.copy()
        for parameter in strategy_params:
            recommended_runs = recommended_runs[recommended_runs[parameter] == recommended[parameter]]

        with tab_strategy:
            st.caption(
                "The recommendation is selected across stocks using only holdout quality, "
                "consistency, and drawdown control. It is not the largest in-sample winner."
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
                    'Strategy Score', 'Bear Period', 'Bear Buy <', 'Bull Buy <',
                    'Trend SMA', 'Max Hold', 'Profit Target', 'Stop Loss',
                    'Validation Avg Trade', 'Validation Win Rate',
                    'Median Validation Return', 'Positive Stocks %',
                    'Validation Coverage %', 'Validation Drawdown',
                    'Validation Trades', 'Stocks Validated',
                    'Generalization Gap',
                ],
                hide_index=True, width='stretch',
            )

        with tab_stocks:
            st.caption("Per-stock results for the single recommended parameter set.")
            st.dataframe(
                recommended_runs.sort_values(
                    ['Validation Avg Trade', 'Validation Trades'], ascending=False,
                ),
                column_config=col_cfg,
                column_order=[
                    'Symbol', 'Validation Avg Trade', 'Validation Win Rate',
                    'Validation Return', 'Validation Drawdown', 'Validation Trades',
                    'Avg Trade', 'Win Rate', '# Trades',
                ],
                hide_index=True, width='stretch',
            )

    with tab_all:
        st.caption(
            f"Strategies need at least {min_validation_trades} holdout trades across "
            f"{min_validated_stocks} stocks to qualify for the recommendation."
        )
        st.dataframe(
            strategy_df, column_config=col_cfg,
            column_order=[
                'Strategy Score', 'Bear Period', 'Bear Buy <', 'Bull Buy <',
                'Trend SMA', 'Max Hold', 'Profit Target', 'Stop Loss',
                'Validation Avg Trade', 'Validation Win Rate',
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
        file_name="daytrade_strategy_rankings.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE STOCK BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Run Regime-Aware Day-Trade on a Specific Ticker", divider='red')

active_strategy = st.session_state.get('dt_active_strategy')
stock_strategy_map = st.session_state.get('dt_stock_strategies', {})

main_col1, main_col2 = st.columns([1, 2])

with main_col1:
    try:
        stock_list = sorted(pd.read_csv('data/jkse/valuation.csv')['stock'].unique().tolist())
    except Exception:
        stock_list = ['BBCA.JK']
    try:
        default_idx = stock_list.index('BBCA.JK')
    except ValueError:
        default_idx = 0

    stock = st.selectbox('Stock Ticker', options=stock_list, index=default_idx, key="dt_single_stock")

with main_col2:
    start_date = st.date_input('Start Date', value=datetime.date(2023, 1, 1), key="dt_single_start")
    end_date   = st.date_input('End Date',   value=datetime.date.today(),     key="dt_single_end")

stock_specific_strategy = stock_strategy_map.get(stock)
single_strategy = stock_specific_strategy or active_strategy
if single_strategy:
    single_strategy_signature = (
        stock,
        st.session_state.get('dt_stock_strategies_cache_key'),
        single_strategy['rsi_period'], single_strategy['bear_buy'],
        single_strategy['bull_buy'], single_strategy['trend_sma'],
        single_strategy['max_hold'], single_strategy['profit_target'],
        single_strategy['stop_loss'],
    )
    if st.session_state.get('dt_single_strategy_signature') != single_strategy_signature:
        single_defaults = {
            'bt_bear_period': single_strategy['rsi_period'],
            'bt_bear_oversold': single_strategy['bear_buy'],
            'bt_bull_period': single_strategy['rsi_period'],
            'bt_bull_dip': single_strategy['bull_buy'],
            'bt_bull_sma': single_strategy['trend_sma'],
            'bt_max_hold': single_strategy['max_hold'],
            'bt_pt': single_strategy['profit_target'],
            'bt_sl': single_strategy['stop_loss'],
        }
        for key, value in single_defaults.items():
            st.session_state[key] = value
        st.session_state['dt_single_strategy_signature'] = single_strategy_signature

with st.expander("Strategy Parameters & Costs", expanded=False):
    if single_strategy:
        parameter_source = "stock-specific grid result" if stock_specific_strategy else "global recommendation"
        return_comparison = ""
        if stock_specific_strategy:
            return_comparison = (
                f" Validation return {single_strategy['validation_return']:.2f}% vs "
                f"global {single_strategy['global_validation_return']:.2f}% "
                f"(+{single_strategy['return_uplift']:.2f}%)."
            )
        st.info(
            f"Using {parameter_source} for **{stock}**: "
            f"RSI {single_strategy['rsi_period']}, bear < {single_strategy['bear_buy']}, "
            f"bull < {single_strategy['bull_buy']}, SMA {single_strategy['trend_sma']}, "
            f"target {single_strategy['profit_target']:.1f}%, "
            f"stop {single_strategy['stop_loss']:.1f}%, hold {single_strategy['max_hold']} days."
            f"{return_comparison}"
        )

    st.markdown("#### Bear Market — Oversold Bounce")
    bsc1, bsc2 = st.columns(2)
    with bsc1:
        bt_bear_period = st.number_input("RSI Period", min_value=3, max_value=50, value=14, step=1, key="bt_bear_period")
    with bsc2:
        bt_bear_oversold = st.number_input("Buy if RSI <", min_value=5, max_value=70, value=30, step=1, key="bt_bear_oversold")

    st.markdown("#### Bull Market — Dip Buy")
    buc1, buc2, buc3 = st.columns(3)
    with buc1:
        bt_bull_period = st.number_input("RSI Period", min_value=3, max_value=50, value=14, step=1, key="bt_bull_period")
    with buc2:
        bt_bull_dip = st.number_input("Buy if RSI <", min_value=10, max_value=70, value=40, step=1, key="bt_bull_dip")
    with buc3:
        bt_bull_sma = st.number_input("Trend SMA (close >)", min_value=10, max_value=200, value=50, step=10, key="bt_bull_sma")

    st.markdown("#### Exit Rules & Costs")
    ex_c1, ex_c2, ex_c3 = st.columns(3)
    with ex_c1:
        bt_max_hold = st.number_input("Max Hold Days", min_value=1, max_value=20, value=5, step=1, key="bt_max_hold")
    with ex_c2:
        bt_profit_target = st.number_input("Profit Target %", min_value=0.0, max_value=20.0, value=3.0, step=0.5, key="bt_pt")
    with ex_c3:
        bt_stop_loss = st.number_input("Stop Loss %", min_value=0.0, max_value=20.0, value=2.0, step=0.5, key="bt_sl")

    cost_cols = st.columns(3)
    bt_buy_fee = cost_cols[0].number_input("Buy Fee %", min_value=0.0, max_value=5.0, value=0.15, step=0.01, key="bt_buy_fee")
    bt_sell_fee = cost_cols[1].number_input("Sell Fee %", min_value=0.0, max_value=5.0, value=0.15, step=0.01, key="bt_sell_fee")
    bt_sell_tax = cost_cols[2].number_input("Sell Tax %", min_value=0.0, max_value=5.0, value=0.1, step=0.01, key="bt_sell_tax")

st.divider()

if st.button('Run Day-Trade Backtest', key="dt_run_single"):
    if stock:
        regime_series = compute_regime_series(jkse_df)

        with st.spinner("Fetching data from Supabase..."):
            start_date_str = start_date.strftime('%Y-%m-%d')
            try:
                res = conn.table("historical_prices").select("date,close").eq("symbol", stock).gte("date", start_date_str).execute()

                if res.data and len(res.data) > 0:
                    price_df = pd.DataFrame(res.data).sort_values("date").reset_index(drop=True)
                    price_df['date'] = pd.to_datetime(price_df['date'])
                else:
                    st.warning("Insufficient data available in Supabase for the selected date range.")
                    price_df = pd.DataFrame()
            except Exception as e:
                st.error(f"Error fetching data from Supabase: {e}")
                price_df = pd.DataFrame()

            if not price_df.empty:
                price_df = price_df.sort_values('date').set_index('date')
                price_df = price_df[price_df.index <= pd.to_datetime(end_date)]

                if len(price_df) > 50:
                    close = price_df['close']

                    signals = _build_regime_signals(
                        close, regime_series, bt_bear_period, bt_bear_oversold,
                        bt_bull_period, bt_bull_dip, bt_bull_sma, bt_max_hold,
                        bt_profit_target, bt_stop_loss,
                    )
                    if signals is None:
                        st.warning("Not enough data to calculate the configured indicators.")
                    else:
                        close, entries, exits, regime_aligned = signals

                    if signals is not None and not entries.any():
                        st.warning("No entry signals generated. Try loosening thresholds.")
                    elif signals is not None:
                        fees = pd.Series(0.0, index=close.index)
                        fees[entries] = bt_buy_fee / 100.0
                        fees[exits]   = (bt_sell_fee + bt_sell_tax) / 100.0

                        pf = vbt.Portfolio.from_signals(close, entries, exits, freq='1D', fees=fees)

                        st.subheader('Performance Outline')
                        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

                        tot_ret = pf.total_return() * 100 if pf.total_return() is not None else 0.0
                        bh_ret  = ((close.iloc[-1] / close.iloc[0]) - 1) * 100

                        try:  win_r = pf.trades.win_rate() * 100 if len(pf.trades) > 0 else 0.0
                        except: win_r = 0.0
                        try:  max_dd = pf.max_drawdown() * 100 if pf.max_drawdown() is not None else 0.0
                        except: max_dd = 0.0
                        try:  avg_trade = pf.trades.returns.mean() * 100 if len(pf.trades) > 0 else 0.0
                        except: avg_trade = 0.0

                        bear_n = int((entries & (regime_aligned == 'bear')).sum())
                        bull_n = int((entries & (regime_aligned == 'bull')).sum())

                        metrics_col1.metric("Total Return", f"{tot_ret:.2f}%", delta=f"vs B&H {bh_ret:.2f}%")
                        metrics_col2.metric("Win Rate",     f"{win_r:.2f}%")
                        metrics_col3.metric("Avg Trade",    f"{avg_trade:.3f}%")
                        metrics_col4.metric("Max Drawdown", f"{max_dd:.2f}%")

                        st.caption(f"Bear entries: {bear_n}  |  Bull entries: {bull_n}  |  Total trades: {len(pf.trades)}")

                        st.plotly_chart(pf.plot())

                        with st.expander('View Trade Log'):
                            if len(pf.trades) > 0:
                                st.dataframe(pf.trades.records_readable)
                            else:
                                st.info("No day trades executed within this time window.")
                else:
                    st.warning("Not enough data to run the strategy. Try selecting a broader date range.")
            else:
                st.error("No historical data found for the given stock and timeframe.")


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE DAY-TRADE SIGNAL SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Live Day-Trade Entry Scanner", divider='orange')
st.caption(
    "Scan today's shortlist closes for regime-appropriate entry signals. "
    "Bear market: deep oversold bounce.  Bull market: dip buy in uptrend. "
    "Signals assume a market-on-close entry at today's closing price. "
    "During market hours, signals remain provisional until the close."
)

curr_regime = get_current_regime(full_regime_series)
if curr_regime:
    regime_label = f"Current regime: **{curr_regime.upper()}**"
    if curr_regime == 'bear':
        st.info(f"🐻 {regime_label} — looking for deeply oversold stocks (mean-reversion bounce).")
    else:
        st.info(f"🐂 {regime_label} — looking for pullback dips in uptrending stocks.")

alert_col1, alert_col2, alert_col3, alert_col4 = st.columns(4)
with alert_col1:
    alert_bear_period = st.number_input("Bear RSI Period",  min_value=3, max_value=50, value=14, step=1, key="dt_alert_bear_p")
with alert_col2:
    alert_bear_buy    = st.number_input(
        "Bear Buy if RSI <", min_value=5, max_value=70, value=30, step=1,
        key="dt_alert_bear_b", help="Enter at today's close when RSI is below this level",
    )
with alert_col3:
    alert_bull_period = st.number_input("Bull RSI Period",  min_value=3, max_value=50, value=14, step=1, key="dt_alert_bull_p")
with alert_col4:
    alert_bull_buy    = st.number_input(
        "Bull Buy if RSI <", min_value=10, max_value=70, value=40, step=1,
        key="dt_alert_bull_b", help="Enter at today's close when RSI is below this level and price is above the trend SMA",
    )

alert_col5, alert_col6, alert_col7 = st.columns(3)
with alert_col5:
    alert_bull_sma    = st.number_input("Bull Trend SMA",   min_value=10, max_value=200, value=50, step=10, key="dt_alert_sma")
with alert_col6:
    alert_lookback    = st.number_input("Days of History",  min_value=30, max_value=365, value=250, step=10, key="dt_alert_lookback")
with alert_col7:
    alert_watch_band  = st.number_input("Watch Band (+)",   min_value=1, max_value=20, value=10, step=1, key="dt_alert_watch",
                                        help="Stocks with RSI between Buy threshold and Buy+Watch are flagged as 'Watch'")

scanner_profit_target = active_strategy['profit_target'] if active_strategy else 3.0
scanner_stop_loss = active_strategy['stop_loss'] if active_strategy else 2.0
scanner_max_hold = active_strategy['max_hold'] if active_strategy else 5
if stock_strategy_map:
    st.caption(
        f"Using stock-specific grid parameters for {len(stock_strategy_map)} tickers; "
        "the global recommendation is used for all others."
    )
elif active_strategy:
    st.caption(
        f"Optimized exit plan: take profit at +{scanner_profit_target:.1f}%, "
        f"stop at -{scanner_stop_loss:.1f}%, or exit after {scanner_max_hold} sessions."
    )

scan_button = st.button("Scan Today's Closing Entries", type="primary", disabled=len(shortlist_tickers) == 0)

def _compute_current_signal(price_series: pd.Series, bear_period: int, bear_buy: int,
                              bull_period: int, bull_buy: int, bull_sma: int,
                              regime: str | None):
    """Return (rsi, trend_ok, entry_fires, bear_rsi_val, bull_rsi_val)."""
    min_len = max(bear_period, bull_period, bull_sma, 35) + 5
    if len(price_series) < min_len:
        return None, False, False, None, None

    bear_rsi = _wilder_rsi(price_series, bear_period)
    bull_rsi = _wilder_rsi(price_series, bull_period)
    sma = price_series.rolling(window=bull_sma, min_periods=bull_sma).mean()

    bear_rsi_val = float(bear_rsi.iloc[-1]) if not pd.isna(bear_rsi.iloc[-1]) else None
    bull_rsi_val = float(bull_rsi.iloc[-1]) if not pd.isna(bull_rsi.iloc[-1]) else None
    above_sma = True
    if bull_sma > 0 and not pd.isna(sma.iloc[-1]):
        above_sma = float(price_series.iloc[-1]) > float(sma.iloc[-1])

    if regime == 'bear':
        main_rsi = bear_rsi_val
        fires = main_rsi is not None and main_rsi < bear_buy
    elif regime == 'bull':
        main_rsi = bull_rsi_val
        fires = main_rsi is not None and main_rsi < bull_buy and above_sma
    else:
        main_rsi = bear_rsi_val
        fires = main_rsi is not None and main_rsi < bear_buy

    return main_rsi, above_sma, fires, bear_rsi_val, bull_rsi_val

if scan_button:
    stock_rsi_periods = [params['rsi_period'] for params in stock_strategy_map.values()]
    stock_trend_smas = [params['trend_sma'] for params in stock_strategy_map.values()]
    required_rows = max(
        int(alert_bear_period), int(alert_bull_period), int(alert_bull_sma),
        max(stock_rsi_periods, default=0), max(stock_trend_smas, default=0), 35,
    ) + 5
    history_days = max(int(alert_lookback), required_rows * 2)
    alert_start = (
        datetime.date.today() - datetime.timedelta(days=history_days)
    ).strftime('%Y-%m-%d')

    with st.spinner("Fetching live prices..."):
        try:
            cp_df = hd.get_company_profile(shortlist_tickers)
        except Exception as e:
            st.warning(f"Could not fetch live prices via get_company_profile: {e}")
            cp_df = pd.DataFrame()

    scan_progress = st.progress(0, text="Scanning shortlist...")
    scan_rows = []

    cached_prices = st.session_state.get('dt_prices', pd.DataFrame())
    if not cached_prices.empty:
        cached_prices = cached_prices.copy()
        cached_prices['date'] = pd.to_datetime(cached_prices['date'], errors='coerce')
        grouped_alert = {
            sym: grp.dropna(subset=['date']).set_index('date')['close'].sort_index()
            for sym, grp in cached_prices.groupby('symbol')
        }
    else:
        grouped_alert = {}

    def _clean_series(series: pd.Series | None) -> pd.Series:
        if series is None or series.empty:
            return pd.Series(dtype=float)
        clean = pd.to_numeric(series, errors='coerce').dropna().copy()
        clean.index = pd.to_datetime(clean.index, errors='coerce')
        clean = clean[~clean.index.isna()]
        return clean[~clean.index.duplicated(keep='last')].sort_index()

    def _series_is_current(series: pd.Series) -> bool:
        if len(series) < required_rows:
            return False
        latest_date = series.index[-1].date()
        return latest_date >= datetime.date.today() - datetime.timedelta(days=7)

    def _scan_series(series: pd.Series, ticker: str, price_source: str):
        series = _clean_series(series)
        history_date = series.index[-1].date()
        stock_params = stock_strategy_map.get(ticker)
        if stock_params:
            bear_period = bull_period = int(stock_params['rsi_period'])
            bear_buy = int(stock_params['bear_buy'])
            bull_buy = int(stock_params['bull_buy'])
            trend_sma = int(stock_params['trend_sma'])
            profit_target = float(stock_params['profit_target'])
            stop_loss = float(stock_params['stop_loss'])
            max_hold = int(stock_params['max_hold'])
            parameter_source = "Stock-specific"
            strategy_score = float(stock_params['score'])
            return_uplift = float(stock_params['return_uplift'])
        else:
            bear_period = int(alert_bear_period)
            bull_period = int(alert_bull_period)
            bear_buy = int(alert_bear_buy)
            bull_buy = int(alert_bull_buy)
            trend_sma = int(alert_bull_sma)
            profit_target = scanner_profit_target
            stop_loss = scanner_stop_loss
            max_hold = scanner_max_hold
            parameter_source = "Global fallback"
            strategy_score = active_strategy['score'] if active_strategy else np.nan
            return_uplift = np.nan

        if ticker in cp_df.index:
            live_price = pd.to_numeric(cp_df.loc[ticker, 'price'], errors='coerce')
            previous_close = float(series.iloc[-1])
            if pd.notna(live_price) and live_price > 0 and 0.65 <= live_price / previous_close <= 1.35:
                today_ts = pd.Timestamp(datetime.date.today()).normalize()
                series.loc[today_ts] = float(live_price)
                series = series.sort_index()
        rsi_val, above_sma, fires, bear_rsi_v, bull_rsi_v = _compute_current_signal(
            series, bear_period, bear_buy,
            bull_period, bull_buy, trend_sma,
            curr_regime
        )
        if rsi_val is not None:
            buy_threshold = bear_buy if curr_regime != 'bull' else bull_buy
            scan_rows.append({
                'Symbol':        ticker,
                'Current Price': float(series.iloc[-1]),
                'History Through': history_date,
                'Price Source':   price_source,
                'Parameter Source': parameter_source,
                'Strategy Score': strategy_score,
                'Return Uplift':  return_uplift,
                'RSI Period':     bear_period if curr_regime != 'bull' else bull_period,
                'Buy RSI <':      buy_threshold,
                'Trend SMA':      trend_sma,
                'RSI':           rsi_val,
                'Bear RSI':      round(bear_rsi_v, 2) if bear_rsi_v is not None else None,
                'Bull RSI':      round(bull_rsi_v, 2) if bull_rsi_v is not None else None,
                'Above SMA':     "Yes" if above_sma else "No",
                'Profit Target': profit_target,
                'Stop Loss':     stop_loss,
                'Max Hold':      max_hold,
                'Entry Fires':   fires,
            })

    stale_skipped = 0
    for i, ticker in enumerate(shortlist_tickers):
        scan_progress.progress((i + 1) / len(shortlist_tickers), text=f"Checking {ticker}...")
        series = _clean_series(grouped_alert.get(ticker))
        price_source = "Optimizer cache"

        if not _series_is_current(series):
            try:
                res = (
                    conn.table("historical_prices")
                        .select("symbol,date,close")
                        .eq("symbol", ticker)
                        .order("date", desc=True)
                        .limit(required_rows + 50)
                        .execute()
                )
                rows = res.data or []
                if rows:
                    series = _clean_series(
                        pd.DataFrame(rows)
                          .assign(date=lambda d: pd.to_datetime(d['date']),
                                  close=lambda d: pd.to_numeric(d['close'], errors='coerce'))
                          .sort_values('date')
                          .set_index('date')['close']
                    )
            except Exception:
                pass
            price_source = "Supabase"

        # Never bridge a stale historical close directly to today's profile
        # price: missing sessions materially distort RSI (for example KLBF).
        if not _series_is_current(series):
            live_history = load_live_daily_prices(ticker, alert_start)
            if not live_history.empty:
                series = _clean_series(live_history.set_index('date')['close'])
                price_source = "Live daily API"

        if _series_is_current(series):
            _scan_series(series, ticker, price_source)
        else:
            stale_skipped += 1

    scan_progress.empty()
    if stale_skipped:
        st.warning(
            f"Skipped {stale_skipped} stock(s) because no sufficiently recent daily history was available."
        )

    if not scan_rows:
        st.warning("Could not evaluate signals for any stock. Try increasing 'Days of History'.")
    else:
        alert_df = pd.DataFrame(scan_rows)

        def _classify(row):
            rsi = row['RSI']
            if row['Entry Fires']:
                return "Enter Today at Close"
            elif rsi < row['Buy RSI <'] + int(alert_watch_band):
                return "Watch"
            else:
                return "Neutral"

        alert_df['Signal'] = alert_df.apply(_classify, axis=1)
        alert_df['RSI'] = alert_df['RSI'].round(2)
        alert_df = alert_df.sort_values('RSI')

        n_enter  = (alert_df['Signal'] == "Enter Today at Close").sum()
        n_watch  = (alert_df['Signal'] == "Watch").sum()
        n_total  = len(alert_df)

        sum_cols = st.columns(3)
        sum_cols[0].metric("Enter Today at Close", n_enter, help="RSI is below its stock-specific buy threshold")
        sum_cols[1].metric("Watch", n_watch, help="RSI is within the configured watch band above its stock-specific threshold")
        sum_cols[2].metric("Stocks Scanned", n_total)

        if n_enter > 0:
            st.success(
                f"**{n_enter} stock(s) firing an entry signal** — enter at today's close; "
                "use each row's target, stop, and maximum hold."
            )

        enter_df = alert_df[alert_df['Signal'] == "Enter Today at Close"]
        watch_df = alert_df[alert_df['Signal'] == "Watch"]
        neutral_df = alert_df[alert_df['Signal'] == "Neutral"]

        col_cfg = {
            "Symbol":        st.column_config.TextColumn("Symbol"),
            "Current Price": st.column_config.NumberColumn("Price", format="%.0f"),
            "History Through": st.column_config.DateColumn("History Through", format="YYYY-MM-DD"),
            "Price Source":   st.column_config.TextColumn("Price Source"),
            "Parameter Source": st.column_config.TextColumn("Parameters"),
            "Strategy Score": st.column_config.ProgressColumn("Stock Strategy Score", format="%.1f", min_value=0, max_value=100),
            "Return Uplift":  st.column_config.NumberColumn("Return vs Global", format="%+.2f%%"),
            "RSI Period":     st.column_config.NumberColumn("RSI Period", format="%d"),
            "Buy RSI <":      st.column_config.NumberColumn("Buy RSI <", format="%d"),
            "Trend SMA":      st.column_config.NumberColumn("Trend SMA", format="%d"),
            "RSI":           st.column_config.ProgressColumn("RSI", help="Current RSI (regime-appropriate period)", format="%.1f", min_value=0, max_value=100),
            "Bear RSI":      st.column_config.NumberColumn("Bear RSI", format="%.1f"),
            "Bull RSI":      st.column_config.NumberColumn("Bull RSI", format="%.1f"),
            "Above SMA":     st.column_config.TextColumn("Above SMA"),
            "Profit Target": st.column_config.NumberColumn("Target", format="%.1f%%"),
            "Stop Loss":     st.column_config.NumberColumn("Stop", format="%.1f%%"),
            "Max Hold":      st.column_config.NumberColumn("Max Hold Days", format="%d"),
            "Signal":        st.column_config.TextColumn("Signal"),
        }

        if not enter_df.empty:
            st.markdown("#### Enter Today at Close")
            st.dataframe(enter_df, column_config=col_cfg, hide_index=True, width='stretch')

        if not watch_df.empty:
            st.markdown("#### Approaching Entry Zone")
            st.dataframe(watch_df, column_config=col_cfg, hide_index=True, width='stretch')

        with st.expander(f"Neutral stocks ({len(neutral_df)})"):
            st.dataframe(neutral_df, column_config=col_cfg, hide_index=True, width='stretch')
