import itertools
import hashlib
import concurrent.futures
import os
import streamlit as st
import datetime
import pandas as pd
import numpy as np
from st_supabase_connection import SupabaseConnection

import harvest.data as hd

STRATEGY_ENGINE_VERSION = 7
DEFAULT_PROFIT_TARGET = 5.0
DEFAULT_STOP_LOSS = 2.0
DEFAULT_MAX_HOLD = 5

STRATEGY_REGIME_RSI = "Regime RSI"
STRATEGY_BOLLINGER_LOWER = "Bollinger Lower-Band Reversion"
STRATEGY_RSI_RECOVERY = "RSI Recovery"
STRATEGY_DONCHIAN = "Donchian Breakout"
STRATEGY_SQUEEZE = "Volatility Squeeze Breakout"
STRATEGY_TREND_PULLBACK = "Trend Pullback"
STRATEGY_RELATIVE_STRENGTH = "Relative-Strength Pullback"
STRATEGY_CROSS_MOMENTUM = "Cross-Sectional Momentum"

STRATEGY_TYPES = [
    STRATEGY_REGIME_RSI,
    STRATEGY_BOLLINGER_LOWER,
    STRATEGY_RSI_RECOVERY,
    STRATEGY_DONCHIAN,
    STRATEGY_SQUEEZE,
    STRATEGY_TREND_PULLBACK,
    STRATEGY_RELATIVE_STRENGTH,
    STRATEGY_CROSS_MOMENTUM,
]

STRATEGY_PARAMETER_LABELS = {
    STRATEGY_REGIME_RSI: ("RSI Period", "Bear Buy <", "Trend SMA", "Bull Buy <"),
    STRATEGY_BOLLINGER_LOWER: ("Band Period", "Std Dev", "RSI Period", "Max RSI"),
    STRATEGY_RSI_RECOVERY: ("RSI Period", "Oversold Level", "Trend SMA", "Recovery Level"),
    STRATEGY_DONCHIAN: ("Breakout Window", "Exit Window", "Trend SMA", "Breakout Buffer %"),
    STRATEGY_SQUEEZE: ("Band Period", "Squeeze Percentile", "Breakout Window", "Bandwidth Lookback"),
    STRATEGY_TREND_PULLBACK: ("Fast EMA", "Slow EMA", "Pullback Tolerance %", "Setup Window"),
    STRATEGY_RELATIVE_STRENGTH: ("RS Lookback", "Min Outperformance %", "RSI Period", "Recovery Level"),
    STRATEGY_CROSS_MOMENTUM: ("Momentum Lookback", "Top Percentile", "RSI Period", "Recovery Level"),
}

STRATEGY_PARAMETER_INPUTS = {
    STRATEGY_REGIME_RSI: ((2.0, 50.0, 1.0), (5.0, 70.0, 1.0), (5.0, 200.0, 5.0), (10.0, 70.0, 1.0)),
    STRATEGY_BOLLINGER_LOWER: ((5.0, 100.0, 5.0), (0.5, 4.0, 0.25), (2.0, 50.0, 1.0), (5.0, 70.0, 1.0)),
    STRATEGY_RSI_RECOVERY: ((2.0, 50.0, 1.0), (5.0, 50.0, 1.0), (0.0, 200.0, 5.0), (10.0, 70.0, 1.0)),
    STRATEGY_DONCHIAN: ((2.0, 200.0, 5.0), (2.0, 100.0, 5.0), (0.0, 200.0, 5.0), (0.0, 5.0, 0.25)),
    STRATEGY_SQUEEZE: ((5.0, 100.0, 5.0), (5.0, 50.0, 5.0), (2.0, 100.0, 5.0), (20.0, 252.0, 10.0)),
    STRATEGY_TREND_PULLBACK: ((2.0, 100.0, 1.0), (5.0, 200.0, 5.0), (0.0, 10.0, 0.5), (1.0, 20.0, 1.0)),
    STRATEGY_RELATIVE_STRENGTH: ((2.0, 252.0, 5.0), (0.0, 30.0, 1.0), (2.0, 50.0, 1.0), (10.0, 70.0, 1.0)),
    STRATEGY_CROSS_MOMENTUM: ((2.0, 252.0, 5.0), (50.0, 100.0, 5.0), (2.0, 50.0, 1.0), (10.0, 70.0, 1.0)),
}


def _load_vectorbt():
    import vectorbt
    return vectorbt

st.set_page_config(page_title='Short-Term Swing Trading - Panen Dividen')
st.title('Short-Term Swing Trading Strategy Lab')

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
    **Short-term swing trading** uses completed daily closes to research 1–10
    session mean-reversion, pullback, relative-strength, and breakout setups.

    * **Mean reversion:** regime RSI, Bollinger lower-band, and RSI recovery.
    * **Trend and momentum:** Donchian, squeeze, EMA pullback, relative strength,
      and cross-sectional momentum.
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


def _align_to_prices(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    clean = pd.to_numeric(series, errors='coerce').dropna().sort_index()
    clean = clean[~clean.index.duplicated(keep='last')]
    return clean.reindex(index, method='ffill')


def _strategy_components(
    close: pd.Series,
    regime: pd.Series,
    benchmark_close: pd.Series | None,
    momentum_percentile: pd.Series | None,
    strategy_type: str,
    param_a: float,
    param_b: float,
    param_c: float,
    param_d: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return closing setup, strategy exit, and a comparable display indicator."""
    if strategy_type == STRATEGY_REGIME_RSI:
        period = int(param_a)
        bear_buy = float(param_b)
        trend_sma = close.rolling(int(param_c), min_periods=int(param_c)).mean()
        rsi = _wilder_rsi(close, period)
        bear_setup = rsi < bear_buy
        bull_setup = (rsi < float(param_d)) & (close > trend_sma)
        setup = pd.Series(
            np.where(regime == 'bear', bear_setup, bull_setup),
            index=close.index,
            dtype=bool,
        )
        strategy_exit = pd.Series(False, index=close.index)
        indicator = rsi
    elif strategy_type == STRATEGY_BOLLINGER_LOWER:
        period = int(param_a)
        middle = close.rolling(period, min_periods=period).mean()
        deviation = close.rolling(period, min_periods=period).std()
        lower = middle - float(param_b) * deviation
        rsi = _wilder_rsi(close, int(param_c))
        setup = (
            (close < lower)
            & (close.shift(1) >= lower.shift(1))
            & (rsi <= float(param_d))
        )
        strategy_exit = (close >= middle) & (close.shift(1) < middle.shift(1))
        indicator = (close / lower - 1) * 100
    elif strategy_type == STRATEGY_RSI_RECOVERY:
        rsi = _wilder_rsi(close, int(param_a))
        was_oversold = (rsi <= float(param_b)).rolling(5, min_periods=1).max().shift(1).fillna(0).astype(bool)
        recovery = (rsi > float(param_d)) & (rsi.shift(1) <= float(param_d))
        setup = was_oversold & recovery
        if int(param_c) > 0:
            trend = close.rolling(int(param_c), min_periods=int(param_c)).mean()
            setup &= close > trend
        strategy_exit = (rsi >= 70) & (rsi.shift(1) < 70)
        indicator = rsi
    elif strategy_type == STRATEGY_DONCHIAN:
        entry_window = int(param_a)
        exit_window = int(param_b)
        prior_high = close.rolling(entry_window, min_periods=entry_window).max().shift(1)
        prior_low = close.rolling(exit_window, min_periods=exit_window).min().shift(1)
        breakout_level = prior_high * (1 + float(param_d) / 100)
        setup = (close > breakout_level) & (close.shift(1) <= breakout_level.shift(1))
        if int(param_c) > 0:
            trend = close.rolling(int(param_c), min_periods=int(param_c)).mean()
            setup &= close > trend
        setup &= regime == 'bull'
        strategy_exit = close < prior_low
        indicator = (close / breakout_level - 1) * 100
    elif strategy_type == STRATEGY_SQUEEZE:
        band_period = int(param_a)
        middle = close.rolling(band_period, min_periods=band_period).mean()
        deviation = close.rolling(band_period, min_periods=band_period).std()
        bandwidth = deviation * 4 / middle * 100
        lookback = int(param_d)
        squeeze_threshold = bandwidth.rolling(lookback, min_periods=lookback).quantile(float(param_b) / 100)
        recent_squeeze = (bandwidth <= squeeze_threshold).rolling(5, min_periods=1).max().shift(1).fillna(0).astype(bool)
        prior_high = close.rolling(int(param_c), min_periods=int(param_c)).max().shift(1)
        setup = recent_squeeze & (close > prior_high) & (close.shift(1) <= prior_high.shift(1))
        setup &= regime == 'bull'
        strategy_exit = close < middle
        indicator = bandwidth
    elif strategy_type == STRATEGY_TREND_PULLBACK:
        fast = close.ewm(span=int(param_a), adjust=False, min_periods=int(param_a)).mean()
        slow = close.ewm(span=int(param_b), adjust=False, min_periods=int(param_b)).mean()
        tolerance = float(param_c) / 100
        touched_fast = (
            (close <= fast * (1 + tolerance))
            .rolling(int(param_d), min_periods=1)
            .max()
            .shift(1)
            .fillna(0)
            .astype(bool)
        )
        setup = touched_fast & (fast > slow) & (close > fast) & (close.shift(1) <= fast.shift(1))
        setup &= regime == 'bull'
        strategy_exit = (fast < slow) | ((close < fast) & (close.shift(1) >= fast.shift(1)))
        indicator = (close / fast - 1) * 100
    elif strategy_type == STRATEGY_RELATIVE_STRENGTH:
        benchmark = _align_to_prices(benchmark_close, close.index)
        stock_return = close.pct_change(int(param_a))
        benchmark_return = benchmark.pct_change(int(param_a))
        relative_strength = (stock_return - benchmark_return) * 100
        rsi = _wilder_rsi(close, int(param_c))
        recovery = (rsi > float(param_d)) & (rsi.shift(1) <= float(param_d))
        setup = (relative_strength >= float(param_b)) & recovery & (regime == 'bull')
        strategy_exit = relative_strength < 0
        indicator = relative_strength
    elif strategy_type == STRATEGY_CROSS_MOMENTUM:
        percentile = _align_to_prices(momentum_percentile, close.index)
        rsi = _wilder_rsi(close, int(param_c))
        recovery = (rsi > float(param_d)) & (rsi.shift(1) <= float(param_d))
        setup = (percentile >= float(param_b)) & recovery & (regime == 'bull')
        strategy_exit = percentile < 50
        indicator = percentile
    else:
        raise ValueError(f"Unknown swing strategy: {strategy_type}")

    return setup.fillna(False), strategy_exit.fillna(False), indicator


def _strategy_warmup(
    strategy_type: str,
    param_a: float,
    param_b: float,
    param_c: float,
    param_d: float,
) -> int:
    if strategy_type == STRATEGY_SQUEEZE:
        return int(param_a) + int(param_d) + 5
    if strategy_type in (STRATEGY_RELATIVE_STRENGTH, STRATEGY_CROSS_MOMENTUM):
        return max(int(param_a), int(param_c), 50) + 5
    if strategy_type == STRATEGY_TREND_PULLBACK:
        return max(int(param_a), int(param_b), 50) + int(param_d) + 5
    if strategy_type == STRATEGY_BOLLINGER_LOWER:
        return max(int(param_a), int(param_c), 50) + 5
    if strategy_type == STRATEGY_RSI_RECOVERY:
        return max(int(param_a), int(param_c), 50) + 10
    return max(int(param_a), int(param_b), int(param_c), 50) + 5


def _build_regime_signals(
    price_series: pd.Series,
    regime_series: pd.Series,
    strategy_type: str,
    param_a: float,
    param_b: float,
    param_c: float,
    param_d: float,
    max_hold: int,
    profit_target_pct: float,
    stop_loss_pct: float,
    trade_start: pd.Timestamp | None = None,
    benchmark_close: pd.Series | None = None,
    momentum_percentile: pd.Series | None = None,
):
    """Build next-session entries and close-based exits from daily closing setups."""
    price_series = pd.to_numeric(price_series, errors='coerce').dropna().sort_index()
    price_series = price_series[~price_series.index.duplicated(keep='last')]
    min_len = _strategy_warmup(
        strategy_type, param_a, param_b, param_c, param_d,
    )
    if len(price_series) < min_len:
        return None

    if regime_series.empty:
        regime_aligned = pd.Series('bear', index=price_series.index, dtype=str)
    else:
        regime_aligned = regime_series.reindex(price_series.index, method='ffill').fillna('bear')

    setup, strategy_exit, _ = _strategy_components(
        price_series, regime_aligned, benchmark_close, momentum_percentile,
        strategy_type, param_a, param_b, param_c, param_d,
    )
    entry_candidates = setup.shift(1, fill_value=False)
    strategy_exit = strategy_exit.shift(1, fill_value=False)
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
            if hit_target or hit_stop or strategy_exit.iloc[i] or days_held >= max_hold:
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
    strategy_type: str,
    param_a: float,
    param_b: float,
    param_c: float,
    param_d: float,
    max_hold: int,
    profit_target_pct: float,
    stop_loss_pct: float,
    buy_fee: float,
    sell_fee: float,
    sell_tax: float,
    trade_start: pd.Timestamp | None = None,
    benchmark_close: pd.Series | None = None,
    momentum_percentile: pd.Series | None = None,
):
    """Run one short-term swing strategy with close-based risk controls."""
    signals = _build_regime_signals(
        price_series, regime_series, strategy_type, param_a, param_b, param_c,
        param_d, max_hold, profit_target_pct, stop_loss_pct, trade_start,
        benchmark_close, momentum_percentile,
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
        vbt = _load_vectorbt()
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
        "strategy_params": {
            STRATEGY_REGIME_RSI: [(14, 30, 50, 45)],
            STRATEGY_BOLLINGER_LOWER: [(20, 2.0, 14, 40)],
            STRATEGY_RSI_RECOVERY: [(14, 30, 0, 35)],
            STRATEGY_DONCHIAN: [(20, 10, 50, 0.0)],
            STRATEGY_SQUEEZE: [(20, 20, 10, 120)],
            STRATEGY_TREND_PULLBACK: [(20, 50, 2.0, 5)],
            STRATEGY_RELATIVE_STRENGTH: [(20, 2.0, 14, 40)],
            STRATEGY_CROSS_MOMENTUM: [(20, 80, 14, 40)],
        },
        "hold_days": [5],
        "exit_rules": [(DEFAULT_PROFIT_TARGET, DEFAULT_STOP_LOSS)],
    },
    "Balanced": {
        "strategy_params": {
            STRATEGY_REGIME_RSI: [(7, 30, 50, 45), (14, 30, 50, 45), (14, 35, 100, 50)],
            STRATEGY_BOLLINGER_LOWER: [(20, 1.5, 14, 45), (20, 2.0, 14, 40)],
            STRATEGY_RSI_RECOVERY: [(7, 25, 0, 35), (14, 30, 50, 40)],
            STRATEGY_DONCHIAN: [(10, 5, 20, 0.0), (20, 10, 50, 0.0)],
            STRATEGY_SQUEEZE: [(20, 20, 10, 120), (20, 30, 20, 120)],
            STRATEGY_TREND_PULLBACK: [(10, 30, 1.5, 3), (20, 50, 2.0, 5)],
            STRATEGY_RELATIVE_STRENGTH: [(20, 0.0, 14, 40), (60, 3.0, 14, 45)],
            STRATEGY_CROSS_MOMENTUM: [(20, 75, 14, 40), (60, 80, 14, 45)],
        },
        "hold_days": [3, 5],
        "exit_rules": [(DEFAULT_PROFIT_TARGET, DEFAULT_STOP_LOSS)],
    },
    "Deep": {
        "strategy_params": {
            STRATEGY_REGIME_RSI: [(7, 25, 20, 40), (7, 30, 50, 45), (14, 30, 50, 45), (14, 35, 100, 50)],
            STRATEGY_BOLLINGER_LOWER: [(10, 1.5, 7, 40), (20, 1.5, 14, 45), (20, 2.0, 14, 40), (30, 2.0, 14, 45)],
            STRATEGY_RSI_RECOVERY: [(7, 25, 0, 35), (14, 25, 50, 35), (14, 30, 50, 40)],
            STRATEGY_DONCHIAN: [(10, 5, 20, 0.0), (20, 10, 50, 0.0), (50, 20, 100, 0.5)],
            STRATEGY_SQUEEZE: [(10, 20, 10, 60), (20, 20, 10, 120), (20, 30, 20, 120)],
            STRATEGY_TREND_PULLBACK: [(5, 20, 1.0, 3), (10, 30, 1.5, 3), (20, 50, 2.0, 5)],
            STRATEGY_RELATIVE_STRENGTH: [(20, 0.0, 7, 40), (20, 2.0, 14, 40), (60, 3.0, 14, 45)],
            STRATEGY_CROSS_MOMENTUM: [(10, 70, 7, 40), (20, 75, 14, 40), (60, 80, 14, 45)],
        },
        "hold_days": [3, 5, 7],
        "exit_rules": [(DEFAULT_PROFIT_TARGET, 1.5), (DEFAULT_PROFIT_TARGET, DEFAULT_STOP_LOSS)],
    },
}
OPTIMIZER_WORKERS = min(8, max(2, os.cpu_count() or 2))


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    valid = frame[[value_col, weight_col]].dropna()
    valid = valid[valid[weight_col] > 0]
    if valid.empty:
        return np.nan
    return float(np.average(valid[value_col], weights=valid[weight_col]))


def _format_strategy_params(
    strategy_type: str,
    param_a: float,
    param_b: float,
    param_c: float,
    param_d: float,
) -> str:
    labels = STRATEGY_PARAMETER_LABELS[strategy_type]
    values = (param_a, param_b, param_c, param_d)
    formatted = []
    for label, value in zip(labels, values):
        display = f"{value:.2f}".rstrip('0').rstrip('.')
        formatted.append(f"{label}: {display}")
    return ", ".join(formatted)


def _scanner_strategy_plan(
    stock_params: dict | None,
    active_params: dict | None,
) -> list[dict]:
    """Build one scanner configuration for every supported strategy."""
    plan = []
    for strategy_type in STRATEGY_TYPES:
        if stock_params and stock_params['strategy_type'] == strategy_type:
            source = stock_params
            parameter_source = "Stock-specific optimized"
            score = float(stock_params['score'])
            uplift = float(stock_params['return_uplift'])
        elif active_params and active_params['strategy_type'] == strategy_type:
            source = active_params
            parameter_source = "Global optimized"
            score = float(active_params['score'])
            uplift = np.nan
        else:
            defaults = SEARCH_PRESETS['Quick']['strategy_params'][strategy_type][0]
            source = {
                'param_a': defaults[0],
                'param_b': defaults[1],
                'param_c': defaults[2],
                'param_d': defaults[3],
                'max_hold': DEFAULT_MAX_HOLD,
                'profit_target': DEFAULT_PROFIT_TARGET,
                'stop_loss': DEFAULT_STOP_LOSS,
            }
            parameter_source = "Baseline preset"
            score = np.nan
            uplift = np.nan
        plan.append({
            'strategy_type': strategy_type,
            'params': tuple(float(source[key]) for key in ('param_a', 'param_b', 'param_c', 'param_d')),
            'profit_target': DEFAULT_PROFIT_TARGET,
            'stop_loss': float(source['stop_loss']),
            'max_hold': int(source['max_hold']),
            'parameter_source': parameter_source,
            'score': score,
            'return_uplift': uplift,
        })
    return plan


def _build_momentum_percentiles(
    grouped_prices: dict[str, pd.Series],
    lookbacks: set[int],
) -> dict[int, pd.DataFrame]:
    """Rank each stock's trailing return against the available universe by date."""
    if not grouped_prices or not lookbacks:
        return {}
    close_frame = pd.concat(grouped_prices, axis=1, sort=False).sort_index()
    return {
        lookback: close_frame.pct_change(lookback).rank(axis=1, pct=True) * 100
        for lookback in lookbacks
    }


def _summarize_strategies(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each parameter set across stocks and score holdout robustness."""
    params = [
        'Strategy', 'Param A', 'Param B', 'Param C', 'Param D',
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
    summary['Parameters'] = summary.apply(
        lambda row: _format_strategy_params(
            row['Strategy'], row['Param A'], row['Param B'], row['Param C'], row['Param D'],
        ),
        axis=1,
    )
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
            (group['Strategy'] == global_strategy['strategy_type'])
            & np.isclose(group['Param A'], global_strategy['param_a'])
            & np.isclose(group['Param B'], global_strategy['param_b'])
            & np.isclose(group['Param C'], global_strategy['param_c'])
            & np.isclose(group['Param D'], global_strategy['param_d'])
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
            'strategy_type': best['Strategy'],
            'param_a': float(best['Param A']),
            'param_b': float(best['Param B']),
            'param_c': float(best['Param C']),
            'param_d': float(best['Param D']),
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
    buy_fee: float,
    sell_fee: float,
    sell_tax: float,
    benchmark_close: pd.Series | None,
    momentum_percentiles: dict[int, pd.Series],
) -> tuple[list[dict], str]:
    """Test every strategy for one stock; safe to execute in a worker thread."""
    if price_series is None or price_series.empty:
        return [], 'missing_data'

    price_series = price_series[~price_series.index.duplicated(keep='last')].sort_index()
    split_idx = int(len(price_series) * (1 - validation_pct / 100))
    indicator_warmup = max(
        _strategy_warmup(combo[0], *combo[1:5]) for combo in param_combos
    )
    if split_idx < indicator_warmup or len(price_series) - split_idx < indicator_warmup:
        return [], 'insufficient_history'

    train_prices = price_series.iloc[:split_idx]
    validation_start = price_series.index[split_idx]
    validation_prices = price_series.iloc[max(0, split_idx - indicator_warmup):]
    rows = []

    for strategy_type, param_a, param_b, param_c, param_d, hold_d, profit_target, stop_loss in param_combos:
        momentum_percentile = momentum_percentiles.get(int(param_a))
        result = _run_regime_backtest(
            train_prices, regime_series,
            strategy_type=strategy_type,
            param_a=param_a, param_b=param_b, param_c=param_c, param_d=param_d,
            max_hold=int(hold_d),
            profit_target_pct=float(profit_target), stop_loss_pct=float(stop_loss),
            buy_fee=buy_fee, sell_fee=sell_fee, sell_tax=sell_tax,
            benchmark_close=benchmark_close, momentum_percentile=momentum_percentile,
        )
        if result is None:
            continue

        tot_ret, win_rate, max_dd, num_trades, avg_trade, n_bear, n_bull = result
        validation_result = _run_regime_backtest(
            validation_prices, regime_series,
            strategy_type=strategy_type,
            param_a=param_a, param_b=param_b, param_c=param_c, param_d=param_d,
            max_hold=int(hold_d),
            profit_target_pct=float(profit_target), stop_loss_pct=float(stop_loss),
            buy_fee=buy_fee, sell_fee=sell_fee, sell_tax=sell_tax,
            trade_start=validation_start,
            benchmark_close=benchmark_close, momentum_percentile=momentum_percentile,
        )
        if validation_result is None:
            val_ret, val_win, val_dd, val_trades, val_avg = np.nan, np.nan, np.nan, 0, np.nan
        else:
            val_ret, val_win, val_dd, val_trades, val_avg, _, _ = validation_result

        rows.append({
            'Symbol': ticker,
            'Strategy': strategy_type,
            'Param A': param_a,
            'Param B': param_b,
            'Param C': param_c,
            'Param D': param_d,
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

with st.expander("Advanced: market regime and trading costs"):
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
    f"{STRATEGY_ENGINE_VERSION}{param_combos}{regime_fast}{regime_slow}{opt_start}{opt_end}{shortlist_tickers}{opt_buy_fee}{opt_sell_fee}{opt_sell_tax}{opt_validation_pct}".encode()
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
        momentum_lookbacks = {
            int(combo[1]) for combo in param_combos
            if combo[0] == STRATEGY_CROSS_MOMENTUM
        }
        momentum_frames = _build_momentum_percentiles(grouped, momentum_lookbacks)
        benchmark_close = jkse_df['close'] if not jkse_df.empty else pd.Series(dtype=float)
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
                    float(opt_buy_fee),
                    float(opt_sell_fee),
                    float(opt_sell_tax),
                    benchmark_close,
                    {
                        lookback: frame[ticker]
                        for lookback, frame in momentum_frames.items()
                        if ticker in frame.columns
                    },
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
            'strategy_type': recommended['Strategy'],
            'param_a': float(recommended['Param A']),
            'param_b': float(recommended['Param B']),
            'param_c': float(recommended['Param C']),
            'param_d': float(recommended['Param D']),
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
                    'Strategy Score', 'Strategy', 'Parameters',
                    'Max Hold', 'Profit Target', 'Stop Loss',
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
                'Strategy Score', 'Strategy', 'Parameters',
                'Max Hold', 'Profit Target', 'Stop Loss',
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
            f"({_format_strategy_params(single_strategy['strategy_type'], single_strategy['param_a'], single_strategy['param_b'], single_strategy['param_c'], single_strategy['param_d'])}); "
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
    cost_cols = st.columns(3)
    bt_buy_fee = cost_cols[0].number_input(
        "Buy Fee %", min_value=0.0, max_value=5.0, step=0.01, key="bt_buy_fee",
    )
    bt_sell_fee = cost_cols[1].number_input(
        "Sell Fee %", min_value=0.0, max_value=5.0, step=0.01, key="bt_sell_fee",
    )
    bt_sell_tax = cost_cols[2].number_input(
        "Sell Tax %", min_value=0.0, max_value=5.0, step=0.01, key="bt_sell_tax",
    )

st.divider()

if st.button('Run Short-Term Swing Backtest', key="dt_run_single") and stock:
    regime_series = compute_regime_series(jkse_df)
    with st.spinner("Fetching data from Supabase..."):
        try:
            res = conn.table("historical_prices").select("date,close").eq("symbol", stock).gte("date", start_date.strftime('%Y-%m-%d')).execute()
            price_df = pd.DataFrame(res.data or [])
        except Exception as e:
            st.error(f"Error fetching data from Supabase: {e}")
            price_df = pd.DataFrame()

    if price_df.empty:
        st.error("No historical data found for the given stock and timeframe.")
    else:
        price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
        price_df['close'] = pd.to_numeric(price_df['close'], errors='coerce')
        close = price_df.dropna().drop_duplicates('date', keep='last').sort_values('date').set_index('date')['close']
        close = close[close.index <= pd.to_datetime(end_date)]
        cached_prices = st.session_state.get('dt_prices', pd.DataFrame())
        momentum_percentile = None
        if bt_strategy_type == STRATEGY_CROSS_MOMENTUM and not cached_prices.empty:
            grouped_single = {
                symbol: group.assign(date=pd.to_datetime(group['date'])).set_index('date')['close']
                for symbol, group in cached_prices.groupby('symbol')
            }
            frame = _build_momentum_percentiles(grouped_single, {int(bt_param_a)}).get(int(bt_param_a))
            if frame is not None and stock in frame:
                momentum_percentile = frame[stock]

        signals = _build_regime_signals(
            close, regime_series, bt_strategy_type, bt_param_a, bt_param_b,
            bt_param_c, bt_param_d, bt_max_hold, bt_profit_target, bt_stop_loss,
            benchmark_close=jkse_df['close'] if not jkse_df.empty else None,
            momentum_percentile=momentum_percentile,
        )
        if signals is None:
            st.warning("Not enough data to calculate the configured indicators.")
        else:
            close, entries, exits, regime_aligned = signals
            if not entries.any():
                detail = " Run the optimizer first to populate universe ranks." if bt_strategy_type == STRATEGY_CROSS_MOMENTUM and momentum_percentile is None else ""
                st.warning(f"No entry signals generated. Try loosening thresholds.{detail}")
            else:
                fees = pd.Series(0.0, index=close.index)
                fees[entries] = bt_buy_fee / 100.0
                fees[exits] = (bt_sell_fee + bt_sell_tax) / 100.0
                vbt = _load_vectorbt()
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
                st.plotly_chart(pf.plot())
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

curr_regime = get_current_regime(full_regime_series)
if curr_regime:
    st.info(f"Current JKSE regime: **{curr_regime.upper()}**. Strategies apply their own regime filters where relevant.")

scanner_cols = st.columns(2)
alert_lookback = scanner_cols[0].number_input("Calendar Days of History", min_value=90, max_value=730, value=365, step=30, key="dt_alert_lookback")
alert_watch_band = scanner_cols[1].number_input("Watch Proximity", min_value=1.0, max_value=20.0, value=5.0, step=1.0, key="dt_alert_watch")

if stock_strategy_map:
    st.caption(
        f"All eight strategies are evaluated. Stock-specific optimized parameters apply to "
        f"{len(stock_strategy_map)} tickers; other strategies use the global recommendation or baseline presets."
    )
elif active_strategy:
    st.caption("All eight strategies are evaluated using the global recommendation where applicable and baseline presets otherwise.")
else:
    st.caption("All eight strategies are evaluated with baseline presets. Run the optimizer to apply validated parameters.")
st.caption(f"Every scanner candidate uses a +{DEFAULT_PROFIT_TARGET:.1f}% profit target.")

scan_button = st.button("Scan Today's Closing Setups", type="primary", disabled=len(shortlist_tickers) == 0)


def _clean_price_series(series: pd.Series | None) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    clean = pd.to_numeric(series, errors='coerce').dropna().copy()
    clean.index = pd.to_datetime(clean.index, errors='coerce')
    clean = clean[~clean.index.isna()]
    return clean[~clean.index.duplicated(keep='last')].sort_index()


def _is_near_setup(strategy_type: str, indicator: float, params: tuple[float, ...], proximity: float) -> bool:
    if not np.isfinite(indicator):
        return False
    _, param_b, _, param_d = params
    if strategy_type == STRATEGY_REGIME_RSI:
        threshold = param_b if curr_regime != 'bull' else param_d
        return indicator <= threshold + proximity
    if strategy_type == STRATEGY_RSI_RECOVERY:
        return param_b <= indicator <= param_d + proximity
    if strategy_type in (STRATEGY_BOLLINGER_LOWER, STRATEGY_DONCHIAN, STRATEGY_TREND_PULLBACK):
        return abs(indicator) <= proximity
    if strategy_type == STRATEGY_RELATIVE_STRENGTH:
        return indicator >= param_b - proximity
    if strategy_type == STRATEGY_CROSS_MOMENTUM:
        return indicator >= param_b - proximity
    return indicator <= proximity


if scan_button:
    scanner_plans = {
        ticker: _scanner_strategy_plan(stock_strategy_map.get(ticker), active_strategy)
        for ticker in shortlist_tickers
    }
    strategy_params_for_warmup = [
        (item['strategy_type'], *item['params'])
        for plan in scanner_plans.values()
        for item in plan
    ]
    required_rows = max(
        _strategy_warmup(strategy_type, *params)
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
    current_series = {}
    price_sources = {}
    stale_skipped = 0
    for index, ticker in enumerate(shortlist_tickers, start=1):
        scan_progress.progress(index / len(shortlist_tickers), text=f"Loading {ticker}...")
        series = _clean_price_series(grouped_alert.get(ticker))
        source = "Optimizer cache"
        if not _series_is_current(series):
            try:
                rows = conn.table("historical_prices").select("symbol,date,close").eq("symbol", ticker).order("date", desc=True).limit(required_rows + 50).execute().data or []
                if rows:
                    frame = pd.DataFrame(rows)
                    frame['date'] = pd.to_datetime(frame['date'], errors='coerce')
                    series = _clean_price_series(frame.sort_values('date').set_index('date')['close'])
            except Exception:
                pass
            source = "Supabase"
        if not _series_is_current(series):
            live_history = load_live_daily_prices(ticker, alert_start)
            if not live_history.empty:
                series = _clean_price_series(live_history.set_index('date')['close'])
                source = "Live daily API"
        if not _series_is_current(series):
            stale_skipped += 1
            continue
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
    momentum_frames = _build_momentum_percentiles(current_series, momentum_lookbacks)
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
            setup, _, indicator_series = _strategy_components(
                series, aligned_regime, benchmark_close, momentum_percentile,
                strategy_type, *params,
            )
            indicator = float(indicator_series.iloc[-1]) if pd.notna(indicator_series.iloc[-1]) else np.nan
            fires = bool(setup.iloc[-1])
            scan_rows.append({
                'Symbol': ticker,
                'Current Price': float(series.iloc[-1]),
                'History Through': series.index[-1].date(),
                'Price Source': price_sources[ticker],
                'Parameter Source': item['parameter_source'],
                'Strategy': strategy_type,
                'Parameters': _format_strategy_params(strategy_type, *params),
                'Indicator': indicator,
                'Strategy Score': item['score'],
                'Return Uplift': item['return_uplift'],
                'Profit Target': item['profit_target'],
                'Stop Loss': item['stop_loss'],
                'Max Hold': item['max_hold'],
                'Setup Fires': fires,
                'Near Setup': _is_near_setup(strategy_type, indicator, params, float(alert_watch_band)),
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
        alert_df = alert_df.sort_values(['Signal', 'Strategy Score', 'Indicator'], ascending=[True, False, False])
        confirmed_df = alert_df[alert_df['Signal'] == "Setup Confirmed"]
        watch_df = alert_df[alert_df['Signal'] == "Watch"]
        neutral_df = alert_df[alert_df['Signal'] == "Neutral"]
        summary_cols = st.columns(3)
        summary_cols[0].metric("Setups Confirmed", len(confirmed_df), help="Completed-close setups modeled for next-session entry")
        summary_cols[1].metric("Watch", len(watch_df))
        summary_cols[2].metric("Strategy Checks", len(alert_df), help=f"{len(current_series)} stocks x {len(STRATEGY_TYPES)} strategies")
        if not confirmed_df.empty:
            st.success(f"{len(confirmed_df)} setup(s) confirmed at the latest close for next-session execution research.")

        scanner_cfg = {
            "Current Price": st.column_config.NumberColumn("Price", format="%.0f"),
            "History Through": st.column_config.DateColumn("History Through", format="YYYY-MM-DD"),
            "Indicator": st.column_config.NumberColumn("Current Indicator", format="%.2f"),
            "Strategy Score": st.column_config.ProgressColumn("Strategy Score", format="%.1f", min_value=0, max_value=100),
            "Return Uplift": st.column_config.NumberColumn("Return vs Global", format="%+.2f%%"),
            "Profit Target": st.column_config.NumberColumn("Target", format="%.1f%%"),
            "Stop Loss": st.column_config.NumberColumn("Stop", format="%.1f%%"),
            "Max Hold": st.column_config.NumberColumn("Max Hold", format="%d sessions"),
        }
        if not confirmed_df.empty:
            st.markdown("#### Setup Confirmed at Latest Close")
            st.dataframe(confirmed_df, column_config=scanner_cfg, hide_index=True, width='stretch')
        if not watch_df.empty:
            st.markdown("#### Approaching Setup")
            st.dataframe(watch_df, column_config=scanner_cfg, hide_index=True, width='stretch')
        with st.expander(f"Neutral stocks ({len(neutral_df)})"):
            st.dataframe(neutral_df, column_config=scanner_cfg, hide_index=True, width='stretch')
