"""Short-term swing trading engine: strategies, regime, costs, backtest and gates.

Pure logic shared by the Swing Trading page and the pipeline jobs. Strategies
work on daily closes: a setup completed at one close enters at the next
session's close, and exits on profit target, stop loss, the strategy's own
exit or the maximum hold, all checked on closes.

Nothing here depends on Streamlit.
"""
import numpy as np
import pandas as pd


STRATEGY_ENGINE_VERSION = 10
DEFAULT_PROFIT_TARGET = 5.0
DEFAULT_STOP_LOSS = 2.0
DEFAULT_MAX_HOLD = 5
DEFAULT_SLIPPAGE_TICKS = 0.5

# Absolute holdout gates a strategy must pass before it can be recommended.
# Percentile scores only rank candidates; these stop a "best of the losers"
# from reaching the scanner.
MIN_PROFIT_FACTOR = 1.2
MIN_T_STAT = 2.0
# Stock-specific overrides need enough holdout trades to be more than luck.
MIN_STOCK_VALIDATION_TRADES = 10

STRATEGY_REGIME_RSI = "Regime RSI"
STRATEGY_BOLLINGER_LOWER = "Bollinger Lower-Band Reversion"
STRATEGY_RSI_RECOVERY = "RSI Recovery"
STRATEGY_DONCHIAN = "Donchian Breakout"
STRATEGY_SQUEEZE = "Volatility Squeeze Breakout"
STRATEGY_TREND_PULLBACK = "Trend Pullback"
STRATEGY_RELATIVE_STRENGTH = "Relative-Strength Pullback"
STRATEGY_CROSS_MOMENTUM = "Cross-Sectional Momentum"
STRATEGY_SUPERTREND_PULLBACK = "Close-Based Supertrend Pullback"
STRATEGY_BREADTH_RSI2 = "Breadth Capitulation RSI(2)"

STRATEGY_TYPES = [
    STRATEGY_REGIME_RSI,
    STRATEGY_BOLLINGER_LOWER,
    STRATEGY_RSI_RECOVERY,
    # STRATEGY_DONCHIAN,
    # STRATEGY_SQUEEZE,
    # STRATEGY_TREND_PULLBACK,
    # STRATEGY_RELATIVE_STRENGTH,
    # STRATEGY_CROSS_MOMENTUM,
    STRATEGY_SUPERTREND_PULLBACK,
    STRATEGY_BREADTH_RSI2,
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
    STRATEGY_SUPERTREND_PULLBACK: ("ATR Period", "ATR Multiplier", "Pullback Drop %", "Support Proximity %"),
    STRATEGY_BREADTH_RSI2: ("RSI Period", "RSI Buy <", "Breadth Max %", "Trend SMA"),
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
    STRATEGY_SUPERTREND_PULLBACK: ((2.0, 50.0, 1.0), (0.5, 10.0, 0.25), (0.25, 10.0, 0.25), (0.0, 10.0, 0.25)),
    STRATEGY_BREADTH_RSI2: ((2.0, 14.0, 1.0), (1.0, 40.0, 1.0), (1.0, 60.0, 1.0), (0.0, 250.0, 10.0)),
}


def load_vectorbt():
    import vectorbt
    return vectorbt


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



def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up   = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up   = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    ema_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs  = ema_up / ema_down
    return 100 - (100 / (1 + rs))


def close_supertrend(
    close: pd.Series,
    period: int,
    multiplier: float,
) -> tuple[pd.Series, pd.Series]:
    """Calculate Supertrend bands when only daily closes are available."""
    true_range = close.diff().abs()
    atr = true_range.ewm(
        alpha=1 / period, adjust=False, min_periods=period,
    ).mean()
    close_values = close.to_numpy(dtype=float)
    atr_values = atr.to_numpy(dtype=float)
    basic_upper = close_values + multiplier * atr_values
    basic_lower = close_values - multiplier * atr_values
    final_upper = np.full(len(close), np.nan)
    final_lower = np.full(len(close), np.nan)
    direction = np.zeros(len(close), dtype=np.int8)

    for i in range(len(close)):
        if not np.isfinite(atr_values[i]):
            continue
        if i == 0 or not np.isfinite(final_upper[i - 1]):
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
            direction[i] = -1
            continue

        previous_close = close_values[i - 1]
        previous_upper = final_upper[i - 1]
        previous_lower = final_lower[i - 1]
        final_upper[i] = (
            basic_upper[i]
            if basic_upper[i] < previous_upper or previous_close > previous_upper
            else previous_upper
        )
        final_lower[i] = (
            basic_lower[i]
            if basic_lower[i] > previous_lower or previous_close < previous_lower
            else previous_lower
        )

        if direction[i - 1] == 1:
            direction[i] = -1 if close_values[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close_values[i] > final_upper[i] else -1

    supertrend = np.where(direction == 1, final_lower, final_upper)
    return (
        pd.Series(supertrend, index=close.index),
        pd.Series(direction, index=close.index),
    )


def dividend_adjust(close: pd.Series, dividends: pd.DataFrame | None) -> pd.Series:
    """Back-adjust closes for cash dividends so ex-dates are not price drops.

    Uses the standard ratio method: every close before an ex-date is scaled by
    ``1 - dividend / previous close``. Closes after the last ex-date are left
    unchanged, so live prices can still be appended to the series.
    """
    if dividends is None or dividends.empty or close.empty:
        return close
    factor = pd.Series(1.0, index=close.index)
    for ex_date, amount in zip(dividends['date'], dividends['dividend']):
        before = close.index < ex_date
        # Skip ex-dates outside the series: no drop is visible in the data.
        if not before.any() or before.all():
            continue
        previous_close = close[before].iloc[-1]
        if not np.isfinite(previous_close) or previous_close <= 0:
            continue
        ratio = 1 - float(amount) / float(previous_close)
        # Ignore implausible events (bad data or special distributions).
        if 0.5 < ratio < 1:
            factor[before] *= ratio
    return close * factor


def idx_tick_size(prices: pd.Series | np.ndarray) -> np.ndarray:
    """IDX fraction (tick size) for each price level."""
    values = np.asarray(prices, dtype=float)
    return np.select(
        [values < 200, values < 500, values < 2000, values < 5000],
        [1.0, 2.0, 5.0, 10.0],
        default=25.0,
    )


def slippage_fraction(prices: pd.Series, ticks: float) -> pd.Series:
    """Cost of crossing ``ticks`` price steps, as a fraction of price."""
    if ticks <= 0:
        return pd.Series(0.0, index=prices.index)
    values = prices.to_numpy(dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction = np.where(values > 0, ticks * idx_tick_size(values) / values, 0.0)
    return pd.Series(fraction, index=prices.index)


def build_trade_fees(
    price_series: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    buy_fee: float,
    sell_fee: float,
    sell_tax: float,
    slippage_ticks: float,
) -> pd.Series:
    """Per-bar fee fractions: broker fees, tax, and tick-based slippage."""
    slippage = slippage_fraction(price_series, slippage_ticks)
    fees = pd.Series(0.0, index=price_series.index)
    fees[entries] = buy_fee / 100.0 + slippage[entries]
    fees[exits] = (sell_fee + sell_tax) / 100.0 + slippage[exits]
    return fees


def build_market_breadth(
    grouped_prices: dict[str, pd.Series],
    window: int = 20,
    min_stocks: int = 10,
) -> pd.Series:
    """Percent of the universe closing above its ``window``-day average."""
    if not grouped_prices:
        return pd.Series(dtype=float)
    close_frame = pd.concat(grouped_prices, axis=1, sort=False).sort_index()
    # Bridge short suspensions so one missing day doesn't void a 20-day average.
    close_frame = close_frame.ffill(limit=5)
    average = close_frame.rolling(window, min_periods=window).mean()
    valid = average.notna() & close_frame.notna()
    above = (close_frame > average) & valid
    counts = valid.sum(axis=1)
    breadth = above.sum(axis=1) / counts.where(counts > 0) * 100
    return breadth.where(counts >= min_stocks).dropna()


def _align_to_prices(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    clean = pd.to_numeric(series, errors='coerce').dropna().sort_index()
    clean = clean[~clean.index.duplicated(keep='last')]
    return clean.reindex(index, method='ffill')


def strategy_components(
    close: pd.Series,
    regime: pd.Series,
    benchmark_close: pd.Series | None,
    momentum_percentile: pd.Series | None,
    strategy_type: str,
    param_a: float,
    param_b: float,
    param_c: float,
    param_d: float,
    breadth: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return closing setup, strategy exit, and a comparable display indicator."""
    if strategy_type == STRATEGY_REGIME_RSI:
        period = int(param_a)
        bear_buy = float(param_b)
        trend_sma = close.rolling(int(param_c), min_periods=int(param_c)).mean()
        rsi = wilder_rsi(close, period)
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
        rsi = wilder_rsi(close, int(param_c))
        setup = (
            (close < lower)
            & (close.shift(1) >= lower.shift(1))
            & (rsi <= float(param_d))
        )
        strategy_exit = (close >= middle) & (close.shift(1) < middle.shift(1))
        indicator = (close / lower - 1) * 100
    elif strategy_type == STRATEGY_RSI_RECOVERY:
        rsi = wilder_rsi(close, int(param_a))
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
        rsi = wilder_rsi(close, int(param_c))
        recovery = (rsi > float(param_d)) & (rsi.shift(1) <= float(param_d))
        setup = (relative_strength >= float(param_b)) & recovery & (regime == 'bull')
        strategy_exit = relative_strength < 0
        indicator = relative_strength
    elif strategy_type == STRATEGY_CROSS_MOMENTUM:
        percentile = _align_to_prices(momentum_percentile, close.index)
        rsi = wilder_rsi(close, int(param_c))
        recovery = (rsi > float(param_d)) & (rsi.shift(1) <= float(param_d))
        setup = (percentile >= float(param_b)) & recovery & (regime == 'bull')
        strategy_exit = percentile < 50
        indicator = percentile
    elif strategy_type == STRATEGY_SUPERTREND_PULLBACK:
        supertrend, direction = close_supertrend(
            close, int(param_a), float(param_b),
        )
        support_distance = (close / supertrend - 1) * 100
        bullish = direction == 1
        established_bullish = (
            bullish
            & bullish.shift(1, fill_value=False)
            & bullish.shift(2, fill_value=False)
        )
        price_drop = close.pct_change() * 100 <= -float(param_c)
        reached_support = support_distance <= float(param_d)
        eligible_support = established_bullish & reached_support
        fresh_support_touch = eligible_support & ~eligible_support.shift(1, fill_value=False)
        setup = (established_bullish & price_drop) | fresh_support_touch
        strategy_exit = direction == -1
        indicator = support_distance
    elif strategy_type == STRATEGY_BREADTH_RSI2:
        # Buy the strongest names on a very short-term dip, but only when the
        # whole market is washed out (few stocks above their 20-day average).
        market_breadth = _align_to_prices(breadth, close.index)
        rsi = wilder_rsi(close, int(param_a))
        setup = (rsi <= float(param_b)) & (market_breadth <= float(param_c))
        if int(param_d) > 0:
            trend = close.rolling(int(param_d), min_periods=int(param_d)).mean()
            setup &= close > trend
        # Mean reversion is complete once price recovers its 5-day average.
        strategy_exit = close > close.rolling(5, min_periods=5).mean()
        indicator = market_breadth
    else:
        raise ValueError(f"Unknown swing strategy: {strategy_type}")

    return setup.fillna(False), strategy_exit.fillna(False), indicator


def strategy_warmup(
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
    if strategy_type == STRATEGY_SUPERTREND_PULLBACK:
        return max(int(param_a) + 5, 20)
    if strategy_type == STRATEGY_BREADTH_RSI2:
        return max(int(param_a), int(param_d), 20) + 5
    return max(int(param_a), int(param_b), int(param_c), 50) + 5


def build_regime_signals(
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
    breadth: pd.Series | None = None,
):
    """Build next-session entries and close-based exits from daily closing setups."""
    price_series = pd.to_numeric(price_series, errors='coerce').dropna().sort_index()
    price_series = price_series[~price_series.index.duplicated(keep='last')]
    min_len = strategy_warmup(
        strategy_type, param_a, param_b, param_c, param_d,
    )
    if len(price_series) < min_len:
        return None

    if regime_series.empty:
        regime_aligned = pd.Series('bear', index=price_series.index, dtype=str)
    else:
        regime_aligned = regime_series.reindex(price_series.index, method='ffill').fillna('bear')

    setup, strategy_exit, _ = strategy_components(
        price_series, regime_aligned, benchmark_close, momentum_percentile,
        strategy_type, param_a, param_b, param_c, param_d, breadth=breadth,
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


def run_regime_backtest(
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
    breadth: pd.Series | None = None,
    slippage_ticks: float = 0.0,
):
    """Run one short-term swing strategy with close-based risk controls.

    Returns a dict of performance metrics, or None when no trade happened.
    """
    signals = build_regime_signals(
        price_series, regime_series, strategy_type, param_a, param_b, param_c,
        param_d, max_hold, profit_target_pct, stop_loss_pct, trade_start,
        benchmark_close, momentum_percentile, breadth,
    )
    if signals is None:
        return None
    price_series, entries, exits, regime_aligned = signals

    if not entries.any():
        return None

    fees = build_trade_fees(
        price_series, entries, exits, buy_fee, sell_fee, sell_tax, slippage_ticks,
    )

    try:
        vbt = load_vectorbt()
        pf = vbt.Portfolio.from_signals(price_series, entries, exits, freq='1D', fees=fees)
        num_trades = len(pf.trades)
        if num_trades == 0:
            return None
        trade_returns = np.asarray(pf.trades.returns.values, dtype=float) * 100
        return {
            'total_return': float(pf.total_return() * 100),
            'win_rate': float(pf.trades.win_rate() * 100),
            'max_drawdown': float(pf.max_drawdown() * 100),
            'trades': num_trades,
            'avg_trade': float(trade_returns.mean()),
            # Sums let trade statistics be pooled across stocks later.
            'gross_win': float(trade_returns[trade_returns > 0].sum()),
            'gross_loss': float(-trade_returns[trade_returns < 0].sum()),
            'sum_sq': float((trade_returns ** 2).sum()),
            'bear_entries': int((entries & (regime_aligned == 'bear')).sum()),
            'bull_entries': int((entries & (regime_aligned == 'bull')).sum()),
        }
    except Exception:
        return None


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
            STRATEGY_SUPERTREND_PULLBACK: [(10, 3.0, 2.0, 0.5)],
            STRATEGY_BREADTH_RSI2: [(2, 10, 20, 200)],
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
            STRATEGY_SUPERTREND_PULLBACK: [(10, 2.5, 1.5, 0.5), (10, 3.0, 2.0, 1.0)],
            STRATEGY_BREADTH_RSI2: [(2, 10, 20, 200), (2, 5, 15, 100)],
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
            STRATEGY_SUPERTREND_PULLBACK: [(7, 2.0, 1.5, 0.5), (10, 2.5, 2.0, 0.75), (10, 3.0, 2.5, 1.0)],
            STRATEGY_BREADTH_RSI2: [(2, 5, 15, 200), (2, 10, 20, 200), (3, 15, 25, 200), (2, 10, 20, 100)],
        },
        "hold_days": [3, 5, 7],
        "exit_rules": [(DEFAULT_PROFIT_TARGET, 1.5), (DEFAULT_PROFIT_TARGET, DEFAULT_STOP_LOSS)],
    },
}


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    valid = frame[[value_col, weight_col]].dropna()
    valid = valid[valid[weight_col] > 0]
    if valid.empty:
        return np.nan
    return float(np.average(valid[value_col], weights=valid[weight_col]))


def format_strategy_params(
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


def scanner_strategy_plan(
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
            expected_trade = float(stock_params['validation_avg_trade'])
            validated = True
        elif active_params and active_params['strategy_type'] == strategy_type:
            source = active_params
            parameter_source = "Global optimized"
            score = float(active_params['score'])
            uplift = np.nan
            expected_trade = float(active_params.get('validation_avg_trade', np.nan))
            validated = True
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
            expected_trade = np.nan
            validated = False
        plan.append({
            'strategy_type': strategy_type,
            'params': tuple(float(source[key]) for key in ('param_a', 'param_b', 'param_c', 'param_d')),
            'profit_target': float(source['profit_target']),
            'stop_loss': float(source['stop_loss']),
            'max_hold': int(source['max_hold']),
            'parameter_source': parameter_source,
            'score': score,
            'return_uplift': uplift,
            'expected_trade': expected_trade,
            'validated': validated,
        })
    return plan


def build_momentum_percentiles(
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


def pooled_trade_stats(frame: pd.DataFrame, prefix: str) -> tuple[float, float]:
    """Profit factor and t-stat of the mean trade, pooled across stocks."""
    trades = frame[f'{prefix}Trades'].sum()
    if trades <= 0:
        return np.nan, np.nan
    gross_win = frame[f'{prefix}Gross Win'].sum()
    gross_loss = frame[f'{prefix}Gross Loss'].sum()
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (np.inf if gross_win > 0 else np.nan)
    mean = (gross_win - gross_loss) / trades
    if trades < 2:
        return float(profit_factor), np.nan
    variance = (frame[f'{prefix}Sum Sq'].sum() - trades * mean ** 2) / (trades - 1)
    if variance <= 0:
        return float(profit_factor), np.nan
    t_stat = mean / np.sqrt(variance / trades)
    return float(profit_factor), float(t_stat)


def summarize_strategies(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each parameter set across stocks and score holdout robustness."""
    params = [
        'Strategy', 'Param A', 'Param B', 'Param C', 'Param D',
        'Max Hold', 'Profit Target', 'Stop Loss',
    ]
    rows = []
    universe_size = max(1, results_df['Symbol'].nunique())
    for values, group in results_df.groupby(params, dropna=False):
        validated = group[group['Validation Trades'] > 0]
        profit_factor, t_stat = pooled_trade_stats(validated, 'Validation ')
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
            'Validation Profit Factor': profit_factor,
            'Validation t-stat': t_stat,
            'Median Validation Return': float(validated['Validation Return'].median()) if not validated.empty else np.nan,
            'Positive Stocks %': float((validated['Validation Return'] > 0).mean() * 100) if not validated.empty else np.nan,
            'Validation Drawdown': _weighted_mean(validated, 'Validation Drawdown', 'Validation Trades'),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary['Parameters'] = summary.apply(
        lambda row: format_strategy_params(
            row['Strategy'], row['Param A'], row['Param B'], row['Param C'], row['Param D'],
        ),
        axis=1,
    )
    summary['Generalization Gap'] = summary['Validation Avg Trade'] - summary['Train Avg Trade']
    summary['Passes Gates'] = passes_profitability_gates(summary)
    score_inputs = {
        'Validation Avg Trade': 0.25,
        'Validation Profit Factor': 0.15,
        'Validation Win Rate': 0.10,
        'Median Validation Return': 0.15,
        'Positive Stocks %': 0.15,
        'Validation Coverage %': 0.10,
    }
    summary['Strategy Score'] = 0.0
    for column, weight in score_inputs.items():
        summary['Strategy Score'] += summary[column].rank(pct=True).fillna(0) * weight * 100
    drawdown_control = (-summary['Validation Drawdown'].abs()).rank(pct=True).fillna(0) * 100
    summary['Strategy Score'] += drawdown_control * 0.10
    return summary.sort_values(
        ['Passes Gates', 'Strategy Score', 'Validation Avg Trade', 'Validation Trades'],
        ascending=False,
    ).reset_index(drop=True)


def passes_profitability_gates(summary: pd.DataFrame) -> pd.Series:
    """Absolute, after-cost holdout requirements for a recommendable strategy."""
    return (
        (summary['Validation Avg Trade'] > 0)
        & (summary['Train Avg Trade'] > 0)
        & (summary['Validation Profit Factor'] >= MIN_PROFIT_FACTOR)
        & (summary['Validation t-stat'] >= MIN_T_STAT)
    ).fillna(False)


def _score_stock_strategy_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Score strategy configurations relative to the candidates for one stock."""
    candidates = candidates.copy()
    score_weights = {
        'Validation Avg Trade': 0.35,
        'Validation Win Rate': 0.20,
        'Validation Return': 0.20,
        'Validation Trades': 0.15,
    }
    candidates['Stock Strategy Score'] = 0.0
    for column, weight in score_weights.items():
        candidates['Stock Strategy Score'] += (
            candidates[column].rank(pct=True).fillna(0) * weight * 100
        )
    drawdown_control = (
        -candidates['Validation Drawdown'].abs()
    ).rank(pct=True).fillna(0) * 100
    candidates['Stock Strategy Score'] += drawdown_control * 0.10
    return candidates


def eligible_stock_candidates(group: pd.DataFrame) -> pd.DataFrame:
    """Per-stock configurations with enough holdout trades and an edge in both periods.

    Requiring a positive training average as well as a positive holdout average
    stops the holdout from simply rewarding the luckiest configuration.
    """
    return group[
        (group['Validation Trades'] >= MIN_STOCK_VALIDATION_TRADES)
        & (group['Validation Avg Trade'] > 0)
        & (group['Avg Trade'] > 0)
    ]


def select_stock_strategies(
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

        candidates = eligible_stock_candidates(group)
        candidates = candidates[candidates['Validation Return'] > global_return + 1e-9].copy()
        if candidates.empty:
            continue

        candidates = _score_stock_strategy_candidates(candidates)
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


def best_stock_strategy_rows(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return each stock's highest-scoring configuration on holdout data."""
    best_rows = []
    for _, group in results_df.groupby('Symbol'):
        candidates = eligible_stock_candidates(group).copy()
        if candidates.empty:
            continue

        candidates = _score_stock_strategy_candidates(candidates)
        best_rows.append(candidates.sort_values(
            ['Stock Strategy Score', 'Validation Trades', 'Validation Avg Trade'],
            ascending=False,
        ).iloc[0])

    if not best_rows:
        return pd.DataFrame()
    best_df = pd.DataFrame(best_rows).reset_index(drop=True)
    best_df['Parameters'] = best_df.apply(
        lambda row: format_strategy_params(
            row['Strategy'], row['Param A'], row['Param B'],
            row['Param C'], row['Param D'],
        ),
        axis=1,
    )
    return best_df.sort_values(
        ['Stock Strategy Score', 'Validation Avg Trade'], ascending=False,
    ).reset_index(drop=True)


def evaluate_stock_strategies(
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
    breadth: pd.Series | None = None,
    slippage_ticks: float = 0.0,
) -> tuple[list[dict], str]:
    """Test every strategy for one stock; safe to execute in a worker thread."""
    if price_series is None or price_series.empty:
        return [], 'missing_data'

    price_series = price_series[~price_series.index.duplicated(keep='last')].sort_index()
    split_idx = int(len(price_series) * (1 - validation_pct / 100))
    indicator_warmup = max(
        strategy_warmup(combo[0], *combo[1:5]) for combo in param_combos
    )
    if split_idx < indicator_warmup or len(price_series) - split_idx < indicator_warmup:
        return [], 'insufficient_history'

    train_prices = price_series.iloc[:split_idx]
    validation_start = price_series.index[split_idx]
    validation_prices = price_series.iloc[max(0, split_idx - indicator_warmup):]
    rows = []

    for strategy_type, param_a, param_b, param_c, param_d, hold_d, profit_target, stop_loss in param_combos:
        backtest_kwargs = dict(
            strategy_type=strategy_type,
            param_a=param_a, param_b=param_b, param_c=param_c, param_d=param_d,
            max_hold=int(hold_d),
            profit_target_pct=float(profit_target), stop_loss_pct=float(stop_loss),
            buy_fee=buy_fee, sell_fee=sell_fee, sell_tax=sell_tax,
            benchmark_close=benchmark_close,
            momentum_percentile=momentum_percentiles.get(int(param_a)),
            breadth=breadth,
            slippage_ticks=slippage_ticks,
        )
        train = run_regime_backtest(train_prices, regime_series, **backtest_kwargs)
        if train is None:
            continue

        validation = run_regime_backtest(
            validation_prices, regime_series, trade_start=validation_start, **backtest_kwargs,
        )
        if validation is None:
            validation = {
                'total_return': np.nan, 'win_rate': np.nan, 'max_drawdown': np.nan,
                'trades': 0, 'avg_trade': np.nan,
                'gross_win': 0.0, 'gross_loss': 0.0, 'sum_sq': 0.0,
            }

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
            'Total Return': train['total_return'],
            'Avg Trade': train['avg_trade'],
            'Win Rate': train['win_rate'],
            'Max Drawdown': train['max_drawdown'],
            '# Trades': train['trades'],
            'Bear Entries': train['bear_entries'],
            'Bull Entries': train['bull_entries'],
            'Validation Return': validation['total_return'],
            'Validation Win Rate': validation['win_rate'],
            'Validation Drawdown': validation['max_drawdown'],
            'Validation Trades': validation['trades'],
            'Validation Avg Trade': validation['avg_trade'],
            'Validation Gross Win': validation['gross_win'],
            'Validation Gross Loss': validation['gross_loss'],
            'Validation Sum Sq': validation['sum_sq'],
        })

    return rows, 'ok' if rows else 'no_signals'


def clean_price_series(series: pd.Series | None) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    clean = pd.to_numeric(series, errors='coerce').dropna().copy()
    clean.index = pd.to_datetime(clean.index, errors='coerce')
    clean = clean[~clean.index.isna()]
    return clean[~clean.index.duplicated(keep='last')].sort_index()


def is_near_setup(
    strategy_type: str,
    indicator: float,
    params: tuple[float, ...],
    proximity: float,
    trend_ok: bool = True,
    regime: str | None = None,
) -> bool:
    """Whether a setup is close to firing.

    ``trend_ok`` is the strategy's trend filter and ``regime`` the current
    market regime ('bull' or 'bear').
    """
    if not np.isfinite(indicator):
        return False
    _, param_b, param_c, param_d = params
    if strategy_type == STRATEGY_REGIME_RSI:
        if regime != 'bull':
            return indicator <= param_b + proximity
        # Bull-regime entries also require price above the trend SMA.
        return trend_ok and indicator <= param_d + proximity
    if strategy_type == STRATEGY_BREADTH_RSI2:
        return trend_ok and indicator <= param_c + proximity
    if strategy_type == STRATEGY_RSI_RECOVERY:
        return param_b <= indicator <= param_d + proximity
    if strategy_type in (STRATEGY_BOLLINGER_LOWER, STRATEGY_DONCHIAN, STRATEGY_TREND_PULLBACK):
        return abs(indicator) <= proximity
    if strategy_type == STRATEGY_SUPERTREND_PULLBACK:
        watch_width = min(proximity, 1.5)
        return param_d < indicator <= param_d + watch_width
    if strategy_type == STRATEGY_RELATIVE_STRENGTH:
        return indicator >= param_b - proximity
    if strategy_type == STRATEGY_CROSS_MOMENTUM:
        return indicator >= param_b - proximity
    return indicator <= proximity
