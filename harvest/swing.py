"""Short-term swing trading engine: strategies, regime, costs, backtest and gates.

Pure logic shared by the Swing Trading page and the pipeline jobs. Strategies
work on daily closes: a setup completed at one close enters at the next
session's close, and exits on profit target, stop loss, the strategy's own
exit or the maximum hold, all checked on closes.

Nothing here depends on Streamlit.
"""
import datetime
from statistics import NormalDist

import numpy as np
import pandas as pd


STRATEGY_ENGINE_VERSION = 11
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


TRADE_COLUMNS = [
    'entry_date', 'entry_price', 'exit_date', 'exit_price', 'exit_reason', 'sessions',
    'return_pct', 'regime',
]


def ohlc_frame(prices: pd.Series | pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Clean ``open, high, low, close`` by date, and whether real OHLC data was given.

    A close series (or a frame without opens) becomes a frame whose open, high
    and low equal the close, which keeps the original close-only rules.
    """
    if isinstance(prices, pd.Series):
        close = pd.to_numeric(prices, errors='coerce')
        frame = pd.DataFrame({'close': close})
    else:
        frame = prices.copy()
        if 'date' in frame.columns:
            frame = frame.set_index('date')
        frame = frame.apply(pd.to_numeric, errors='coerce')
    frame = frame[frame['close'].notna()].sort_index()
    frame = frame[~frame.index.duplicated(keep='last')]
    has_ohlc = 'open' in frame.columns and frame['open'].notna().any()
    for column in ('open', 'high', 'low'):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = frame[column].fillna(frame['close'])
    # A few vendor bars have a high/low that doesn't bracket the open and close.
    bars = frame[['open', 'high', 'low', 'close']]
    frame['high'], frame['low'] = bars.max(axis=1), bars.min(axis=1)
    return frame[['open', 'high', 'low', 'close']], bool(has_ohlc)


def simulate_trades(
    frame: pd.DataFrame,
    setup: pd.Series,
    strategy_exit: pd.Series,
    max_hold: int,
    profit_target_pct: float,
    stop_loss_pct: float,
    trade_start: pd.Timestamp | None = None,
    has_ohlc: bool = True,
) -> list[dict]:
    """Walk the bars and return each trade's entry and exit bar, prices and reason.

    A setup at close *t* enters at the open of *t+1*. With OHLC data a stop or
    target fills at its level, or at the open when the price gaps through it;
    the stop wins when both are touched in one bar, and only the stop is
    checked on the entry day. A strategy exit signalled at a close fills at the
    next open. The position is closed at the close ``max_hold`` sessions after
    the entry day at the latest.

    With close-only data every fill uses the close, as before OHLC existed.
    """
    entry_candidates = np.array(setup.shift(1, fill_value=False), dtype=bool)
    exit_signal = np.array(strategy_exit.shift(1, fill_value=False), dtype=bool)
    if trade_start is not None:
        entry_candidates &= (frame.index >= pd.Timestamp(trade_start))
    opens, highs, lows, closes = (frame[c].to_numpy(dtype=float) for c in ('open', 'high', 'low', 'close'))
    trades = []
    in_pos = False
    entry_i, entry_price, stop, target, held = 0, 0.0, None, None, 0

    def close_trade(i, price, reason):
        trades.append({'entry_i': entry_i, 'exit_i': i, 'entry_price': entry_price,
                       'exit_price': float(price), 'exit_reason': reason})

    for i in range(len(frame)):
        if in_pos:
            held += 1
            exit_price = reason = None
            if has_ohlc:
                if stop is not None and opens[i] <= stop:
                    exit_price, reason = opens[i], 'Stop (gap)'
                elif target is not None and opens[i] >= target:
                    exit_price, reason = opens[i], 'Target (gap)'
                elif exit_signal[i]:
                    exit_price, reason = opens[i], 'Strategy exit'
                elif stop is not None and lows[i] <= stop:
                    exit_price, reason = stop, 'Stop'
                elif target is not None and highs[i] >= target:
                    exit_price, reason = target, 'Target'
                elif held >= max_hold:
                    exit_price, reason = closes[i], 'Max hold'
            else:
                if target is not None and closes[i] >= target:
                    reason = 'Target'
                elif stop is not None and closes[i] <= stop:
                    reason = 'Stop'
                elif exit_signal[i]:
                    reason = 'Strategy exit'
                elif held >= max_hold:
                    reason = 'Max hold'
                exit_price = closes[i] if reason else None
            if reason:
                close_trade(i, exit_price, reason)
                in_pos = False
                continue

        if not in_pos and entry_candidates[i]:
            entry_i, held, in_pos = i, 0, True
            entry_price = float(opens[i] if has_ohlc else closes[i])
            stop = entry_price * (1 - stop_loss_pct / 100) if stop_loss_pct > 0 else None
            target = entry_price * (1 + profit_target_pct / 100) if profit_target_pct > 0 else None
            if has_ohlc and stop is not None and lows[i] <= stop:
                close_trade(i, stop, 'Stop')
                in_pos = False

    # Realize the last open position so every backtest trade is measurable; one
    # opened on the final bar has no result yet and is left out.
    if in_pos and entry_i < len(frame) - 1:
        close_trade(len(frame) - 1, closes[-1], 'End of data')
    return trades


def _signal_inputs(
    prices, regime_series, strategy_type, param_a, param_b, param_c, param_d,
    benchmark_close=None, momentum_percentile=None, breadth=None,
):
    """Clean bars, aligned regime, setup and strategy exit; None when history is too short."""
    frame, has_ohlc = ohlc_frame(prices)
    if len(frame) < strategy_warmup(strategy_type, param_a, param_b, param_c, param_d):
        return None
    if regime_series.empty:
        regime_aligned = pd.Series('bear', index=frame.index, dtype=str)
    else:
        regime_aligned = regime_series.reindex(frame.index, method='ffill').fillna('bear')
    setup, strategy_exit, _ = strategy_components(
        frame['close'], regime_aligned, benchmark_close, momentum_percentile,
        strategy_type, param_a, param_b, param_c, param_d, breadth=breadth,
    )
    return frame, has_ohlc, regime_aligned, setup, strategy_exit


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
    """Next-session entries and exits as boolean series (close-based view of the trades)."""
    inputs = _signal_inputs(
        price_series, regime_series, strategy_type, param_a, param_b, param_c, param_d,
        benchmark_close, momentum_percentile, breadth,
    )
    if inputs is None:
        return None
    frame, has_ohlc, regime_aligned, setup, strategy_exit = inputs
    trades = simulate_trades(
        frame, setup, strategy_exit, max_hold, profit_target_pct, stop_loss_pct, trade_start, has_ohlc,
    )
    entries = np.zeros(len(frame), dtype=bool)
    exits = np.zeros(len(frame), dtype=bool)
    for trade in trades:
        entries[trade['entry_i']] = True
        exits[trade['exit_i']] = True
    return (
        frame['close'],
        pd.Series(entries, index=frame.index),
        pd.Series(exits, index=frame.index),
        regime_aligned,
    )


def backtest_trades(
    prices: pd.Series | pd.DataFrame,
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
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Trades of one strategy after fees, tax and tick slippage, plus the clean bars.

    Each fill pays its side's fee plus ``slippage_ticks`` IDX ticks at the fill
    price. ``return_pct`` is the profit after costs over the entry value. None when history is
    too short for the indicators.
    """
    inputs = _signal_inputs(
        prices, regime_series, strategy_type, param_a, param_b, param_c, param_d,
        benchmark_close, momentum_percentile, breadth,
    )
    if inputs is None:
        return None
    frame, has_ohlc, regime_aligned, setup, strategy_exit = inputs
    raw = simulate_trades(
        frame, setup, strategy_exit, max_hold, profit_target_pct, stop_loss_pct, trade_start, has_ohlc,
    )
    if not raw:
        return pd.DataFrame(columns=TRADE_COLUMNS), frame
    trades = pd.DataFrame(raw)
    entry_cost = buy_fee / 100 + slippage_fraction(trades['entry_price'], slippage_ticks)
    exit_cost = (sell_fee + sell_tax) / 100 + slippage_fraction(trades['exit_price'], slippage_ticks)
    trades['entry_cost'] = entry_cost.to_numpy()
    trades['exit_cost'] = exit_cost.to_numpy()
    # Profit after both sides' costs over the entry value before costs (vectorbt's trade return).
    trades['return_pct'] = (
        trades['exit_price'] * (1 - trades['exit_cost']) / trades['entry_price'] - (1 + trades['entry_cost'])
    ) * 100
    trades['entry_date'] = frame.index[trades['entry_i']]
    trades['exit_date'] = frame.index[trades['exit_i']]
    trades['sessions'] = trades['exit_i'] - trades['entry_i']
    trades['regime'] = regime_aligned.to_numpy()[trades['entry_i']]
    return trades, frame


def equity_curve(trades: pd.DataFrame, frame: pd.DataFrame, start: float = 100.0) -> pd.Series:
    """Account value at each close when every trade uses the whole account.

    Marked to the close while in a position (exit costs are paid at the exit),
    the same convention as a vectorbt signal portfolio.
    """
    closes = frame['close'].to_numpy(dtype=float)
    value = np.full(len(frame), np.nan)
    cash = start
    last = 0
    for trade in trades.itertuples():
        value[last:trade.entry_i] = cash
        shares = cash / (trade.entry_price * (1 + trade.entry_cost))
        value[trade.entry_i:trade.exit_i] = shares * closes[trade.entry_i:trade.exit_i]
        cash = shares * trade.exit_price * (1 - trade.exit_cost)
        value[trade.exit_i] = cash
        last = trade.exit_i + 1
    value[last:] = cash
    return pd.Series(value, index=frame.index)


def trade_metrics(trades: pd.DataFrame, frame: pd.DataFrame) -> dict | None:
    """Headline metrics of a backtest; None without trades."""
    if trades.empty:
        return None
    returns = trades['return_pct'].to_numpy(dtype=float)
    equity = equity_curve(trades, frame)
    drawdown = equity / equity.cummax() - 1
    return {
        'total_return': float((equity.iloc[-1] / 100 - 1) * 100),
        'win_rate': float((returns > 0).mean() * 100),
        'max_drawdown': float(drawdown.min() * 100),
        'trades': int(len(returns)),
        'avg_trade': float(returns.mean()),
        # Sums let trade statistics be pooled across stocks later.
        'gross_win': float(returns[returns > 0].sum()),
        'gross_loss': float(-returns[returns < 0].sum()),
        'sum_sq': float((returns ** 2).sum()),
        'bear_entries': int((trades['regime'] == 'bear').sum()),
        'bull_entries': int((trades['regime'] == 'bull').sum()),
    }


def run_regime_backtest(
    price_series: pd.Series | pd.DataFrame,
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
    """Run one short-term swing strategy; returns performance metrics, or None without trades.

    ``price_series`` is a close series or an OHLC frame (see :func:`simulate_trades`).
    """
    result = backtest_trades(
        price_series, regime_series, strategy_type, param_a, param_b, param_c, param_d,
        max_hold, profit_target_pct, stop_loss_pct, buy_fee, sell_fee, sell_tax, trade_start,
        benchmark_close, momentum_percentile, breadth, slippage_ticks,
    )
    if result is None:
        return None
    return trade_metrics(*result)


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


def stock_holdout_t_stat(frame: pd.DataFrame) -> pd.Series:
    """Holdout t-stat of each per-stock result row, from its pooled trade sums."""
    column = lambda name: pd.to_numeric(frame[name], errors='coerce').astype(float)
    n = column('Validation Trades')
    mean = (column('Validation Gross Win') - column('Validation Gross Loss')) / n.where(n > 0)
    variance = (column('Validation Sum Sq') - n * mean ** 2) / (n - 1).where(n > 1)
    return mean / np.sqrt(variance.where(variance > 0) / n)


def select_stock_strategies(
    results_df: pd.DataFrame,
    global_strategy: dict,
    min_t_stat: float = 0.0,
) -> dict[str, dict]:
    """Select stock overrides only when holdout return beats the global strategy.

    ``min_t_stat`` adds a significance bar on the stock's own holdout trades;
    use :func:`required_t_stat` with the number of combinations each stock
    chose from, since the best of many is flattered by luck.
    """
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
        if min_t_stat > 0:
            candidates = candidates[stock_holdout_t_stat(candidates) >= min_t_stat]
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
            'validation_win_rate': float(best['Validation Win Rate']),
            'validation_t_stat': float(stock_holdout_t_stat(best.to_frame().T).iloc[0]),
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


def holdout_split_date(start, end, validation_pct: float) -> pd.Timestamp:
    """Trades entered on or after this date are the holdout; the same date for every stock."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    return (start + (end - start) * (1 - validation_pct / 100)).normalize()


_EMPTY_METRICS = {
    'total_return': np.nan, 'win_rate': np.nan, 'max_drawdown': np.nan, 'trades': 0, 'avg_trade': np.nan,
    'gross_win': 0.0, 'gross_loss': 0.0, 'sum_sq': 0.0, 'bear_entries': 0, 'bull_entries': 0,
}


def evaluate_stock_strategies(
    ticker: str,
    prices: pd.Series | pd.DataFrame | None,
    regime_series: pd.Series,
    param_combos: list[tuple],
    split_date,
    buy_fee: float,
    sell_fee: float,
    sell_tax: float,
    benchmark_close: pd.Series | None,
    momentum_percentiles: dict[int, pd.Series],
    breadth: pd.Series | None = None,
    slippage_ticks: float = 0.0,
    trades_out: list | None = None,
) -> tuple[list[dict], str]:
    """Test every strategy for one stock; safe to execute in a worker thread.

    Each combination is simulated once over the whole history; trades entered
    before ``split_date`` are training, the rest the holdout. ``trades_out``,
    when given, collects every trade (``combo`` is its index in
    ``param_combos``) for :func:`walk_forward`.
    """
    if prices is None or len(prices) == 0:
        return [], 'missing_data'
    frame, _ = ohlc_frame(prices)
    split = pd.Timestamp(split_date)
    warmup = max(strategy_warmup(combo[0], *combo[1:5]) for combo in param_combos)
    if (frame.index < split).sum() < warmup or (frame.index >= split).sum() < warmup:
        return [], 'insufficient_history'
    rows = []

    for combo_id, (strategy_type, param_a, param_b, param_c, param_d, hold_d, profit_target, stop_loss) in enumerate(param_combos):
        result = backtest_trades(
            frame, regime_series, strategy_type, param_a, param_b, param_c, param_d,
            int(hold_d), float(profit_target), float(stop_loss), buy_fee, sell_fee, sell_tax,
            benchmark_close=benchmark_close, momentum_percentile=momentum_percentiles.get(int(param_a)),
            breadth=breadth, slippage_ticks=slippage_ticks,
        )
        if result is None:
            continue
        trades, bars = result
        in_holdout = trades['entry_date'] >= split
        train = trade_metrics(trades[~in_holdout], bars)
        if train is None:
            continue
        validation = trade_metrics(trades[in_holdout], bars) or _EMPTY_METRICS
        if trades_out is not None:
            trades_out.extend(
                {'symbol': ticker, 'combo': combo_id, 'entry_date': entry, 'return_pct': value}
                for entry, value in zip(trades['entry_date'], trades['return_pct'])
            )

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


# ── Walk-forward selection and stricter gates ────────────────────────────── #

WALK_FORWARD_TRAIN_YEARS = 3
WALK_FORWARD_FOLD_MONTHS = 6
MIN_WALK_FORWARD_TRAIN_TRADES = 30
MIN_WALK_FORWARD_TRADES = 30
# One-sided significance level, shared by all the combinations tried (Bonferroni).
FAMILY_ALPHA = 0.05


def trade_stats(returns) -> dict:
    """Pooled statistics of trade returns in percent."""
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n == 0:
        return {'trades': 0, 'win_rate': np.nan, 'avg_trade': np.nan, 'profit_factor': np.nan, 't_stat': np.nan}
    gains, losses = returns[returns > 0].sum(), -returns[returns < 0].sum()
    std = returns.std(ddof=1) if n > 1 else np.nan
    return {
        'trades': n,
        'win_rate': float((returns > 0).mean() * 100),
        'avg_trade': float(returns.mean()),
        'profit_factor': float(gains / losses) if losses > 0 else (np.inf if gains > 0 else np.nan),
        't_stat': float(returns.mean() / (std / np.sqrt(n))) if n > 1 and std > 0 else np.nan,
    }


def required_t_stat(tests: int, alpha: float = FAMILY_ALPHA) -> float:
    """t-stat a result needs when it is the best of ``tests`` tries (one-sided Bonferroni, at least MIN_T_STAT)."""
    return max(MIN_T_STAT, NormalDist().inv_cdf(1 - alpha / max(int(tests), 1)))


def _pick_combo(trades: pd.DataFrame, min_trades: int):
    """The combination with the highest pooled t-stat among those with enough trades."""
    if trades.empty:
        return None
    grouped = trades.groupby('combo')['return_pct']
    stats = pd.DataFrame({
        'n': grouped.size(),
        'mean': grouped.mean(),
        'std': grouped.std(ddof=1),
    })
    stats = stats[(stats['n'] >= min_trades) & (stats['std'] > 0)]
    if stats.empty:
        return None
    stats['t'] = stats['mean'] / (stats['std'] / np.sqrt(stats['n']))
    stats = stats[stats['mean'] > 0]
    if stats.empty:
        return None
    return int(stats.sort_values(['t', 'mean'], ascending=False).index[0])


def walk_forward(
    trades: pd.DataFrame,
    start,
    end,
    train_years: int = WALK_FORWARD_TRAIN_YEARS,
    fold_months: int = WALK_FORWARD_FOLD_MONTHS,
    min_train_trades: int = MIN_WALK_FORWARD_TRAIN_TRADES,
) -> dict:
    """Walk-forward record of "pick the best combination on the last few years".

    From ``start`` to ``end`` in ``fold_months`` steps, the combination with
    the best pooled t-stat on the previous ``train_years`` (across all stocks)
    is chosen and its trades in the next fold are recorded. The record is the
    honest estimate of the published setup's edge; ``final_combo`` is the
    choice on the latest ``train_years``, the one to publish.
    """
    trades = trades.copy()
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = pd.DateOffset(years=train_years)
    folds, tested = [], []
    fold_start = start
    while fold_start < end:
        fold_end = min(fold_start + pd.DateOffset(months=fold_months), end + pd.Timedelta(days=1))
        train = trades[(trades['entry_date'] >= fold_start - window) & (trades['entry_date'] < fold_start)]
        combo = _pick_combo(train, min_train_trades)
        test = trades[(trades['entry_date'] >= fold_start) & (trades['entry_date'] < fold_end)]
        test = test[test['combo'] == combo] if combo is not None else test.iloc[0:0]
        folds.append({
            'start': fold_start.date().isoformat(), 'end': (fold_end - pd.Timedelta(days=1)).date().isoformat(),
            'combo': combo, **trade_stats(test['return_pct']),
        })
        tested.append(test)
        fold_start = fold_end
    record = pd.concat(tested) if tested else trades.iloc[0:0]
    final = trades[(trades['entry_date'] >= end - window) & (trades['entry_date'] <= end)]
    return {
        'folds': folds,
        'record': trade_stats(record['return_pct']),
        'final_combo': _pick_combo(final, min_train_trades),
    }


def passes_walk_forward_gates(record: dict) -> bool:
    """Absolute requirements on the walk-forward record before a setup is published."""
    return bool(
        record['trades'] >= MIN_WALK_FORWARD_TRADES
        and record['avg_trade'] > 0
        and record['profit_factor'] >= MIN_PROFIT_FACTOR
        and record['t_stat'] >= MIN_T_STAT
    )


# ── IDX trading rules for position sizing ────────────────────────────────── #

# ARB is a flat 15% until the end of 2026, then symmetric with ARA (IDX, Sep 2026).
SYMMETRIC_ARB_FROM = datetime.date(2027, 1, 1)
FLAT_ARB = 0.15
LOT_SIZE = 100


def round_to_tick(price: float, direction: str = 'down') -> float:
    """Round a price to a valid IDX tick, ``down``, ``up`` or ``nearest``."""
    tick = float(idx_tick_size([price])[0])
    rounder = {'down': np.floor, 'up': np.ceil, 'nearest': np.round}[direction]
    return float(rounder(price / tick) * tick)


def price_limits(previous_close: float, on_date=None) -> tuple[float, float]:
    """Next session's auto-reject band (ARB floor, ARA ceiling) around ``previous_close``."""
    on_date = pd.Timestamp(on_date or datetime.date.today()).date()
    if previous_close <= 10:
        return max(previous_close - 1, 1.0), previous_close + 1
    ara = 0.35 if previous_close <= 200 else 0.25 if previous_close <= 5000 else 0.20
    arb = ara if on_date >= SYMMETRIC_ARB_FROM else FLAT_ARB
    return (
        max(round_to_tick(previous_close * (1 - arb), 'up'), 1.0),
        round_to_tick(previous_close * (1 + ara), 'down'),
    )


def position_size(capital: float, risk_pct: float, entry: float, stop: float) -> dict:
    """Whole lots so a stop-out loses at most ``risk_pct`` of ``capital``, and never more than the capital buys."""
    per_share_risk = entry - stop
    if capital <= 0 or risk_pct <= 0 or entry <= 0 or per_share_risk <= 0:
        return {'lots': 0, 'risk_idr': 0.0, 'cost_idr': 0.0}
    by_risk = capital * risk_pct / 100 / (per_share_risk * LOT_SIZE)
    by_cash = capital / (entry * LOT_SIZE)
    lots = int(np.floor(min(by_risk, by_cash)))
    return {
        'lots': lots,
        'risk_idr': float(lots * LOT_SIZE * per_share_risk),
        'cost_idr': float(lots * LOT_SIZE * entry),
    }


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


# ── Shortlist, published setups and price adjustment (page and pipeline) ─── #

BLUE_CHIPS = [
    'BBCA.JK', 'BMRI.JK', 'BBRI.JK', 'BBNI.JK', 'INDF.JK', 'ICBP.JK',
    'UNVR.JK', 'AMRT.JK', 'TLKM.JK', 'EXCL.JK', 'ISAT.JK', 'ASII.JK',
    'UNTR.JK', 'ADRO.JK', 'PTBA.JK', 'ITMG.JK', 'BRIS.JK', 'KLBF.JK', 'GOTO.JK',
]
VOLATILITY_LOOKBACK = 60
# Daily moves beyond this are treated as corporate actions or bad data, not volatility.
MAX_PLAUSIBLE_DAILY_MOVE = 0.35
SETUP_KEYS = ('strategy_type', 'param_a', 'param_b', 'param_c', 'param_d', 'max_hold', 'profit_target', 'stop_loss')


def price_volatility(prices: pd.DataFrame, lookback: int = VOLATILITY_LOOKBACK) -> pd.DataFrame:
    """Recent daily volatility (%) per stock from a long ``symbol, date, close`` frame."""
    columns = ['stock', 'Daily Volatility', 'Volatility Observations']
    if prices.empty:
        return pd.DataFrame(columns=columns)
    prices = prices.sort_values(['symbol', 'date'])
    prices = prices.groupby('symbol', group_keys=False).tail(int(lookback) + 1).copy()
    prices['return'] = prices.groupby('symbol')['close'].pct_change()
    prices.loc[prices['return'].abs() > MAX_PLAUSIBLE_DAILY_MOVE, 'return'] = np.nan
    volatility = prices.groupby('symbol')['return'].agg(['std', 'count']).reset_index()
    volatility['Daily Volatility'] = volatility['std'] * 100
    volatility = volatility.rename(columns={'symbol': 'stock', 'count': 'Volatility Observations'})
    return volatility[columns]


def build_shortlist(
    universe: pd.DataFrame,
    volatility: pd.DataFrame,
    min_mcap_trill: float = 1.0,
    min_value_traded_bn: float = 5.0,
    min_volatility: float = 1.0,
    max_stocks: int = 200,
    volatility_lookback: int = VOLATILITY_LOOKBACK,
    pin_blue_chips: bool = True,
) -> pd.DataFrame:
    """Liquid stocks that move enough to swing trade, most volatile first.

    ``universe`` has ``stock``, ``mktCap``, ``price`` and ``volAvg``;
    ``volatility`` comes from :func:`price_volatility`.
    """
    frame = universe.merge(volatility, on='stock', how='inner')
    mask = pd.Series(True, index=frame.index)
    if 'mktCap' in frame.columns:
        mask &= frame['mktCap'] >= min_mcap_trill * 1_000_000_000_000
    if {'volAvg', 'price'}.issubset(frame.columns):
        frame['Avg Value Traded'] = frame['volAvg'] * frame['price']
        mask &= frame['Avg Value Traded'].fillna(0) >= min_value_traded_bn * 1_000_000_000
    mask &= frame['Volatility Observations'] >= max(15, int(volatility_lookback) // 2)
    mask &= frame['Daily Volatility'] >= float(min_volatility)
    frame = frame[mask].copy()
    frame['Volatility Score'] = frame['Daily Volatility'].rank(pct=True) * 100
    ranked = frame.sort_values(['Volatility Score', 'Daily Volatility'], ascending=False)
    if pin_blue_chips:
        pinned = ranked[ranked['stock'].isin(BLUE_CHIPS)]
        ranked = pd.concat([pinned, ranked[~ranked['stock'].isin(BLUE_CHIPS)]])
    return ranked.drop_duplicates('stock').head(int(max_stocks)).reset_index(drop=True)


def dividend_adjust_ohlc(frame: pd.DataFrame, dividends: pd.DataFrame | None) -> pd.DataFrame:
    """Back-adjust open, high, low and close by the same factor as :func:`dividend_adjust`."""
    if dividends is None or dividends.empty or frame.empty:
        return frame
    factor = dividend_adjust(frame['close'], dividends) / frame['close']
    adjusted = frame.copy()
    for column in ('open', 'high', 'low', 'close'):
        if column in adjusted.columns:
            adjusted[column] = adjusted[column] * factor
    return adjusted


def combo_setup(combo: tuple) -> dict:
    """A parameter combination as a published setup dict."""
    return dict(zip(SETUP_KEYS, (combo[0], *(float(v) for v in combo[1:5]), int(combo[5]),
                                 float(combo[6]), float(combo[7]))))


def preset_combos(preset: str = 'Balanced', strategy_types=None) -> list[tuple]:
    """Every (strategy, params, max hold, target, stop) combination of a search preset."""
    space = SEARCH_PRESETS[preset]
    return [
        (strategy_type, *params, hold, target, stop)
        for strategy_type in (strategy_types or STRATEGY_TYPES)
        for params in space['strategy_params'][strategy_type]
        for hold in space['hold_days']
        for target, stop in space['exit_rules']
    ]
