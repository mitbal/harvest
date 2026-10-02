"""Backtest logic for the dividend play (pre-dividend run-up) strategy.

The idea: many dividend stocks drift lower in the months after the previous
payout and rally as the next ex-date approaches, because investors buy to
qualify for the dividend. The strategy buys inside an entry window a few
months before the expected ex-date and sells into that run-up, either before
the ex-date (capturing only the price gain) or after it (also collecting the
dividend but absorbing the ex-date price drop).

The engine works on NumPy arrays so a parameter grid can be swept across a
few hundred stocks in seconds.
"""
import datetime

import numpy as np
import pandas as pd


ENTRY_FIXED = 'fixed'
ENTRY_DIP = 'dip'
EXIT_CUM = 'cum'
EXIT_AFTER = 'after'

ENTRY_LABELS = {
    ENTRY_DIP: 'Buy the dip',
    ENTRY_FIXED: 'Buy at window start',
}
EXIT_LABELS = {
    EXIT_CUM: 'Sell before ex-date',
    EXIT_AFTER: 'Hold through ex-date',
}

EVENT_COLUMNS = [
    'ex_date', 'dividend', 'entry_date', 'entry_price', 'dip_triggered',
    'exit_date', 'exit_price', 'exit_reason', 'days_held', 'collected_dividend',
    'price_return', 'dividend_return', 'total_return', 'max_drawdown',
]


def prepare_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean ``date``/``close`` frame sorted oldest first."""
    if price_df is None or price_df.empty or not {'date', 'close'}.issubset(price_df.columns):
        return pd.DataFrame(columns=['date', 'close'])
    prices = price_df[['date', 'close']].copy()
    prices['date'] = pd.to_datetime(prices['date'], errors='coerce')
    prices['close'] = pd.to_numeric(prices['close'], errors='coerce')
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna()
    prices = prices[prices['close'] > 0]
    return prices.drop_duplicates('date', keep='last').sort_values('date').reset_index(drop=True)


def prepare_dividends(div_df: pd.DataFrame) -> pd.DataFrame:
    """Return ex-dates with one summed dividend amount per date, oldest first."""
    if div_df is None or div_df.empty or 'date' not in div_df.columns:
        return pd.DataFrame(columns=['date', 'dividend'])
    divs = div_df.copy()
    divs['date'] = pd.to_datetime(divs['date'], errors='coerce')
    amount = None
    for column in ('adjDividend', 'dividend'):
        if column in divs.columns:
            values = pd.to_numeric(divs[column], errors='coerce')
            amount = values if amount is None else amount.fillna(values)
    divs['dividend'] = amount if amount is not None else np.nan
    divs = divs.dropna(subset=['date', 'dividend'])
    divs = divs[divs['dividend'] > 0]
    divs = divs.groupby('date', as_index=False)['dividend'].sum()
    return divs.sort_values('date').reset_index(drop=True)


class PlayData:
    """Price and dividend arrays for one stock, with cached moving averages."""

    def __init__(self, price_df: pd.DataFrame, div_df: pd.DataFrame):
        prices = prepare_prices(price_df)
        divs = prepare_dividends(div_df)
        self.dates = prices['date'].to_numpy(dtype='datetime64[D]')
        self.close = prices['close'].to_numpy(dtype=float)
        self.ex_dates = divs['date'].to_numpy(dtype='datetime64[D]')
        self.dividends = divs['dividend'].to_numpy(dtype=float)
        self._sma = {}

    @property
    def empty(self) -> bool:
        return len(self.close) == 0 or len(self.ex_dates) == 0

    def sma(self, window: int) -> np.ndarray:
        window = int(window)
        if window not in self._sma:
            values = np.full(len(self.close), np.nan)
            if len(self.close) >= window:
                cumsum = np.cumsum(np.insert(self.close, 0, 0.0))
                values[window - 1:] = (cumsum[window:] - cumsum[:-window]) / window
            self._sma[window] = values
        return self._sma[window]


def _simulate(
    data: PlayData,
    entry_start_days: int,
    entry_end_days: int,
    entry_rule: str,
    dip_ma_days: int,
    dip_pct: float,
    exit_rule: str,
    hold_after_days: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    buy_fee_pct: float,
    sell_fee_pct: float,
    dividend_tax_pct: float,
) -> list[dict]:
    dates, close = data.dates, data.close
    if data.empty or entry_start_days <= entry_end_days:
        return []
    first_date, last_date = dates[0], dates[-1]
    sma = data.sma(dip_ma_days) if entry_rule == ENTRY_DIP else None
    buy_cost = 1 + buy_fee_pct / 100
    sell_keep = 1 - sell_fee_pct / 100
    div_keep = 1 - dividend_tax_pct / 100
    start_offset = np.timedelta64(int(entry_start_days), 'D')
    end_offset = np.timedelta64(int(entry_end_days), 'D')
    hold_offset = np.timedelta64(int(hold_after_days), 'D')
    trades = []

    for ex_date, dividend in zip(data.ex_dates, data.dividends):
        window_start = ex_date - start_offset
        if window_start < first_date:
            continue
        i0 = int(np.searchsorted(dates, window_start, 'left'))
        i1 = int(np.searchsorted(dates, ex_date - end_offset, 'right')) - 1
        if i1 < i0:
            continue

        entry_idx, dip_triggered = i0, False
        if entry_rule == ENTRY_DIP:
            hits = np.flatnonzero(close[i0:i1 + 1] <= sma[i0:i1 + 1] * (1 - dip_pct / 100))
            if len(hits):
                entry_idx, dip_triggered = i0 + int(hits[0]), True
            else:
                entry_idx = i1

        if exit_rule == EXIT_AFTER:
            target = ex_date + hold_offset
            if target > last_date:
                continue
            planned_idx = int(np.searchsorted(dates, target, 'left'))
        else:
            if ex_date > last_date:
                continue
            planned_idx = int(np.searchsorted(dates, ex_date, 'left')) - 1
        if planned_idx <= entry_idx:
            continue

        entry_price = close[entry_idx]
        path = close[entry_idx + 1:planned_idx + 1]
        earned = np.where(dates[entry_idx + 1:planned_idx + 1] >= ex_date, dividend, 0.0)
        path_returns = (path + earned) / entry_price - 1

        exit_offset = len(path) - 1
        exit_reason = 'Before ex-date' if exit_rule == EXIT_CUM else 'After ex-date'
        if take_profit_pct > 0:
            hits = np.flatnonzero(path_returns * 100 >= take_profit_pct)
            if len(hits) and hits[0] < exit_offset:
                exit_offset, exit_reason = int(hits[0]), 'Take profit'
        if stop_loss_pct > 0:
            hits = np.flatnonzero(path_returns * 100 <= -stop_loss_pct)
            if len(hits) and (hits[0] < exit_offset or (hits[0] == exit_offset and exit_reason != 'Take profit')):
                exit_offset, exit_reason = int(hits[0]), 'Stop loss'

        exit_idx = entry_idx + 1 + exit_offset
        exit_price = close[exit_idx]
        collected = bool(dates[exit_idx] >= ex_date)
        received = dividend * div_keep if collected else 0.0
        cost_basis = entry_price * buy_cost
        price_return = (exit_price * sell_keep / cost_basis - 1) * 100
        dividend_return = received / cost_basis * 100
        trades.append({
            'ex_date': ex_date,
            'dividend': float(dividend),
            'entry_date': dates[entry_idx],
            'entry_price': float(entry_price),
            'dip_triggered': dip_triggered,
            'exit_date': dates[exit_idx],
            'exit_price': float(exit_price),
            'exit_reason': exit_reason,
            'days_held': int((dates[exit_idx] - dates[entry_idx]).astype(int)),
            'collected_dividend': collected,
            'price_return': float(price_return),
            'dividend_return': float(dividend_return),
            'total_return': float(price_return + dividend_return),
            'max_drawdown': float(min(0.0, path_returns[:exit_offset + 1].min()) * 100),
        })
    return trades


def backtest_dividend_play(
    price_df,
    div_df: pd.DataFrame | None = None,
    entry_start_days: int = 90,
    entry_end_days: int = 30,
    entry_rule: str = ENTRY_DIP,
    dip_ma_days: int = 50,
    dip_pct: float = 0.0,
    exit_rule: str = EXIT_CUM,
    hold_after_days: int = 5,
    take_profit_pct: float = 0.0,
    stop_loss_pct: float = 0.0,
    buy_fee_pct: float = 0.0,
    sell_fee_pct: float = 0.0,
    dividend_tax_pct: float = 0.0,
) -> pd.DataFrame:
    """Simulate one trade per historical ex-date.

    ``price_df`` may be a price frame (with ``div_df``) or a prepared
    :class:`PlayData`. Entry happens inside
    ``[ex - entry_start_days, ex - entry_end_days]``:

    - ``fixed``: buy at the first close of the window.
    - ``dip``: buy at the first close that is at least ``dip_pct`` percent below
      its ``dip_ma_days`` simple moving average; if none occurs, buy at the last
      close of the window so every dividend still gets a trade.

    Exit:

    - ``cum``: sell at the last close before the ex-date (no dividend).
    - ``after``: hold through the ex-date, collect the dividend, and sell at the
      first close ``hold_after_days`` calendar days after the ex-date.

    A positive ``take_profit_pct`` or ``stop_loss_pct`` closes the trade early
    when the gross return (including any dividend already earned) crosses it.
    Reported returns are percentages net of fees and dividend tax.
    """
    data = price_df if isinstance(price_df, PlayData) else PlayData(price_df, div_df)
    trades = _simulate(
        data, int(entry_start_days), int(entry_end_days), entry_rule, int(dip_ma_days),
        float(dip_pct), exit_rule, int(hold_after_days), float(take_profit_pct),
        float(stop_loss_pct), float(buy_fee_pct), float(sell_fee_pct), float(dividend_tax_pct),
    )
    frame = pd.DataFrame(trades, columns=EVENT_COLUMNS)
    for column in ('ex_date', 'entry_date', 'exit_date'):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def summarize_trades(trades: pd.DataFrame) -> dict:
    """Aggregate per-trade results into headline statistics."""
    if trades is None or trades.empty:
        return {'trades': 0}
    returns = trades['total_return']
    avg_days = float(trades['days_held'].mean())
    compounded = float((np.prod(1 + returns / 100) - 1) * 100)
    years = max((trades['exit_date'].max() - trades['entry_date'].min()).days / 365.25, 1 / 365.25)
    gains, losses = returns[returns > 0].sum(), -returns[returns < 0].sum()
    return {
        'trades': int(len(trades)),
        'win_rate': float((returns > 0).mean() * 100),
        'avg_return': float(returns.mean()),
        'median_return': float(returns.median()),
        'best': float(returns.max()),
        'worst': float(returns.min()),
        'avg_days': avg_days,
        'avg_drawdown': float(trades['max_drawdown'].mean()),
        'profit_factor': float(gains / losses) if losses > 0 else np.inf,
        'compounded': compounded,
        'cagr': float(((1 + compounded / 100) ** (1 / years) - 1) * 100),
        'return_per_month': float(returns.mean() / max(avg_days, 1) * 30.44),
    }


def estimate_next_ex_dates(divs: pd.DataFrame, as_of=None, horizon_days: int = 365) -> pd.DataFrame:
    """Project upcoming ex-dates by rolling the last year's schedule forward.

    Each ex-date from the most recent 12 months of history is shifted forward
    whole years until it lands after ``as_of``. Dates beyond ``horizon_days``
    are dropped. Returns ``expected_date``, ``based_on`` and ``dividend``.
    """
    columns = ['expected_date', 'based_on', 'dividend']
    if divs is None or divs.empty:
        return pd.DataFrame(columns=columns)
    as_of = pd.Timestamp(as_of or datetime.date.today()).normalize()
    last = divs['date'].max()
    recent = divs[divs['date'] > last - pd.DateOffset(days=365)]
    rows = []
    for event in recent.itertuples():
        expected = event.date
        while expected <= as_of:
            expected = expected + pd.DateOffset(years=1)
        if expected <= as_of + pd.Timedelta(days=horizon_days):
            rows.append({'expected_date': expected, 'based_on': event.date, 'dividend': event.dividend})
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values('expected_date').reset_index(drop=True)


def entry_status(prices: pd.DataFrame, expected_ex, entry_start_days, entry_end_days, dip_ma_days, dip_pct, as_of=None) -> dict:
    """Describe where today sits relative to the next expected entry window."""
    as_of = pd.Timestamp(as_of or datetime.date.today()).normalize()
    expected_ex = pd.Timestamp(expected_ex)
    window_start = expected_ex - pd.Timedelta(days=int(entry_start_days))
    window_end = expected_ex - pd.Timedelta(days=int(entry_end_days))
    prices = prepare_prices(prices)
    close = sma = np.nan
    if not prices.empty:
        close = float(prices['close'].iloc[-1])
        tail = prices['close'].tail(int(dip_ma_days))
        if len(tail) == int(dip_ma_days):
            sma = float(tail.mean())
    below = bool(np.isfinite(sma) and close <= sma * (1 - dip_pct / 100))

    if as_of < window_start:
        status = 'Waiting'
    elif as_of <= window_end:
        status = 'Buy zone - dip active' if below else 'In entry window'
    elif as_of < expected_ex:
        status = 'Run-up phase'
    else:
        status = 'Passed'
    return {
        'status': status,
        'window_start': window_start,
        'window_end': window_end,
        'days_to_window': int((window_start - as_of).days),
        'days_to_ex': int((expected_ex - as_of).days),
        'close': close,
        'sma': sma,
        'dip_active': below,
        'vs_sma': (close / sma - 1) * 100 if np.isfinite(sma) and sma > 0 else np.nan,
    }


# --------------------------------------------------------------------------- #
# Parameter search                                                             #
# --------------------------------------------------------------------------- #
SETUP_FIELDS = (
    'entry_start_days', 'entry_end_days', 'entry_rule', 'dip_ma_days', 'dip_pct',
    'exit_rule', 'hold_after_days', 'take_profit_pct', 'stop_loss_pct',
)


def build_setup_grid(windows, entries, exits, exit_limits) -> list[dict]:
    """Expand search dimensions into setup dicts.

    ``windows`` are ``(start, end)`` day pairs, ``entries`` are
    ``(rule, ma_days, dip_pct)``, ``exits`` are ``(rule, hold_after_days)`` and
    ``exit_limits`` are ``(take_profit_pct, stop_loss_pct)``.
    """
    grid = []
    for start, end in windows:
        for rule, ma_days, dip_pct in entries:
            for exit_rule, hold_after in exits:
                for take_profit, stop_loss in exit_limits:
                    grid.append({
                        'entry_start_days': int(start), 'entry_end_days': int(end),
                        'entry_rule': rule, 'dip_ma_days': int(ma_days), 'dip_pct': float(dip_pct),
                        'exit_rule': exit_rule, 'hold_after_days': int(hold_after),
                        'take_profit_pct': float(take_profit), 'stop_loss_pct': float(stop_loss),
                    })
    return grid


def format_setup(setup: dict) -> str:
    entry = (
        f"dip {setup['dip_pct']:g}% < MA{setup['dip_ma_days']}"
        if setup['entry_rule'] == ENTRY_DIP else 'window start'
    )
    exit_text = (
        'sell before ex' if setup['exit_rule'] == EXIT_CUM
        else f"sell {setup['hold_after_days']}d after ex"
    )
    limits = []
    if setup['take_profit_pct'] > 0:
        limits.append(f"TP {setup['take_profit_pct']:g}%")
    if setup['stop_loss_pct'] > 0:
        limits.append(f"SL {setup['stop_loss_pct']:g}%")
    text = f"Buy {setup['entry_start_days']}-{setup['entry_end_days']}d before ex, {entry}; {exit_text}"
    return text + (f" ({', '.join(limits)})" if limits else '')


def _trade_stats(returns: np.ndarray, days: np.ndarray, prefix: str) -> dict:
    count = len(returns)
    if count == 0:
        return {f'{prefix} Trades': 0, f'{prefix} Avg Trade': np.nan, f'{prefix} Win Rate': np.nan,
                f'{prefix} Avg Days': np.nan}
    return {
        f'{prefix} Trades': count,
        f'{prefix} Avg Trade': float(returns.mean()),
        f'{prefix} Win Rate': float((returns > 0).mean() * 100),
        f'{prefix} Avg Days': float(days.mean()),
    }


def evaluate_stock_setups(symbol: str, data: PlayData, setups: list[dict], split_date, costs: dict) -> list[dict]:
    """Backtest every setup on one stock and split trades into train/holdout by ex-date."""
    if data is None or data.empty:
        return []
    split = np.datetime64(pd.Timestamp(split_date).date(), 'D')
    rows = []
    for setup_id, setup in enumerate(setups):
        trades = _simulate(data, **setup, **costs)
        if not trades:
            continue
        returns = np.array([trade['total_return'] for trade in trades])
        days = np.array([trade['days_held'] for trade in trades])
        drawdowns = np.array([trade['max_drawdown'] for trade in trades])
        holdout = np.array([trade['ex_date'] >= split for trade in trades])
        rows.append({
            'Symbol': symbol,
            'Setup ID': setup_id,
            **_trade_stats(returns[~holdout], days[~holdout], 'Train'),
            **_trade_stats(returns[holdout], days[holdout], 'Validation'),
            'Validation Sum': float(returns[holdout].sum()),
            'Validation Gains': float(returns[holdout][returns[holdout] > 0].sum()),
            'Validation Losses': float(-returns[holdout][returns[holdout] < 0].sum()),
            'Validation Sum Sq': float((returns[holdout] ** 2).sum()),
            'Train Sum': float(returns[~holdout].sum()),
            'Trades': len(returns),
            'Avg Trade': float(returns.mean()),
            'Win Rate': float((returns > 0).mean() * 100),
            'Avg Days': float(days.mean()),
            'Avg Drawdown': float(drawdowns.mean()),
        })
    return rows


def _pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().sum() <= 1:
        return pd.Series(50.0, index=series.index)
    ranked = numeric.rank(pct=True, method='average', ascending=ascending) * 100
    return ranked.fillna(0.0)


def summarize_setups(results: pd.DataFrame, setups: list[dict], min_profit_factor: float, min_t_stat: float) -> pd.DataFrame:
    """Pool per-stock results into one row per setup and score them."""
    if results is None or results.empty:
        return pd.DataFrame()
    grouped = results.groupby('Setup ID')
    summary = pd.DataFrame({
        'Stocks Tested': grouped['Symbol'].nunique(),
        'Train Trades': grouped['Train Trades'].sum(),
        'Train Sum': grouped['Train Sum'].sum(),
        'Validation Trades': grouped['Validation Trades'].sum(),
        'Validation Sum': grouped['Validation Sum'].sum(),
        'Validation Sum Sq': grouped['Validation Sum Sq'].sum(),
        'Validation Gains': grouped['Validation Gains'].sum(),
        'Validation Losses': grouped['Validation Losses'].sum(),
        'Stocks Validated': grouped['Validation Trades'].apply(lambda trades: int((trades > 0).sum())),
        'Positive Stocks': grouped['Validation Avg Trade'].apply(lambda avg: int((avg > 0).sum())),
    })
    weighted_win = (results['Validation Win Rate'].fillna(0) * results['Validation Trades']).groupby(results['Setup ID']).sum()
    weighted_days = (results['Avg Days'] * results['Trades']).groupby(results['Setup ID']).sum()
    total_trades = grouped['Trades'].sum()

    n = summary['Validation Trades'].replace(0, np.nan)
    summary['Train Avg Trade'] = summary['Train Sum'] / summary['Train Trades'].replace(0, np.nan)
    summary['Validation Avg Trade'] = summary['Validation Sum'] / n
    variance = (summary['Validation Sum Sq'] / n - summary['Validation Avg Trade'] ** 2) * n / (n - 1)
    std = np.sqrt(variance.clip(lower=0))
    summary['Validation t-stat'] = summary['Validation Avg Trade'] / (std / np.sqrt(n))
    summary['Validation Profit Factor'] = summary['Validation Gains'] / summary['Validation Losses'].replace(0, np.nan)
    summary.loc[(summary['Validation Losses'] == 0) & (summary['Validation Gains'] > 0), 'Validation Profit Factor'] = np.inf
    summary['Validation Win Rate'] = weighted_win / n
    summary['Avg Days'] = weighted_days / total_trades
    summary['Return / Month'] = summary['Validation Avg Trade'] / summary['Avg Days'].clip(lower=1) * 30.44
    summary['Positive Stocks %'] = summary['Positive Stocks'] / summary['Stocks Validated'].replace(0, np.nan) * 100
    summary['Generalization Gap'] = summary['Validation Avg Trade'] - summary['Train Avg Trade']
    summary['Passes Gates'] = (
        (summary['Train Avg Trade'] > 0)
        & (summary['Validation Avg Trade'] > 0)
        & (summary['Validation Profit Factor'] >= min_profit_factor)
        & (summary['Validation t-stat'] >= min_t_stat)
    )
    summary['Setup Score'] = (
        _pct_rank(summary['Validation Avg Trade']) * 0.30
        + _pct_rank(summary['Return / Month']) * 0.20
        + _pct_rank(summary['Validation Win Rate']) * 0.20
        + _pct_rank(summary['Positive Stocks %']) * 0.15
        + _pct_rank(summary['Validation t-stat']) * 0.15
    )

    setup_frame = pd.DataFrame(setups)
    setup_frame.index.name = 'Setup ID'
    summary = summary.join(setup_frame)
    summary['Setup'] = [format_setup(setups[setup_id]) for setup_id in summary.index]
    return summary.reset_index().sort_values(
        ['Passes Gates', 'Setup Score'], ascending=False,
    ).reset_index(drop=True)


def best_stock_setups(
    results: pd.DataFrame, setups: list[dict], min_validation_trades: int, min_win_rate: float = 0.0,
) -> pd.DataFrame:
    """Pick each stock's best setup, requiring positive train and holdout trades."""
    if results is None or results.empty:
        return pd.DataFrame()
    eligible = results[
        (results['Validation Trades'] >= min_validation_trades)
        & (results['Validation Avg Trade'] > 0)
        & (results['Train Avg Trade'] > 0)
        & (results['Validation Win Rate'] >= min_win_rate)
    ].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible['Return / Month'] = eligible['Validation Avg Trade'] / eligible['Avg Days'].clip(lower=1) * 30.44
    by_stock = eligible.groupby('Symbol')

    def stock_rank(column):
        return by_stock[column].rank(pct=True, method='average').fillna(0.0) * 100

    eligible['Stock Setup Score'] = (
        stock_rank('Validation Avg Trade') * 0.35
        + stock_rank('Return / Month') * 0.25
        + stock_rank('Validation Win Rate') * 0.25
        + stock_rank('Avg Drawdown') * 0.15
    )
    best = eligible.sort_values(
        ['Stock Setup Score', 'Validation Avg Trade'], ascending=False,
    ).drop_duplicates('Symbol')
    setup_frame = pd.DataFrame(setups)
    setup_frame.index.name = 'Setup ID'
    best = best.join(setup_frame, on='Setup ID')
    best['Setup'] = [format_setup(setups[setup_id]) for setup_id in best['Setup ID']]
    return best.sort_values('Stock Setup Score', ascending=False).reset_index(drop=True)
