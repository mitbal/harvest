"""Portfolio simulation: a backtest's trades taken with one account instead of one at a time.

A per-trade average assumes every trade can be taken in full. A real account has fixed capital,
room for a few positions, buys in 100-share lots and can only take a small share of a stock's
daily trading without moving the price. This replays a list of trades under those limits and
returns the account's equity curve next to the JKSE.

Rules, in order on each session:

1. Entries (alphabetical when several start the same day). A trade is skipped when the stock is
   already held or every slot is taken. Otherwise it gets the smaller of the cash and one slot's
   share of the account (equity / max positions), in whole lots, capped at ``liquidity_pct`` of the
   stock's average daily traded value over the previous ``LIQUIDITY_DAYS`` sessions. A trade that
   can't afford one lot is skipped.
2. Exits on or after the trade's exit date return the cost times ``1 + return_pct / 100``, so the
   trade's own costs, slippage and dividends carry over.
3. Open positions are marked at the session's close.

Entries come before exits, so a slot or cash freed on a day is used from the next day; this is
slightly conservative for trades that enter at the open.
"""
import numpy as np
import pandas as pd

from harvest.swing import LOT_SIZE


DEFAULT_CAPITAL = 100_000_000
DEFAULT_MAX_POSITIONS = 5
LIQUIDITY_DAYS = 20
DEFAULT_LIQUIDITY_PCT = 5.0
TRADE_COLUMNS = ['symbol', 'entry_date', 'exit_date', 'entry_price', 'return_pct']


def average_traded_value(bars: pd.DataFrame, days: int = LIQUIDITY_DAYS) -> pd.Series:
    """Average daily traded value (close x volume) of the previous ``days`` sessions, by date."""
    if 'volume' not in bars.columns:
        return pd.Series(np.nan, index=bars.index)
    value = pd.to_numeric(bars['close'], errors='coerce') * pd.to_numeric(bars['volume'], errors='coerce')
    return value.rolling(days, min_periods=max(days // 2, 1)).mean().shift(1)


def max_drawdown(values: pd.Series) -> float | None:
    """Worst peak-to-trough fall of a value series, in percent (negative)."""
    values = values.dropna()
    if values.empty:
        return None
    return float((values / values.cummax() - 1).min() * 100)


def _growth(values: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    values = values.dropna()
    if len(values) < 2:
        return {'total_return': None, 'cagr': None, 'max_drawdown': None}
    total = values.iloc[-1] / values.iloc[0] - 1
    years = max((end - start).days / 365.25, 1 / 365.25)
    return {
        'total_return': float(total * 100),
        'cagr': float(((1 + total) ** (1 / years) - 1) * 100) if total > -1 else -100.0,
        'max_drawdown': max_drawdown(values),
    }


def simulate_portfolio(
    trades: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    capital: float = DEFAULT_CAPITAL,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    liquidity_pct: float = DEFAULT_LIQUIDITY_PCT,
    benchmark: pd.Series | None = None,
    start=None,
    end=None,
) -> dict:
    """Replay ``trades`` (``TRADE_COLUMNS``) as one account; returns stats, settings and a weekly curve.

    ``bars`` are OHLCV frames by date (``close`` and ``volume`` are used). ``benchmark`` is the
    JKSE close by date. The curve is indexed to 100 at ``start`` for both the account and the JKSE.
    """
    trades = trades.reindex(columns=list(dict.fromkeys([*TRADE_COLUMNS, *trades.columns])))
    trades['entry_date'] = pd.to_datetime(trades['entry_date']).dt.normalize()
    trades['exit_date'] = pd.to_datetime(trades['exit_date']).dt.normalize()
    trades = trades.dropna(subset=['entry_date', 'exit_date', 'entry_price', 'return_pct'])
    trades = trades[trades['entry_price'] > 0]
    settings = {'capital': float(capital), 'max_positions': int(max_positions), 'liquidity_pct': float(liquidity_pct),
                'liquidity_days': LIQUIDITY_DAYS, 'lot_size': LOT_SIZE}
    if trades.empty:
        return {'settings': settings, 'stats': None, 'curve': []}

    start = pd.Timestamp(start).normalize() if start is not None else trades['entry_date'].min()
    end = pd.Timestamp(end).normalize() if end is not None else trades['exit_date'].max()
    trades = trades[(trades['entry_date'] >= start) & (trades['entry_date'] <= end)]
    symbols = sorted(set(trades['symbol']) & set(bars))
    trades = trades[trades['symbol'].isin(symbols)].sort_values(['entry_date', 'symbol'])
    calendar = sorted({day for symbol in symbols for day in bars[symbol].index if start <= day <= end})
    if not calendar:
        return {'settings': settings, 'stats': None, 'curve': []}
    calendar = pd.DatetimeIndex(calendar)
    closes = pd.DataFrame({s: bars[s]['close'] for s in symbols}).reindex(calendar).ffill()
    liquidity = pd.DataFrame({s: average_traded_value(bars[s]) for s in symbols}).reindex(calendar)

    entries = {day: group for day, group in trades.groupby('entry_date')}
    cash, held = float(capital), {}
    equity = float(capital)
    curve, counts, invested = [], [], []
    taken, results = 0, []
    skipped = {'held': 0, 'full': 0, 'too_small': 0}
    capped = 0
    for day in calendar:
        for trade in entries.get(day, pd.DataFrame(columns=TRADE_COLUMNS)).itertuples(index=False):
            if trade.symbol in held:
                skipped['held'] += 1
                continue
            if len(held) >= max_positions:
                skipped['full'] += 1
                continue
            lot_cost = float(trade.entry_price) * LOT_SIZE
            lots = int(min(cash, equity / max_positions) // lot_cost)
            traded_value = liquidity.at[day, trade.symbol]
            if pd.notna(traded_value):
                limit = int(traded_value * liquidity_pct / 100 // lot_cost)
                if limit < lots:
                    lots, capped = limit, capped + 1
            if lots < 1:
                skipped['too_small'] += 1
                continue
            cost = lots * lot_cost
            cash -= cost
            held[trade.symbol] = {'shares': lots * LOT_SIZE, 'cost': cost, 'exit_date': trade.exit_date,
                                  'return_pct': float(trade.return_pct)}
            taken += 1
        for symbol in [s for s, p in held.items() if p['exit_date'] <= day]:
            position = held.pop(symbol)
            proceeds = position['cost'] * (1 + position['return_pct'] / 100)
            cash += proceeds
            results.append(proceeds - position['cost'])
        marked = sum(p['shares'] * closes.at[day, s] if pd.notna(closes.at[day, s]) else p['cost']
                     for s, p in held.items())
        equity = cash + marked
        curve.append(equity)
        counts.append(len(held))
        invested.append(marked / equity if equity > 0 else 0.0)

    account = pd.Series(curve, index=calendar)
    stats = {
        'start': calendar[0].date().isoformat(),
        'end': calendar[-1].date().isoformat(),
        'final_equity': float(account.iloc[-1]),
        **_growth(account, calendar[0], calendar[-1]),
        'trades_offered': int(len(trades)),
        'trades_taken': taken,
        'skipped': skipped,
        'liquidity_capped': capped,
        'wins': int(sum(r > 0 for r in results)),
        'closed': len(results),
        'avg_positions': float(np.mean(counts)),
        'avg_invested_pct': float(np.mean(invested) * 100),
    }
    index = None
    if benchmark is not None and not benchmark.empty:
        index = pd.Series(benchmark).sort_index().reindex(calendar, method='ffill')
        if index.notna().sum() >= 2:
            stats['benchmark'] = _growth(index.ffill().bfill(), calendar[0], calendar[-1])
        else:
            index = None

    weekly = account.resample('W-FRI').last().dropna()
    frame = pd.DataFrame({'account': weekly / capital * 100})
    if index is not None:
        base = index.ffill().bfill()
        frame['jkse'] = (base / base.iloc[0] * 100).resample('W-FRI').last().reindex(frame.index)
    return {
        'settings': settings,
        'stats': stats,
        'curve': [{'date': day.date().isoformat(), **{k: (None if pd.isna(v) else round(float(v), 2))
                                                      for k, v in row.items()}}
                  for day, row in frame.iterrows()],
    }
