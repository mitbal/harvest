"""Realised outcomes of published trading signals: what following each card would have returned.

Each tool's cards are replayed against the prices that came after them, using only what the
card told the reader:

- Dividend Play: a limit order at the card's "entry at or below" price (rounded down to an IDX
  tick) on the session after the signal. If the low reaches it, the fill is the open or the
  limit, whichever is lower. An unfilled card is retried with the next day's card while the
  stock stays in its buy zone; when the streak ends unfilled it counts as ``not_filled``. A filled
  trade is sold at the close of its sell-by date (or earlier on the setup's take profit or stop
  loss), and a dividend whose ex-date falls inside the holding period is received after tax.
- Swing: buy at the next open, stop and target as percentages of the fill, sell at the close
  after the maximum hold, with the backtest's fill rules (``swing.simulate_trades``), fees and
  tick slippage. The strategy's own early exit is not on the card, so it is not used. A
  limit-entry card buys only if the next session trades below its limit (at the open or the
  limit, whichever is lower) and sells at the open after the maximum hold; an order the next
  session doesn't fill counts as ``not_filled``.
- Long Term: bought at the close of the day a company first appears on the screen, closed when
  the price reaches the valuation target on that first card; otherwise still open.

One position per stock at a time, as in the backtests: cards for a stock already held are part
of that trade. Returns are percentages after costs; ``benchmark_pct`` is the JKSE price index
over the same days. Open trades are marked to the latest close.
"""
import numpy as np
import pandas as pd

import harvest.dividend_play as dp
import harvest.swing as sw


STATUSES = ('open', 'closed', 'not_filled')
FINAL_STATUSES = ('closed', 'not_filled')
OUTCOME_COLUMNS = [
    'tool', 'symbol', 'signal_date', 'status', 'entry_date', 'entry_price', 'exit_date', 'exit_price',
    'exit_reason', 'return_pct', 'benchmark_pct', 'detail',
]
SWING_COSTS = {'buy_fee': 0.15, 'sell_fee': 0.15, 'sell_tax': 0.1, 'slippage_ticks': sw.DEFAULT_SLIPPAGE_TICKS}

REASON_SELL_BY = 'Sell-by date'
REASON_TARGET_VALUE = 'Reached mean valuation'


def _ts(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _day(value) -> str | None:
    return None if value is None or pd.isna(value) else _ts(value).date().isoformat()


def benchmark_return(benchmark: pd.Series | None, start, end) -> float | None:
    """JKSE price change from the close on ``start`` to the close on ``end`` (latest on or before each)."""
    if benchmark is None or benchmark.empty:
        return None
    before_start = benchmark[benchmark.index <= _ts(start)]
    before_end = benchmark[benchmark.index <= _ts(end)]
    if before_start.empty or before_end.empty:
        return None
    return float((before_end.iloc[-1] / before_start.iloc[-1] - 1) * 100)


def _outcome(tool: str, symbol: str, signal_date, status: str, **fields) -> dict:
    row = {column: None for column in OUTCOME_COLUMNS}
    row.update({'tool': tool, 'symbol': symbol, 'signal_date': _day(signal_date), 'status': status, 'detail': {}})
    row.update(fields)
    for column in ('entry_date', 'exit_date'):
        row[column] = _day(row[column])
    return row


def _expected(detail: dict | None) -> float | None:
    return ((detail or {}).get('holdout') or {}).get('avg_return')


# ── Dividend Play ─────────────────────────────────────────────────────────── #

def _dividend_trade(symbol, first, signal, fill_date, fill, bars, dividends, costs, benchmark, signals_seen) -> dict:
    detail = signal.detail or {}
    params = detail.get('params') or {}
    exit_by = _ts(signal.exit_by)
    closes = bars['close']
    path = closes[(closes.index > fill_date) & (closes.index <= exit_by)]
    divs = dividends if dividends is not None else pd.DataFrame(columns=['date', 'dividend'])
    divs = divs[pd.to_datetime(divs['date']) > fill_date]
    earned = np.zeros(len(path))
    for ex_date, amount in zip(pd.to_datetime(divs['date']), divs['dividend']):
        earned += np.where(path.index >= ex_date, float(amount), 0.0)

    exit_i, reason = len(path) - 1, REASON_SELL_BY
    if len(path):
        path_returns = (path.to_numpy() + earned) / fill - 1
        take_profit, stop_loss = float(params.get('take_profit_pct') or 0), float(params.get('stop_loss_pct') or 0)
        if take_profit > 0:
            hits = np.flatnonzero(path_returns * 100 >= take_profit)
            if len(hits) and hits[0] < exit_i:
                exit_i, reason = int(hits[0]), 'Take profit'
        if stop_loss > 0:
            hits = np.flatnonzero(path_returns * 100 <= -stop_loss)
            if len(hits) and (hits[0] < exit_i or (hits[0] == exit_i and reason != 'Take profit')):
                exit_i, reason = int(hits[0]), 'Stop loss'
    # Sold on the sell-by date once prices reach it; otherwise still held.
    closed = reason != REASON_SELL_BY or (len(path) and closes.index[-1] >= exit_by)
    if len(path):
        mark_date, mark_price, dividend = path.index[exit_i], float(path.iloc[exit_i]), float(earned[exit_i])
    else:
        mark_date, mark_price, dividend = fill_date, float(closes.loc[fill_date]), 0.0
    cost_basis = fill * (1 + costs['buy_fee_pct'] / 100)
    proceeds = mark_price * (1 - costs['sell_fee_pct'] / 100) + dividend * (1 - costs['dividend_tax_pct'] / 100)
    return _outcome(
        'dividend_play', symbol, first.as_of, 'closed' if closed else 'open',
        entry_date=fill_date, entry_price=float(fill),
        exit_date=mark_date if closed else None, exit_price=mark_price if closed else None,
        exit_reason=reason if closed else None,
        return_pct=float((proceeds / cost_basis - 1) * 100),
        benchmark_pct=benchmark_return(benchmark, fill_date, mark_date),
        detail={
            'limit': float(sw.round_to_tick(signal.entry_price, 'down')), 'sell_by': _day(exit_by),
            'dividend': dividend or None, 'cards': signals_seen, 'setup': detail.get('setup'),
            'expected': _expected(detail), 'marked_at': None if closed else _day(mark_date),
            'last_close': None if closed else mark_price,
        },
    )


def dividend_play_outcomes(
    signals: pd.DataFrame,
    scan_dates: list,
    bars: dict[str, pd.DataFrame],
    dividends: dict[str, pd.DataFrame],
    benchmark: pd.Series | None = None,
    costs: dict | None = None,
) -> list[dict]:
    """Outcomes of every actionable Dividend Play card, oldest first per stock.

    ``signals`` are the stored actionable rows; ``scan_dates`` every successful scan day, so a
    buy-zone streak is known to have ended; ``bars`` OHLC frames indexed by date.
    """
    costs = costs or dp.trade_costs()
    scans = sorted(_ts(day) for day in scan_dates)
    next_scan = dict(zip(scans[:-1], scans[1:]))
    outcomes = []
    for symbol, group in signals.assign(as_of=pd.to_datetime(signals['as_of'])).sort_values('as_of').groupby('symbol'):
        frame = bars.get(symbol)
        if frame is None or frame.empty:
            continue
        rows = list(group.itertuples(index=False))
        held_until, streak = None, []
        for k, signal in enumerate(rows):
            as_of = _ts(signal.as_of)
            # A card on the day a trade is sold can be bought the next session.
            if held_until is not None and as_of < held_until:
                continue
            streak.append(signal)
            later = frame.index[frame.index > as_of]
            if not len(later):
                break  # Published after the latest close; nothing to check yet.
            session = later[0]
            limit = float(sw.round_to_tick(signal.entry_price, 'down'))
            if frame.at[session, 'low'] <= limit:
                fill = min(float(frame.at[session, 'open']), limit)
                trade = _dividend_trade(symbol, streak[0], signal, session, fill, frame, dividends.get(symbol),
                                        costs, benchmark, len(streak))
                outcomes.append(trade)
                if trade['status'] == 'open':
                    break
                held_until, streak = _ts(trade['exit_date']), []
                continue
            following = rows[k + 1] if k + 1 < len(rows) else None
            continues = following is not None and next_scan.get(as_of) == _ts(following.as_of)
            if not continues and as_of in next_scan:
                outcomes.append(_outcome(
                    'dividend_play', symbol, streak[0].as_of, 'not_filled',
                    detail={'limit': limit, 'cards': len(streak), 'last_card': _day(as_of),
                            'setup': (signal.detail or {}).get('setup'), 'expected': _expected(signal.detail)},
                ))
                streak = []
    return outcomes


# ── Swing ─────────────────────────────────────────────────────────────────── #

def _swing_return(entry: float, exit_price: float, costs: dict) -> float:
    entry_cost = costs['buy_fee'] / 100 + float(sw.slippage_fraction(pd.Series([entry]), costs['slippage_ticks']).iloc[0])
    exit_cost = ((costs['sell_fee'] + costs['sell_tax']) / 100
                 + float(sw.slippage_fraction(pd.Series([exit_price]), costs['slippage_ticks']).iloc[0]))
    return float((exit_price * (1 - exit_cost) / entry - (1 + entry_cost)) * 100)


def swing_outcomes(
    signals: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    benchmark: pd.Series | None = None,
    costs: dict | None = None,
) -> list[dict]:
    """Outcomes of every actionable swing card, with the backtest's fills (next open or the card's limit)."""
    costs = costs or SWING_COSTS
    outcomes = []
    for symbol, group in signals.assign(as_of=pd.to_datetime(signals['as_of'])).sort_values('as_of').groupby('symbol'):
        if symbol not in bars or bars[symbol].empty:
            continue
        frame, _ = sw.ohlc_frame(bars[symbol])
        held_until = None
        for signal in group.itertuples(index=False):
            as_of = _ts(signal.as_of)
            if held_until is not None and as_of < held_until:
                continue
            window = frame[frame.index >= as_of]
            if len(window) < 2 or window.index[0] != as_of:
                continue  # No session after the card yet, or no bar on the signal day.
            detail = signal.detail or {}
            fired = pd.Series(window.index == as_of, index=window.index)
            limit = None
            if detail.get('entry') == 'limit':
                limit = pd.Series(np.nan, index=window.index)
                limit.iloc[0] = float(signal.entry_price)
            raw = sw.simulate_trades(
                window, fired, pd.Series(False, index=window.index), int(detail.get('max_hold') or sw.DEFAULT_MAX_HOLD),
                float(detail.get('target_pct') or 0), float(detail.get('stop_pct') or 0), has_ohlc=True,
                entry_limit=limit, exit_at_open=detail.get('exit') == 'open',
            )
            if limit is not None and not raw and not window['low'].iloc[1] < limit.iloc[0]:
                outcomes.append(_outcome(
                    'swing', symbol, as_of, 'not_filled', exit_reason='Limit not reached',
                    detail={'limit': float(signal.entry_price), 'cards': 1, 'setup': detail.get('setup'),
                            'expected': _expected(detail), 'session': _day(window.index[1])},
                ))
                continue
            if raw:
                trade = raw[0]
                entry_date, exit_date = window.index[trade['entry_i']], window.index[trade['exit_i']]
                entry, exit_price, reason = trade['entry_price'], trade['exit_price'], trade['exit_reason']
            else:  # Bought in the latest session and still held.
                entry_date = exit_date = window.index[1]
                entry = float(window['open'].iloc[1]) if limit is None else min(float(window['open'].iloc[1]),
                                                                                 float(limit.iloc[0]))
                exit_price, reason = float(window['close'].iloc[1]), 'End of data'
            closed = reason != 'End of data'
            outcomes.append(_outcome(
                'swing', symbol, as_of, 'closed' if closed else 'open',
                entry_date=entry_date, entry_price=float(entry),
                exit_date=exit_date if closed else None, exit_price=float(exit_price) if closed else None,
                exit_reason=reason if closed else None,
                return_pct=_swing_return(float(entry), float(exit_price), costs),
                benchmark_pct=benchmark_return(benchmark, entry_date, exit_date),
                detail={
                    'stop_pct': detail.get('stop_pct'), 'target_pct': detail.get('target_pct'),
                    'max_hold': detail.get('max_hold'), 'sell_by': _day(signal.exit_by),
                    'sessions': int(window.index.get_loc(exit_date) - window.index.get_loc(entry_date)),
                    'setup': detail.get('setup'), 'expected': _expected(detail),
                    'marked_at': None if closed else _day(exit_date), 'last_close': None if closed else float(exit_price),
                },
            ))
            if not closed:
                break
            held_until = exit_date
    return outcomes


# ── Long Term ─────────────────────────────────────────────────────────────── #

def long_term_outcomes(
    signals: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    benchmark: pd.Series | None = None,
    costs: dict | None = None,
) -> list[dict]:
    """Each company from the close of its first day on the screen until it reaches its target."""
    costs = costs or dp.trade_costs()
    buy_keep, sell_keep = 1 + costs['buy_fee_pct'] / 100, 1 - costs['sell_fee_pct'] / 100
    outcomes = []
    for symbol, group in signals.assign(as_of=pd.to_datetime(signals['as_of'])).sort_values('as_of').groupby('symbol'):
        frame = bars.get(symbol)
        if frame is None or frame.empty:
            continue
        closes = frame['close']
        held_until = None
        for signal in group.itertuples(index=False):
            as_of = _ts(signal.as_of)
            if held_until is not None and as_of <= held_until:
                continue
            entry, target = float(signal.entry_price), signal.target_price
            path = closes[closes.index > as_of]
            hits = path[path >= float(target)] if target is not None and not pd.isna(target) else path.iloc[:0]
            closed = not hits.empty
            if closed:
                mark_date, mark_price = hits.index[0], float(hits.iloc[0])
            elif not path.empty:
                mark_date, mark_price = path.index[-1], float(path.iloc[-1])
            else:
                mark_date, mark_price = as_of, entry
            detail = getattr(signal, 'detail', None) or {}
            outcomes.append(_outcome(
                'long_term', symbol, as_of, 'closed' if closed else 'open',
                entry_date=as_of, entry_price=entry,
                exit_date=mark_date if closed else None, exit_price=mark_price if closed else None,
                exit_reason=REASON_TARGET_VALUE if closed else None,
                return_pct=float((mark_price * sell_keep / (entry * buy_keep) - 1) * 100),
                benchmark_pct=benchmark_return(benchmark, as_of, mark_date),
                detail={
                    'target': None if target is None or pd.isna(target) else float(target),
                    'rank': detail.get('rank'), 'company': detail.get('company'), 'sector': detail.get('sector'),
                    'marked_at': None if closed else _day(mark_date), 'last_close': None if closed else mark_price,
                },
            ))
            if not closed:
                break
            held_until = mark_date
    return outcomes


# ── Summaries (page) ──────────────────────────────────────────────────────── #

def summarize(outcomes: pd.DataFrame) -> dict:
    """Headline numbers over closed trades, plus counts of open and unfilled cards."""
    closed = outcomes[outcomes['status'] == 'closed'] if not outcomes.empty else outcomes
    returns = pd.to_numeric(closed['return_pct'], errors='coerce').dropna() if not closed.empty else pd.Series(dtype=float)
    expected = closed['detail'].map(lambda d: (d or {}).get('expected')).dropna() if not closed.empty else pd.Series(dtype=float)
    benchmark = pd.to_numeric(closed['benchmark_pct'], errors='coerce').dropna() if not closed.empty else pd.Series(dtype=float)
    count = lambda status: int((outcomes['status'] == status).sum()) if not outcomes.empty else 0
    return {
        'closed': len(returns),
        'wins': int((returns > 0).sum()),
        'win_rate': float((returns > 0).mean() * 100) if len(returns) else None,
        'avg_return': float(returns.mean()) if len(returns) else None,
        'expected': float(pd.to_numeric(expected).mean()) if len(expected) else None,
        'avg_benchmark': float(benchmark.mean()) if len(benchmark) else None,
        'open': count('open'),
        'not_filled': count('not_filled'),
    }
