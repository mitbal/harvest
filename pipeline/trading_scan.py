"""Daily job: turn each trading tool's published setups into today's signals.

    python -m pipeline.trading_scan --tool dividend_play

Runs after the daily price update. Signals are written to ``trading_signals``
for the latest session in the price history and are never overwritten on
later days, so they form the public track record.
"""
import argparse
import datetime
import json
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

import harvest.dividend_play as dp
import harvest.trading_store as store
from pipeline import trading_data


logger = logging.getLogger(__name__)

# A stock whose latest close is older than this is skipped instead of signalled.
MAX_PRICE_AGE_DAYS = 7
SIGNAL_STATUS = {dp.SIGNAL_BUY: "actionable", dp.SIGNAL_SOON: "opening_soon"}


def _finite(value) -> Optional[float]:
    value = float(value)
    return value if np.isfinite(value) else None


def cum_date(ex_date) -> datetime.date:
    """IDX cum date: the last trading day before the ex-date."""
    ex_date = pd.Timestamp(ex_date).date()
    days = store.idx_trading_days(ex_date - datetime.timedelta(days=14), ex_date - datetime.timedelta(days=1))
    return days[-1]


def planned_exit(setup: dict, ex_date) -> datetime.date:
    """Last session to hold: the cum date, or the first session ``hold_after_days`` after the ex-date."""
    if setup["exit_rule"] == dp.EXIT_CUM:
        return cum_date(ex_date)
    target = pd.Timestamp(ex_date).date() + datetime.timedelta(days=int(setup["hold_after_days"]))
    return store.idx_trading_days(target, target + datetime.timedelta(days=14))[0]


def window_progress(window_start, window_end, as_of) -> dict:
    """Session N of M inside the entry window, counting IDX trading days."""
    sessions = store.idx_trading_days(window_start, window_end)
    day = sum(1 for session in sessions if session <= pd.Timestamp(as_of).date())
    return {"day": day, "of": len(sessions), "sessions_left": max(len(sessions) - day, 0)}


def dividend_play_signal(
    symbol: str,
    prices: pd.DataFrame,
    divs: pd.DataFrame,
    setup: dict,
    scope: str,
    holdout: dict,
    company: dict,
    as_of: pd.Timestamp,
    opening_within: int = dp.DEFAULT_OPENING_WITHIN,
) -> Optional[dict]:
    """The stored signal for one stock, or None when it is neither actionable nor opening soon."""
    event = dp.next_play_event(divs, setup, as_of)
    if event is None:
        return None
    status = dp.entry_status(
        prices, event["expected_date"], setup["entry_start_days"], setup["entry_end_days"],
        setup["dip_ma_days"], setup["dip_pct"], as_of=as_of,
    )
    signal = dp.scan_signal(status, setup["entry_rule"], opening_within)
    if signal not in SIGNAL_STATUS:
        return None

    close = status["close"]
    entry_price = None
    if signal == dp.SIGNAL_BUY:
        entry_price = close
        if setup["entry_rule"] == dp.ENTRY_DIP and np.isfinite(status["sma"]):
            # The dip rule buys at or below this level; today's close is already under it.
            entry_price = status["sma"] * (1 - setup["dip_pct"] / 100)
    stop_price = target_price = None
    if entry_price is not None:
        if setup["stop_loss_pct"] > 0:
            stop_price = entry_price * (1 - setup["stop_loss_pct"] / 100)
        if setup["take_profit_pct"] > 0:
            target_price = entry_price * (1 + setup["take_profit_pct"] / 100)

    # The setup's record on this stock's whole history, for "historical run-up" copy.
    history = dp.summarize_trades(dp.backtest_dividend_play(
        dp.play_data(prices, divs, dp.DEFAULT_START, as_of), **setup, **dp.trade_costs(),
    ))
    trades = int(holdout.get("trades") or 0)
    win_rate = holdout.get("win_rate")
    detail = {
        "company": company.get("companyName"),
        "yield": _finite(company.get("yield", np.nan)),
        "close": _finite(close),
        "price_date": prices["date"].max().date().isoformat(),
        "expected_ex_date": event["expected_date"].date().isoformat(),
        "based_on": event["based_on"].date().isoformat(),
        "next_dividend": _finite(event["dividend"]),
        "next_yield": _finite(event["dividend"] / close * 100) if close and np.isfinite(close) else None,
        "setup": dp.format_setup(setup),
        "params": setup,
        "window_start": status["window_start"].date().isoformat(),
        "window_end": status["window_end"].date().isoformat(),
        "days_to_window": status["days_to_window"],
        "dip": {
            "rule": setup["entry_rule"] == dp.ENTRY_DIP,
            "ma_days": setup["dip_ma_days"],
            "pct": setup["dip_pct"],
            "active": status["dip_active"],
            "vs_ma_pct": _finite(status["vs_sma"]),
        },
        "holdout": {
            "level": holdout.get("level"),
            "trades": trades,
            "wins": int(round(win_rate * trades / 100)) if win_rate is not None else None,
            "win_rate": win_rate,
            "avg_return": holdout.get("avg_trade"),
        },
        "history": {
            "trades": history["trades"],
            "win_rate": _finite(history.get("win_rate", np.nan)),
            "avg_return": _finite(history.get("avg_return", np.nan)),
            "avg_days": _finite(history.get("avg_days", np.nan)),
        },
    }
    if signal == dp.SIGNAL_BUY:
        detail["window"] = window_progress(status["window_start"], status["window_end"], as_of)

    return {
        "symbol": symbol,
        "status": SIGNAL_STATUS[signal],
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "exit_by": planned_exit(setup, event["expected_date"]),
        "key_date": cum_date(event["expected_date"]),
        "key_date_estimated": not event["announced"],
        "setup_scope": scope,
        "detail": detail,
    }


def dividend_play(client, as_of=None, opening_within: int = dp.DEFAULT_OPENING_WITHIN) -> dict:
    """Publish today's Dividend Play signals from the stored setups.

    ``as_of`` defaults to the latest session in the price history, so a run on
    a holiday or weekend re-publishes the last session instead of inventing one.
    """
    started = time.monotonic()
    setups = store.read_setups(client, "dividend_play")
    events = trading_data.load_dividend_events(client)
    universe = trading_data.load_universe()
    shortlist, _ = dp.build_shortlist(universe, events)
    symbols = shortlist["stock"].tolist()
    histories = trading_data.load_price_histories(
        symbols, pd.Timestamp(dp.DEFAULT_START) - pd.Timedelta(days=dp.PRICE_WARMUP_DAYS),
    )
    if as_of is None:
        if not histories:
            raise RuntimeError("No price history for the Dividend Play shortlist")
        as_of = max(frame["date"].max() for frame in histories.values())
    as_of = pd.Timestamp(as_of).normalize()

    run_id = store.start_run(client, "dividend_play", "scan", as_of)
    try:
        companies = shortlist.set_index("stock").to_dict("index")
        counts = {"no_setup": 0, "no_data": 0, "stale_prices": 0, "not_signalled": 0}
        signals = []
        for symbol in symbols:
            scope = symbol if symbol in setups else "global"
            published = setups.get(scope)
            if published is None:
                counts["no_setup"] += 1
                continue
            prices = histories.get(symbol)
            divs = events.get(symbol)
            if prices is None or prices.empty or divs is None:
                counts["no_data"] += 1
                continue
            prices = prices[prices["date"] <= as_of]
            if prices.empty or prices["date"].max() < as_of - pd.Timedelta(days=MAX_PRICE_AGE_DAYS):
                counts["stale_prices"] += 1
                continue
            signal = dividend_play_signal(
                symbol, prices, divs, published["params"], scope, published["holdout"],
                companies.get(symbol, {}), as_of, opening_within,
            )
            if signal is None:
                counts["not_signalled"] += 1
            else:
                signals.append(signal)

        store.write_signals(client, "dividend_play", as_of, signals)
        stats = {
            "as_of": as_of.date().isoformat(),
            "setups": len(setups),
            "global_setup": "global" in setups,
            "shortlist": len(symbols),
            "actionable": sum(s["status"] == "actionable" for s in signals),
            "opening_soon": sum(s["status"] == "opening_soon" for s in signals),
            **counts,
            "duration_seconds": round(time.monotonic() - started, 1),
        }
        store.finish_run(client, run_id, "ok", stats)
        return stats
    except Exception as exc:
        store.finish_run(client, run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"[:500]})
        raise


TOOLS = {"dividend_play": dividend_play}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish today's trading signals")
    parser.add_argument("--tool", choices=sorted(TOOLS), required=True)
    parser.add_argument("--as-of", help="Session to scan (YYYY-MM-DD), default the latest price date")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        trading_data.validate_environment()
        stats = TOOLS[args.tool](trading_data.get_supabase_client(), as_of=args.as_of)
        logger.info("Scan summary for %s: %s", args.tool, json.dumps(stats, sort_keys=True, default=str))
        return 0
    except Exception:
        logger.exception("Scan job failed for %s", args.tool)
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    sys.exit(main())
