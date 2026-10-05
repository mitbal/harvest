"""Daily job: turn each trading tool's published setups into today's signals.

    python -m pipeline.trading_scan --tool dividend_play

Runs after the daily price update. Signals are written to ``trading_signals``
for the latest session in the price history and are never overwritten on
later days, so they form the public track record.
"""
import argparse
import concurrent.futures
import datetime
import json
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

import harvest.alerts as alerts
import harvest.dividend_play as dp
import harvest.position as pos
import harvest.swing as sw
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
        try:
            alert_stats = alerts.send_dividend_play_alerts(client, as_of, signals)
        except Exception as exc:
            # Signals are published; a Telegram problem must not fail the run.
            logger.exception("Dividend Play alerts failed")
            alert_stats = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        stats = {
            "as_of": as_of.date().isoformat(),
            "setups": len(setups),
            "global_setup": "global" in setups,
            "shortlist": len(symbols),
            "actionable": sum(s["status"] == "actionable" for s in signals),
            "opening_soon": sum(s["status"] == "opening_soon" for s in signals),
            **counts,
            "alerts": alert_stats,
            "duration_seconds": round(time.monotonic() - started, 1),
        }
        store.finish_run(client, run_id, "ok", stats)
        return stats
    except Exception as exc:
        store.finish_run(client, run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"[:500]})
        raise


# How close an indicator must be to its trigger to make the watch list (as in the Lab scanner).
SWING_WATCH_PROXIMITY = 5.0
SWING_HISTORY_DAYS = 500


def swing_signal(
    symbol: str,
    frame: pd.DataFrame,
    regime: pd.Series,
    setup: dict,
    scope: str,
    holdout: dict,
    company: dict,
    as_of: pd.Timestamp,
    benchmark: pd.Series,
    breadth: pd.Series,
    momentum: Optional[pd.Series] = None,
) -> Optional[dict]:
    """Today's swing signal for one stock: actionable when the setup fired at the latest close.

    Entry is the next session's open, as in the backtest, so stop and target
    are percentages of that fill; the card's reference prices use the latest
    close, rounded to IDX ticks, with the next session's auto-reject band.
    """
    bars, _ = sw.ohlc_frame(frame)
    if bars.empty or bars.index[-1] != as_of:
        return None
    params = [setup[k] for k in ("param_a", "param_b", "param_c", "param_d")]
    aligned = regime.reindex(bars.index, method="ffill").fillna("bear") if not regime.empty \
        else pd.Series("bear", index=bars.index)
    fired, _, indicator = sw.strategy_components(
        bars["close"], aligned, benchmark, momentum, setup["strategy_type"], *params, breadth=breadth,
    )
    indicator_now = float(indicator.iloc[-1]) if pd.notna(indicator.iloc[-1]) else np.nan
    if bool(fired.iloc[-1]):
        status = "actionable"
    elif sw.is_near_setup(setup["strategy_type"], indicator_now, tuple(params), SWING_WATCH_PROXIMITY,
                          regime=str(aligned.iloc[-1])):
        status = "opening_soon"
    else:
        return None

    close = float(bars["close"].iloc[-1])
    next_session = store.idx_trading_days(as_of + pd.Timedelta(days=1), as_of + pd.Timedelta(days=14))[0]
    sessions = store.idx_trading_days(next_session, next_session + pd.Timedelta(days=60))
    floor, ceiling = sw.price_limits(close, next_session)
    stop = sw.round_to_tick(close * (1 - setup["stop_loss"] / 100), "down") if setup["stop_loss"] > 0 else None
    target = sw.round_to_tick(close * (1 + setup["profit_target"] / 100), "down") if setup["profit_target"] > 0 else None
    trades = int(holdout.get("trades") or 0)
    win_rate = holdout.get("win_rate")
    detail = {
        "company": company.get("companyName"),
        "close": close,
        "price_date": as_of.date().isoformat(),
        "entry": "next_open",
        "entry_session": next_session.isoformat(),
        "setup": f"{setup['strategy_type']}: {sw.format_strategy_params(setup['strategy_type'], *params)}",
        "params": setup,
        "stop_pct": setup["stop_loss"],
        "target_pct": setup["profit_target"],
        "max_hold": int(setup["max_hold"]),
        "indicator": _finite(indicator_now) if np.isfinite(indicator_now) else None,
        "regime": str(aligned.iloc[-1]),
        "price_limits": {"floor": floor, "ceiling": ceiling},
        "stop_below_floor": bool(stop is not None and stop < floor),
        "daily_volatility": _finite(company.get("Daily Volatility", np.nan)),
        "holdout": {
            "level": holdout.get("level"),
            "method": holdout.get("method"),
            "trades": trades,
            "wins": int(round(win_rate * trades / 100)) if win_rate is not None and trades else None,
            "win_rate": win_rate,
            "avg_return": holdout.get("avg_trade"),
            "profit_factor": holdout.get("profit_factor"),
        },
    }
    return {
        "symbol": symbol,
        "status": status,
        "entry_price": close if status == "actionable" else None,
        "stop_price": stop if status == "actionable" else None,
        "target_price": target if status == "actionable" else None,
        "exit_by": sessions[min(int(setup["max_hold"]), len(sessions) - 1)] if status == "actionable" else None,
        "key_date": next_session if status == "actionable" else None,
        "key_date_estimated": False,
        "setup_scope": scope,
        "detail": detail,
    }


def swing(client, as_of=None) -> dict:
    """Publish today's swing signals from the stored setups (latest price session by default)."""
    from pipeline.trading_optimize import swing_inputs

    started = time.monotonic()
    setups = store.read_setups(client, "swing")
    latest = pd.Timestamp(as_of or datetime.date.today()).normalize()
    shortlist, histories, regime, breadth, benchmark = swing_inputs(
        client, latest, latest - pd.Timedelta(days=SWING_HISTORY_DAYS),
    )
    if not histories:
        raise RuntimeError("No price history for the swing shortlist")
    if as_of is None:
        latest = max(frame.index.max() for frame in histories.values())
    as_of = pd.Timestamp(latest).normalize()

    run_id = store.start_run(client, "swing", "scan", as_of)
    try:
        companies = shortlist.set_index("stock").to_dict("index")
        momentum = sw.build_momentum_percentiles(
            {symbol: frame["close"] for symbol, frame in histories.items()},
            {int(p["params"]["param_a"]) for p in setups.values()
             if p["params"]["strategy_type"] == sw.STRATEGY_CROSS_MOMENTUM},
        )
        counts = {"no_setup": 0, "stale_prices": 0, "not_signalled": 0}
        signals = []
        for symbol in shortlist["stock"]:
            scope = symbol if symbol in setups else "global"
            published = setups.get(scope)
            frame = histories.get(symbol)
            if published is None:
                counts["no_setup"] += 1
                continue
            if frame is None or frame.empty or frame.index.max() < as_of:
                counts["stale_prices"] += 1
                continue
            ranks = momentum.get(int(published["params"]["param_a"]))
            signal = swing_signal(
                symbol, frame[frame.index <= as_of], regime, published["params"], scope, published["holdout"],
                companies.get(symbol, {}), as_of, benchmark, breadth,
                momentum=ranks[symbol] if ranks is not None and symbol in ranks else None,
            )
            if signal is None:
                counts["not_signalled"] += 1
            else:
                signals.append(signal)

        store.write_signals(client, "swing", as_of, signals)
        stats = {
            "as_of": as_of.date().isoformat(),
            "setups": len(setups),
            "global_setup": "global" in setups,
            "shortlist": len(shortlist),
            "actionable": sum(s["status"] == "actionable" for s in signals),
            "watch": sum(s["status"] == "opening_soon" for s in signals),
            **counts,
            "duration_seconds": round(time.monotonic() - started, 1),
        }
        store.finish_run(client, run_id, "ok", stats)
        return stats
    except Exception as exc:
        store.finish_run(client, run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"[:500]})
        raise


# ── Long Term ───────────────────────────────────────────────────────────── #

# The published screen analyzes more companies than the page's default form.
LONG_TERM_MAX_COMPANIES = 200
LONG_TERM_DEFAULTS = {**pos.DEFAULT_FILTERS, "max_companies": LONG_TERM_MAX_COMPANIES}
MAX_FAILURES_LISTED = 20


def long_term_screen(setups: dict) -> dict:
    """The published screen's filters (global setup), else the defaults."""
    return {**LONG_TERM_DEFAULTS, **((setups.get("global") or {}).get("params") or {})}


def analyze_long_term(
    profiles: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    history_years: int,
) -> tuple[dict[str, dict], dict]:
    """Screen rows by symbol plus coverage: skipped by reason and failed symbols with the error.

    A stock is *skipped* when its data is insufficient (no prices, too few
    statements, too short a valuation history) and *failed* when fetching or
    analyzing it raised.
    """
    def analyze(symbol: str, profile: dict) -> dict:
        prices = histories.get(symbol)
        if prices is None or prices.empty:
            raise pos.SkipStock("no prices")
        financials, statements = trading_data.load_long_term_fundamentals(symbol)
        return pos.analyze_frames(symbol, profile, prices, financials, history_years, statements=statements)

    rows, skipped, failed = {}, {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=pos.SCAN_WORKERS) as executor:
        futures = {
            executor.submit(analyze, str(symbol), dict(profile)): str(symbol)
            for symbol, profile in profiles.iterrows()
        }
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                rows[symbol] = future.result()
            except pos.SkipStock as exc:
                skipped.setdefault(str(exc), []).append(symbol)
            except Exception as exc:
                failed[symbol] = f"{type(exc).__name__}: {exc}"[:200]
    coverage = {
        "requested": len(profiles),
        "analyzed": len(rows),
        "skipped": sum(len(symbols) for symbols in skipped.values()),
        "skipped_by_reason": {reason: len(symbols) for reason, symbols in sorted(skipped.items())},
        "failed": len(failed),
        "failed_symbols": dict(sorted(failed.items())[:MAX_FAILURES_LISTED]),
    }
    return rows, coverage


def _quality_detail(row: dict) -> dict:
    coverage = row.get("Interest Coverage", np.nan)
    return {
        "debt_to_equity": row.get("Debt to Equity"),
        "interest_coverage": coverage,
        "no_interest_expense": bool(coverage == np.inf),
        "cash_conversion": row.get("Cash Conversion"),
        "roe": row.get("ROE"),
        "share_change": row.get("Share Change 1Y"),
        "complete": bool(row.get("Quality Data")),
    }


def long_term_signal(row: pd.Series, rank: int, close: float, price_date, holdout: dict) -> dict:
    """A ranked candidate: entry at today's close, target at the historical mean valuation."""
    basis = row["Value Basis"]
    second = "PB" if basis == "P/B" else "PS"
    detail = {
        "company": row["Company"],
        "sector": row["Sector"],
        "rank": rank,
        "close": close,
        "price_date": pd.Timestamp(price_date).date().isoformat(),
        "market_cap": row["Market Cap"],
        "basis": basis,
        "scores": {
            "score": row["Score"], "valuation": row["Valuation Score"],
            "growth": row["Growth Score"], "quality": row["Quality Score"],
        },
        "pe": {
            "current": row["Current PE"], "mean": row["PE Mean"], "median": row["PE Median"],
            "discount": row["PE Discount"], "percentile": row["PE Percentile"],
        },
        "second": {
            "label": basis, "current": row[f"Current {second}"], "mean": row[f"{second} Mean"],
            "median": row[f"{second} Median"], "discount": row[f"{second} Discount"],
            "percentile": row[f"{second} Percentile"],
        },
        "growth": {
            "revenue_5y": row["Revenue Growth 5Y"], "revenue_ttm": row["Revenue Growth TTM"],
            "earnings_5y": row["Earnings Growth 5Y"], "earnings_ttm": row["Earnings Growth TTM"],
            "margin_ttm": row["Margin TTM"],
        },
        "quality": _quality_detail(row),
        "upside": {
            "to_mean": row["Upside to Mean"], "potential_1y": row["Potential Upside 1Y"],
            "potential_5y": row["Potential Upside 5Y"],
            "implied_1y": row["Implied Price 1Y"], "implied_5y": row["Implied Price 5Y"],
        },
        "backtest": holdout or None,
    }
    return {
        "symbol": row["Symbol"],
        "status": "candidate",
        "entry_price": close,
        "stop_price": None,
        "target_price": close * (1 + row["Upside to Mean"] / 100),
        "exit_by": None,
        "key_date": row["Statement Date"],
        "key_date_estimated": False,
        "setup_scope": "global",
        "detail": detail,
    }


def long_term(client, as_of=None) -> dict:
    """Publish today's ranked Long Term candidates and send exit alerts for tracked positions."""
    started = time.monotonic()
    setups = store.read_setups(client, "long_term")
    screen = long_term_screen(setups)
    history_years = int(screen["history_years"])
    universe = trading_data.load_long_term_universe()
    selected = pos.select_universe(universe, screen["min_market_cap"], screen["max_companies"])
    positions = store.open_positions_all(client, "long_term")
    extra = sorted(set(positions["symbol"]) - set(selected.index))
    profiles = pd.concat([selected, universe.loc[universe.index.intersection(extra)]])
    histories = trading_data.load_price_histories(
        profiles.index.tolist(), pos.price_start(history_years),
    )
    if not histories:
        raise RuntimeError("No price history for the Long Term universe")
    if as_of is None:
        as_of = max(frame["date"].max() for frame in histories.values())
    as_of = pd.Timestamp(as_of).normalize()
    histories = {symbol: frame[frame["date"] <= as_of] for symbol, frame in histories.items()}

    run_id = store.start_run(client, "long_term", "scan", as_of)
    try:
        rows, coverage = analyze_long_term(profiles, histories, history_years)
        stale = [
            symbol for symbol, frame in histories.items()
            if symbol in rows and frame["date"].max() < as_of - pd.Timedelta(days=MAX_PRICE_AGE_DAYS)
        ]
        for symbol in stale:
            rows.pop(symbol)
        # Position-only symbols are analyzed for exit alerts but not ranked.
        scored = pos.score_results([row for symbol, row in rows.items() if symbol in selected.index])
        screened = (
            scored.assign(**{"Passes All": False, "Failed Rules": ""}) if scored.empty
            else pos.apply_screen(scored, screen)
        )
        candidates = pos.ranked(screened[screened["Passes All"]])
        holdout = (setups.get("global") or {}).get("holdout") or {}
        signals = [
            long_term_signal(
                row, int(rank), float(histories[row["Symbol"]]["close"].iloc[-1]),
                histories[row["Symbol"]]["date"].iloc[-1], holdout,
            )
            for rank, row in candidates.iterrows()
        ]
        store.write_signals(client, "long_term", as_of, signals)
        try:
            alert_stats = alerts.send_long_term_exit_alerts(client, as_of, rows, positions)
        except Exception as exc:
            logger.exception("Long Term alerts failed")
            alert_stats = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        failed_rules = screened.loc[~screened["Passes All"], "Failed Rules"].str.split(", ").explode().value_counts()
        stats = {
            "as_of": as_of.date().isoformat(),
            "screen": screen,
            "published_screen": "global" in setups,
            "universe": len(universe),
            **coverage,
            "stale_prices": len(stale),
            "candidates": len(signals),
            "failed_rules": {str(rule): int(count) for rule, count in failed_rules.items()},
            "positions_checked": int(positions["symbol"].isin(list(rows)).sum()),
            "alerts": alert_stats,
            "duration_seconds": round(time.monotonic() - started, 1),
        }
        store.finish_run(client, run_id, "ok", stats)
        return stats
    except Exception as exc:
        store.finish_run(client, run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"[:500]})
        raise


TOOLS = {"dividend_play": dividend_play, "swing": swing, "long_term": long_term}


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
