"""Weekly job: search each trading tool's setups and publish the ones that pass the gates.

    python -m pipeline.trading_optimize --tool dividend_play

Results go to ``trading_setups`` (one global setup plus per-stock overrides)
with a ``trading_runs`` row recording what was searched.
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

import harvest.dividend_play as dp
import harvest.position as pos
import harvest.swing as sw
import harvest.trading_store as store
from pipeline import trading_data


logger = logging.getLogger(__name__)

DIVIDEND_PLAY_PRESET = "Balanced"


def _number(value) -> Optional[float]:
    value = float(value)
    return value if np.isfinite(value) else None


def _profit_factor(gains: float, losses: float) -> Optional[float]:
    if losses > 0:
        return gains / losses
    return None if gains <= 0 else float("inf")


def _t_stat(total: float, sum_sq: float, trades: int) -> Optional[float]:
    if trades < 2:
        return None
    mean = total / trades
    variance = (sum_sq - trades * mean ** 2) / (trades - 1)
    if variance <= 0:
        return None
    return float(mean / np.sqrt(variance / trades))


def dividend_play_global_holdout(row) -> dict:
    """Pooled holdout record of the recommended setup across the shortlist."""
    return {
        "level": "global",
        "trades": int(row["Validation Trades"]),
        "win_rate": _number(row["Validation Win Rate"]),
        "avg_trade": _number(row["Validation Avg Trade"]),
        "profit_factor": _number(row["Validation Profit Factor"]),
        "t_stat": _number(row["Validation t-stat"]),
        "avg_days": _number(row["Avg Days"]),
        "return_per_month": _number(row["Return / Month"]),
        "stocks_validated": int(row["Stocks Validated"]),
        "positive_stocks_pct": _number(row["Positive Stocks %"]),
        "setup_score": _number(row["Setup Score"]),
    }


def dividend_play_stock_holdout(row) -> dict:
    """Holdout record of one stock's own best setup."""
    trades = int(row["Validation Trades"])
    return {
        "level": "stock",
        "trades": trades,
        "win_rate": _number(row["Validation Win Rate"]),
        "avg_trade": _number(row["Validation Avg Trade"]),
        "profit_factor": _profit_factor(float(row["Validation Gains"]), float(row["Validation Losses"])),
        "t_stat": _t_stat(float(row["Validation Sum"]), float(row["Validation Sum Sq"]), trades),
        "avg_days": _number(row["Avg Days"]),
        "return_per_month": _number(row["Return / Month"]),
        "train_avg_trade": _number(row["Train Avg Trade"]),
        "stock_setup_score": _number(row["Stock Setup Score"]),
    }


def dividend_play(client, as_of=None, preset: str = DIVIDEND_PLAY_PRESET) -> dict:
    """Search Dividend Play setups on the default shortlist and publish those that pass.

    Uses the page's defaults: shortlist filters, history from 2015, the latest
    30% of ex-dates held out, default costs and the same holdout gates. The
    global setup is published only when one qualifies; per-stock setups when a
    stock's own best setup passes the stock gates.
    """
    as_of = pd.Timestamp(as_of or datetime.date.today()).normalize()
    run_id = store.start_run(client, "dividend_play", "optimize", as_of)
    started = time.monotonic()
    try:
        events = trading_data.load_dividend_events(client)
        universe = trading_data.load_universe()
        shortlist, eligible = dp.build_shortlist(universe, events)
        symbols = shortlist["stock"].tolist()
        if not symbols:
            raise RuntimeError("The Dividend Play shortlist is empty")

        start = pd.Timestamp(dp.DEFAULT_START)
        histories = trading_data.load_price_histories(
            symbols, start - pd.Timedelta(days=dp.PRICE_WARMUP_DAYS),
        )
        setups = dp.build_setup_grid(**dp.SEARCH_PRESETS[preset])
        split = dp.holdout_split_date(start, as_of, dp.DEFAULT_VALIDATION_PCT)
        costs = dp.trade_costs()

        results = []
        for symbol in symbols:
            if symbol not in histories:
                continue
            data = dp.play_data(histories[symbol], events.get(symbol), start, as_of)
            results.extend(dp.evaluate_stock_setups(symbol, data, setups, split, costs))
        if not results:
            raise RuntimeError("No Dividend Play backtests were produced")
        results = pd.DataFrame(results)

        setup_df = dp.summarize_setups(results, setups, dp.MIN_PROFIT_FACTOR, dp.MIN_T_STAT)
        qualified, min_stocks, min_trades = dp.qualify_setups(setup_df, results)
        best_stock = dp.best_stock_setups(
            results, setups, dp.MIN_STOCK_VALIDATION_TRADES, dp.MIN_STOCK_VALIDATION_WIN_RATE,
        )

        published = []
        if not qualified.empty:
            top = qualified.iloc[0]
            published.append({
                "scope": "global",
                "params": dp.setup_from_row(top),
                "holdout": dividend_play_global_holdout(top),
            })
        for _, row in best_stock.iterrows():
            published.append({
                "scope": row["Symbol"],
                "params": dp.setup_from_row(row),
                "holdout": dividend_play_stock_holdout(row),
            })
        store.write_setups(client, "dividend_play", published, run_id)

        stats = {
            "preset": preset,
            "start": start.date().isoformat(),
            "holdout_from": split.date().isoformat(),
            "costs": costs,
            "eligible_stocks": int(eligible),
            "shortlist": len(symbols),
            "stocks_with_prices": len(histories),
            "stocks_tested": int(results["Symbol"].nunique()),
            "setups_searched": len(setups),
            "setups_passing_gates": int(setup_df["Passes Gates"].sum()),
            "required_validated_stocks": min_stocks,
            "required_holdout_trades": min_trades,
            "global_setup": None if qualified.empty else dp.format_setup(published[0]["params"]),
            "stock_setups": int(len(best_stock)),
            "duration_seconds": round(time.monotonic() - started, 1),
        }
        store.finish_run(client, run_id, "ok", stats)
        return stats
    except Exception as exc:
        store.finish_run(client, run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"[:500]})
        raise


SWING_PRESET = "Balanced"
SWING_HISTORY_YEARS = 7      # 3 years of training before a 3-year walk-forward, plus warm-up
SWING_HOLDOUT_YEARS = 3
SWING_COSTS = {"buy_fee": 0.15, "sell_fee": 0.15, "sell_tax": 0.1, "slippage_ticks": sw.DEFAULT_SLIPPAGE_TICKS}
INDICATOR_WARMUP_DAYS = 400


def swing_inputs(client, as_of: pd.Timestamp, history_start: pd.Timestamp):
    """Shortlist, dividend-adjusted OHLCV per stock, JKSE regime, breadth and the index close."""
    events = trading_data.load_dividend_events(client)
    universe = trading_data.load_universe()
    histories = trading_data.load_ohlcv_histories(
        universe["stock"].tolist(), history_start - pd.Timedelta(days=INDICATOR_WARMUP_DAYS),
    )
    histories = {symbol: frame[frame.index <= as_of] for symbol, frame in histories.items()}
    closes = pd.concat(
        [frame["close"].rename("close").to_frame().assign(symbol=symbol).reset_index()
         for symbol, frame in histories.items()],
        ignore_index=True,
    ) if histories else pd.DataFrame(columns=["symbol", "date", "close"])
    shortlist = sw.build_shortlist(universe, sw.price_volatility(closes))
    adjusted = {
        symbol: sw.dividend_adjust_ohlc(histories[symbol], events.get(symbol))
        for symbol in shortlist["stock"] if symbol in histories
    }
    index = trading_data.load_jkse_index()
    index = index[index.index <= as_of]
    regime = sw.compute_regime_series(index)
    breadth = sw.build_market_breadth({symbol: frame["close"] for symbol, frame in adjusted.items()})
    return shortlist, adjusted, regime, breadth, index["close"]


def swing(client, as_of=None, preset: str = SWING_PRESET) -> dict:
    """Search swing setups on the default shortlist and publish those that pass walk-forward.

    The global setup is the walk-forward's latest choice, published only when
    the walk-forward record passes the gates. Per-stock overrides must beat
    the global setup on their own holdout with a t-stat corrected for the
    number of combinations tried.
    """
    as_of = pd.Timestamp(as_of or datetime.date.today()).normalize()
    run_id = store.start_run(client, "swing", "optimize", as_of)
    started = time.monotonic()
    try:
        history_start = as_of - pd.DateOffset(years=SWING_HISTORY_YEARS)
        split = as_of - pd.DateOffset(years=SWING_HOLDOUT_YEARS)
        shortlist, histories, regime, breadth, benchmark = swing_inputs(client, as_of, history_start)
        if not histories:
            raise RuntimeError("The swing shortlist has no price history")
        combos = sw.preset_combos(preset)
        momentum = sw.build_momentum_percentiles(
            {symbol: frame["close"] for symbol, frame in histories.items()},
            {int(c[1]) for c in combos if c[0] == sw.STRATEGY_CROSS_MOMENTUM},
        )

        results, trades, statuses = [], [], {}
        for symbol, frame in histories.items():
            rows, status = sw.evaluate_stock_strategies(
                symbol, frame, regime, combos, split,
                SWING_COSTS["buy_fee"], SWING_COSTS["sell_fee"], SWING_COSTS["sell_tax"],
                benchmark, {lookback: ranks[symbol] for lookback, ranks in momentum.items() if symbol in ranks},
                breadth=breadth, slippage_ticks=SWING_COSTS["slippage_ticks"], trades_out=trades,
            )
            results.extend(rows)
            statuses[status] = statuses.get(status, 0) + 1
        if not results:
            raise RuntimeError("No swing backtests were produced")
        results = pd.DataFrame(results)
        trades = pd.DataFrame(trades)

        forward = sw.walk_forward(trades, split, as_of)
        record = forward["record"]
        published, stock_setups = [], {}
        if forward["final_combo"] is not None and sw.passes_walk_forward_gates(record):
            global_setup = sw.combo_setup(combos[forward["final_combo"]])
            published.append({
                "scope": "global",
                "params": global_setup,
                "holdout": {"level": "global", "method": "walk_forward", **record,
                            "folds": forward["folds"]},
            })
            stock_setups = sw.select_stock_strategies(
                results, global_setup, min_t_stat=sw.required_t_stat(len(combos)),
            )
            for symbol, chosen in stock_setups.items():
                published.append({
                    "scope": symbol,
                    "params": {key: chosen[key] for key in sw.SETUP_KEYS},
                    "holdout": {
                        "level": "stock", "method": "holdout",
                        "trades": chosen["validation_trades"],
                        "win_rate": chosen["validation_win_rate"],
                        "avg_trade": chosen["validation_avg_trade"],
                        "t_stat": chosen["validation_t_stat"],
                        "return_uplift": chosen["return_uplift"],
                    },
                })
        store.write_setups(client, "swing", published, run_id)

        stats = {
            "preset": preset,
            "history_start": history_start.date().isoformat(),
            "holdout_from": split.date().isoformat(),
            "shortlist": len(shortlist),
            "stocks_with_prices": len(histories),
            "statuses": statuses,
            "combinations": len(combos),
            "walk_forward": record,
            "walk_forward_choices": [fold["combo"] for fold in forward["folds"]],
            "global_setup": (sw.format_strategy_params(*[published[0]["params"][k] for k in SETUP_PARAM_KEYS])
                             if published else None),
            "stock_setups": len(stock_setups),
            "required_stock_t_stat": round(sw.required_t_stat(len(combos)), 2),
            "duration_seconds": round(time.monotonic() - started, 1),
        }
        store.finish_run(client, run_id, "ok", stats)
        return stats
    except Exception as exc:
        store.finish_run(client, run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"[:500]})
        raise


SETUP_PARAM_KEYS = ("strategy_type", "param_a", "param_b", "param_c", "param_d")
# ── Long Term ───────────────────────────────────────────────────────────── #
# Nothing is searched: the job backtests the published screen and stores its
# yearly record with it, so the page can show how the screen would have done.

# Companies this size today make up the base set; each year the backtest
# re-selects by that year's market cap, so today's size doesn't pick the winners.
LONG_TERM_BASE_MIN_CAP = 0.5
LONG_TERM_BASE_MAX = 400
LONG_TERM_PRICE_START = "2010-01-01"


def long_term_inputs(profiles: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    """Backtest inputs per symbol, plus the symbols whose statements could not be fetched."""
    def fetch(symbol: str) -> dict:
        financials, statements = trading_data.load_long_term_fundamentals(symbol)
        return {"financials": financials, "statements": statements}

    inputs, failed = {}, {}
    symbols = [s for s in profiles.index if s in histories]
    with concurrent.futures.ThreadPoolExecutor(max_workers=pos.SCAN_WORKERS) as executor:
        futures = {executor.submit(fetch, symbol): symbol for symbol in symbols}
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                fundamentals = future.result()
            except Exception as exc:
                failed[symbol] = f"{type(exc).__name__}: {exc}"[:200]
                continue
            inputs[symbol] = {
                "prices": histories[symbol],
                "profile": profiles.loc[symbol, ["companyName", "sector", "industry"]].to_dict(),
                **fundamentals,
            }
    return inputs, failed


def long_term_record(result: dict) -> dict:
    """The backtest as stored with the published screen: summary plus one entry per year."""
    years = [
        {
            "start": row["start"].isoformat(),
            "end": row["end"].isoformat(),
            "complete": bool(row["complete"]),
            "analyzed": int(row["analyzed"]),
            "passed": int(row["passed"]),
            "holdings": row["holdings"],
            "portfolio": row["portfolio"],
            "benchmark": row["benchmark"],
        }
        for _, row in result["years"].iterrows()
    ]
    return {
        "method": "yearly_rebalance",
        "rebalance": "first session in April",
        "costs_pct": pos.BACKTEST_ROUND_TRIP_COST,
        "reporting_lag_days": {"interim": pos.INTERIM_LAG_DAYS, "annual": pos.ANNUAL_LAG_DAYS},
        **result["summary"],
        "per_year": years,
    }


def long_term(client, as_of=None) -> dict:
    """Backtest the published Long Term screen and publish it with its record."""
    started = time.monotonic()
    as_of = pd.Timestamp(as_of or datetime.date.today()).normalize()
    run_id = store.start_run(client, "long_term", "optimize", as_of)
    try:
        screen = dict(pos.PUBLISHED_SCREEN)
        universe = trading_data.load_long_term_universe()
        base = pos.select_universe(universe, LONG_TERM_BASE_MIN_CAP, LONG_TERM_BASE_MAX)
        histories = trading_data.load_price_histories(base.index.tolist(), LONG_TERM_PRICE_START)
        histories = {symbol: frame[frame["date"] <= as_of] for symbol, frame in histories.items()}
        inputs, failed = long_term_inputs(base, histories)
        benchmark = trading_data.load_jkse_index()["close"]
        result = pos.backtest_screen(
            inputs, benchmark[benchmark.index <= as_of], screen, start_year=pos.BACKTEST_START_YEAR,
        )
        record = long_term_record(result)
        store.write_setups(client, "long_term", [{"scope": "global", "params": screen, "holdout": record}], run_id)
        stats = {
            "as_of": as_of.date().isoformat(),
            "base": len(base),
            "with_prices": len(histories),
            "with_statements": len(inputs),
            "failed": len(failed),
            "failed_symbols": dict(sorted(failed.items())[:20]),
            **{key: value for key, value in record.items() if key != "per_year"},
            "duration_seconds": round(time.monotonic() - started, 1),
        }
        store.finish_run(client, run_id, "ok", stats)
        return stats
    except Exception as exc:
        store.finish_run(client, run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"[:500]})
        raise


TOOLS = {"dividend_play": dividend_play, "swing": swing, "long_term": long_term}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search and publish trading setups")
    parser.add_argument("--tool", choices=sorted(TOOLS), required=True)
    parser.add_argument("--as-of", help="Date to optimize as of (YYYY-MM-DD), default today")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        trading_data.validate_environment()
        stats = TOOLS[args.tool](trading_data.get_supabase_client(), as_of=args.as_of)
        logger.info("Optimize summary for %s: %s", args.tool, json.dumps(stats, sort_keys=True, default=str))
        return 0
    except Exception:
        logger.exception("Optimize job failed for %s", args.tool)
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    sys.exit(main())
