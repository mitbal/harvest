"""Nightly job: replay every published card against later prices for the public track record.

    python -m pipeline.trading_track --tool dividend_play

Runs after the tool's scan. Reads the stored signals, OHLC prices from R2, dividends and the
JKSE index, and writes ``trading_outcomes`` (see ``harvest/track_record.py`` for the rules).
Settled outcomes are never rewritten; open ones are refreshed every night.
"""
import argparse
import json
import logging
import os
import sys
import time

import pandas as pd

import harvest.track_record as tr
import harvest.trading_store as store
from pipeline import trading_data


logger = logging.getLogger(__name__)

# Prices start this long before the first card, so the first session after it is loaded.
PRICE_LEAD_DAYS = 10
SIGNAL_STATUS = {"dividend_play": "actionable", "swing": "actionable", "long_term": "candidate"}
# Long Term cards carry a large detail blob; the replay needs only these.
LONG_TERM_COLUMNS = "symbol,as_of,entry_price,target_price"


def load_benchmark() -> pd.Series | None:
    """JKSE closes by date, or None when FMP is unavailable (outcomes are stored without it)."""
    try:
        return trading_data.load_jkse_index()["close"]
    except Exception:
        logger.exception("JKSE index unavailable; outcomes are stored without the benchmark")
        return None


def replay(client, tool: str, signals: pd.DataFrame, benchmark: pd.Series | None) -> list[dict]:
    """The tool's outcomes from its stored cards."""
    start = pd.Timestamp(signals["as_of"].min()) - pd.Timedelta(days=PRICE_LEAD_DAYS)
    bars = trading_data.load_ohlcv_histories(sorted(signals["symbol"].unique()), start)
    if tool == "dividend_play":
        return tr.dividend_play_outcomes(
            signals, store.scan_dates(client, tool), bars, trading_data.load_dividend_events(client), benchmark,
        )
    if tool == "swing":
        return tr.swing_outcomes(signals, bars, benchmark)
    return tr.long_term_outcomes(signals, bars, benchmark)


def track(client, tool: str, benchmark: pd.Series | None = None) -> dict:
    """Recompute and store one tool's outcomes; returns run stats."""
    started = time.monotonic()
    columns = LONG_TERM_COLUMNS if tool == "long_term" else "*"
    signals = store.read_signal_history(client, tool, SIGNAL_STATUS[tool], columns)
    as_of = signals["as_of"].max() if not signals.empty else pd.Timestamp.today().date()
    run_id = store.start_run(client, tool, "track", as_of)
    try:
        outcomes = replay(client, tool, signals, benchmark) if not signals.empty else []
        written = store.write_outcomes(client, tool, outcomes)
        frame = pd.DataFrame(outcomes, columns=tr.OUTCOME_COLUMNS)
        stats = {
            "as_of": str(as_of),
            "cards": len(signals),
            "first_card": str(signals["as_of"].min()) if not signals.empty else None,
            **{status: int((frame["status"] == status).sum()) for status in tr.STATUSES},
            "benchmark": benchmark is not None,
            **written,
            "duration_seconds": round(time.monotonic() - started, 1),
        }
        store.finish_run(client, run_id, "ok", stats)
        return stats
    except Exception as exc:
        store.finish_run(client, run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"[:500]})
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update the trading track record")
    parser.add_argument("--tool", choices=sorted(SIGNAL_STATUS) + ["all"], required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    tools = sorted(SIGNAL_STATUS) if args.tool == "all" else [args.tool]
    try:
        trading_data.validate_environment()
        client = trading_data.get_supabase_client()
        benchmark = load_benchmark()
    except Exception:
        logger.exception("Track record job could not start")
        return 1
    failed = 0
    for tool in tools:
        try:
            stats = track(client, tool, benchmark)
            logger.info("Track record for %s: %s", tool, json.dumps(stats, sort_keys=True, default=str))
        except Exception:
            logger.exception("Track record job failed for %s", tool)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    sys.exit(main())
