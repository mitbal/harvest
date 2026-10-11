"""Compare the stored daily prices (FMP-sourced, on R2) with Yahoo Finance.

Checks whether Yahoo could replace FMP as the daily price source: for every
symbol in the stored history over the last ``--days`` days, it downloads the
same window from Yahoo and reports which symbols Yahoo lacks, which dates
either side misses, and how far the OHLCV values drift apart.

The symbol list comes from the stored history itself, so no FMP key is needed.

Usage:
    python -m scripts.compare_yahoo_prices                 # jkse, last 30 days, R2
    python -m scripts.compare_yahoo_prices --source local  # data/jkse/historical_prices.parquet
    python -m scripts.compare_yahoo_prices --symbols BBCA.JK TLKM.JK --days 90

Writes ``rows.csv`` (one row per symbol and date) and ``symbols.csv`` (one row
per symbol) to ``--out-dir``.
"""
import argparse
import datetime
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harvest.prices as prices  # noqa: E402

PRICE_FIELDS = ("open", "high", "low", "close")
YAHOO_COLUMNS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}


def load_stored(exchange: str, source: str, symbols, start) -> pd.DataFrame:
    path = prices.local_path(exchange) if source == "local" else None
    return prices.read_prices(exchange, symbols=symbols, start=start, columns=prices.OHLCV_COLUMNS, path=path)


def download_yahoo(symbols: list[str], start, end, batch_size: int) -> pd.DataFrame:
    """Unadjusted daily OHLCV from Yahoo as a long ``symbol, date, ...`` frame."""
    import yfinance as yf

    frames = []
    for offset in range(0, len(symbols), batch_size):
        batch = symbols[offset:offset + batch_size]
        print(f"  Yahoo batch {offset // batch_size + 1}: {len(batch)} symbols", flush=True)
        raw = yf.download(
            batch,
            start=pd.Timestamp(start).strftime("%Y-%m-%d"),
            # Yahoo's end is exclusive.
            end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if raw is None or raw.empty:
            continue
        for symbol in batch:
            if symbol not in raw.columns.get_level_values(0):
                continue
            frame = raw[symbol].rename(columns=YAHOO_COLUMNS)[list(YAHOO_COLUMNS.values())]
            frame = frame.dropna(subset=["close"])
            if frame.empty:
                continue
            frame = frame.reset_index().rename(columns={frame.index.name or "Date": "date"})
            frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
            frame.insert(0, "symbol", symbol)
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", *YAHOO_COLUMNS.values()])
    return pd.concat(frames, ignore_index=True)


def compare(stored: pd.DataFrame, yahoo: pd.DataFrame) -> pd.DataFrame:
    """Outer join on symbol and date with percent differences (Yahoo vs stored)."""
    stored = stored.assign(date=pd.to_datetime(stored["date"]).dt.normalize())
    rows = stored.merge(yahoo, on=["symbol", "date"], how="outer", suffixes=("_stored", "_yahoo"), indicator=True)
    rows["presence"] = rows.pop("_merge").map({"both": "both", "left_only": "stored_only", "right_only": "yahoo_only"})
    for field in PRICE_FIELDS:
        base = rows[f"{field}_stored"].replace(0, np.nan)
        rows[f"{field}_diff_pct"] = (rows[f"{field}_yahoo"] - base) / base * 100
    rows["volume_ratio"] = rows["volume_yahoo"] / rows["volume_stored"].replace(0, np.nan)
    return rows.sort_values(["symbol", "date"], ignore_index=True)


def summarize(rows: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    def per_symbol(group: pd.DataFrame) -> pd.Series:
        both = group[group["presence"] == "both"]
        close_diff = both["close_diff_pct"].abs()
        ohlc_diff = both[[f"{f}_diff_pct" for f in PRICE_FIELDS]].abs().max(axis=1)
        return pd.Series({
            "stored_days": int(group["presence"].isin(["both", "stored_only"]).sum()),
            "yahoo_days": int(group["presence"].isin(["both", "yahoo_only"]).sum()),
            "matched_days": len(both),
            "stored_only_days": int((group["presence"] == "stored_only").sum()),
            "yahoo_only_days": int((group["presence"] == "yahoo_only").sum()),
            "close_max_diff_pct": close_diff.max(),
            "close_median_diff_pct": close_diff.median(),
            "close_days_over_tol": int((close_diff > tolerance).sum()),
            "ohlc_days_over_tol": int((ohlc_diff > tolerance).sum()),
            "volume_median_ratio": both["volume_ratio"].median(),
        })

    summary = rows.groupby("symbol").apply(per_symbol, include_groups=False).reset_index()
    summary["status"] = np.select(
        [
            summary["yahoo_days"] == 0,
            summary["close_days_over_tol"] > 0,
            summary["stored_only_days"] + summary["yahoo_only_days"] > 0,
        ],
        ["missing_on_yahoo", "close_mismatch", "date_gaps"],
        default="ok",
    )
    return summary


def report(rows: pd.DataFrame, summary: pd.DataFrame, tolerance: float) -> None:
    both = rows[rows["presence"] == "both"]
    print("\n=== Symbols ===")
    print(summary["status"].value_counts().to_string())
    print(f"total: {len(summary)}")

    print("\n=== Rows ===")
    print(rows["presence"].value_counts().to_string())

    print(f"\n=== |Yahoo - stored| % on matched rows (tolerance {tolerance}%) ===")
    quantiles = [0.5, 0.9, 0.99, 1.0]
    table = pd.DataFrame({
        field: both[f"{field}_diff_pct"].abs().quantile(quantiles) for field in PRICE_FIELDS
    })
    table.index = [f"p{int(q * 100)}" for q in quantiles]
    table.loc["share_over_tol_%"] = [
        (both[f"{field}_diff_pct"].abs() > tolerance).mean() * 100 for field in PRICE_FIELDS
    ]
    print(table.round(3).to_string())
    ratio = both["volume_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    if not ratio.empty:
        print(f"\nvolume Yahoo/stored ratio: median {ratio.median():.3f}, "
              f"p10 {ratio.quantile(0.1):.3f}, p90 {ratio.quantile(0.9):.3f}")

    print("\n=== Rows per date (stored-only vs Yahoo-only) ===")
    per_date = rows.pivot_table(index="date", columns="presence", values="symbol", aggfunc="count", fill_value=0)
    print(per_date.tail(40).to_string())

    missing = summary.loc[summary["status"] == "missing_on_yahoo", "symbol"].tolist()
    if missing:
        print(f"\n=== Missing on Yahoo ({len(missing)}) ===")
        print(", ".join(missing))

    worst = summary[summary["matched_days"] > 0].nlargest(20, "close_max_diff_pct")
    print("\n=== 20 worst close mismatches ===")
    print(worst[["symbol", "matched_days", "close_max_diff_pct", "close_median_diff_pct",
                 "close_days_over_tol", "volume_median_ratio"]].round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exchange", default="jkse", choices=["jkse", "sp500"])
    parser.add_argument("--days", type=int, default=30, help="calendar days back from the latest stored date")
    parser.add_argument("--source", default="r2", choices=["r2", "local"], help="where the stored prices are read from")
    parser.add_argument("--symbols", nargs="*", help="limit to these symbols (default: every stored symbol)")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=0.5, help="percent difference counted as a mismatch")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/yahoo_comparison"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    # Generous margin so a stale history still covers --days before its latest date.
    window_start = datetime.date.today() - datetime.timedelta(days=args.days + 60)
    print(f"Reading stored {args.exchange} prices from {args.source}…", flush=True)
    stored = load_stored(args.exchange, args.source, args.symbols, window_start)
    if stored.empty:
        sys.exit("No stored prices in the window")
    end = stored["date"].max()
    start = end - pd.Timedelta(days=args.days)
    stored = stored[stored["date"] >= start]
    symbols = sorted(stored["symbol"].unique())
    print(f"{len(symbols)} symbols, {start.date()} to {end.date()}", flush=True)

    print("Downloading from Yahoo…", flush=True)
    yahoo = download_yahoo(symbols, start, end, args.batch_size)
    yahoo = yahoo[(yahoo["date"] >= start) & (yahoo["date"] <= end)]

    rows = compare(stored, yahoo)
    summary = summarize(rows, args.tolerance)
    report(rows, summary, args.tolerance)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out_dir / "rows.csv", index=False)
    summary.to_csv(args.out_dir / "symbols.csv", index=False)
    print(f"\nWrote {args.out_dir / 'rows.csv'} and {args.out_dir / 'symbols.csv'}")


if __name__ == "__main__":
    main()
