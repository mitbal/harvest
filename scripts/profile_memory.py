#!/usr/bin/env python3
"""
Memory Profiling Script — Stock Picker Page
============================================
Profiles each major data-loading and computation step that runs when the
stock picker page loads so that we can identify the biggest RAM consumers.

Usage
-----
    # Basic run (JKSE list, no stock detail)
    python scripts/profile_memory.py

    # Profile a specific stock detail fetch too
    python scripts/profile_memory.py --stock BBCA.JK

    # Profile S&P 500 list
    python scripts/profile_memory.py --market sp500

    # Full profile with tracemalloc top-lines report
    python scripts/profile_memory.py --stock BBCA.JK --tracemalloc-top 20

Requirements
------------
    pip install psutil tracemalloc memory_profiler pandas redis

Environment variables required (same as the app):
    FMP_API_KEY   — Financial Modelling Prep API key
    REDIS_URL     — Redis connection URL
"""

import os
import sys
import json
import time
import argparse
import tracemalloc
import linecache
import gc

import psutil
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROCESS = psutil.Process(os.getpid())


def rss_mb() -> float:
    """Current resident set size (RSS) of this process in MB."""
    return PROCESS.memory_info().rss / 1024 / 1024


def df_mem_mb(df: pd.DataFrame, name: str = "") -> float:
    """Deep memory usage of a DataFrame in MB."""
    mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    label = f" [{name}]" if name else ""
    return mb


def print_sep(title: str = "", width: int = 70):
    if title:
        side = (width - len(title) - 2) // 2
        print(f"\n{'─' * side} {title} {'─' * (width - side - len(title) - 2)}")
    else:
        print("─" * width)


def snapshot_diff(snap1, snap2, top_n: int = 15):
    """Print the top allocations between two tracemalloc snapshots."""
    top_stats = snap2.compare_to(snap1, "lineno")
    print(f"\n  Top {top_n} memory allocations (net new since last checkpoint):")
    for stat in top_stats[:top_n]:
        print(f"    {stat}")


def display_tracemalloc_top(snapshot, top_n: int = 20):
    """Print the top memory consumers in a tracemalloc snapshot."""
    top_stats = snapshot.statistics("lineno")
    print(f"\n  Top {top_n} memory consumers (cumulative):")
    for idx, stat in enumerate(top_stats[:top_n], 1):
        frame = stat.traceback[0]
        # Try to show source line
        line = linecache.getline(frame.filename, frame.lineno).strip()
        src = f"  → {line}" if line else ""
        size_mb = stat.size / 1024 / 1024
        print(f"    {idx:>2}. {stat.size / 1024:>8.1f} KB  {frame.filename}:{frame.lineno}{src}")


# ---------------------------------------------------------------------------
# Checkpointed profiler
# ---------------------------------------------------------------------------

class MemoryProfiler:
    """Tracks RSS and optionally tracemalloc snapshots at named checkpoints."""

    def __init__(self, use_tracemalloc: bool = True):
        self.use_tracemalloc = use_tracemalloc
        self.checkpoints: list[dict] = []
        self._snap_prev = None
        self._t0 = None

        if use_tracemalloc:
            tracemalloc.start(25)  # keep 25 frames of traceback

    def baseline(self, label: str = "Baseline (imports only)"):
        gc.collect()
        rss = rss_mb()
        snap = tracemalloc.take_snapshot() if self.use_tracemalloc else None
        self._snap_prev = snap
        self._t0 = time.perf_counter()
        self.checkpoints.append({"label": label, "rss_mb": rss, "delta_mb": 0.0, "elapsed_s": 0.0})
        print(f"  📍 {label:<50}  RSS={rss:>7.1f} MB")
        return snap

    def mark(self, label: str, show_diff: bool = False, top_n: int = 0):
        """Record a checkpoint."""
        gc.collect()
        rss = rss_mb()
        elapsed = time.perf_counter() - self._t0
        prev_rss = self.checkpoints[-1]["rss_mb"] if self.checkpoints else rss
        delta = rss - prev_rss

        snap = tracemalloc.take_snapshot() if self.use_tracemalloc else None

        emoji = "🔴" if delta > 50 else ("🟡" if delta > 10 else "🟢")
        print(f"  {emoji} {label:<50}  RSS={rss:>7.1f} MB  Δ={delta:>+7.1f} MB  t={elapsed:.2f}s")

        if show_diff and snap and self._snap_prev:
            snapshot_diff(self._snap_prev, snap, top_n=top_n or 10)
        if top_n > 0 and snap:
            display_tracemalloc_top(snap, top_n=top_n)

        self._snap_prev = snap
        self.checkpoints.append({"label": label, "rss_mb": rss, "delta_mb": delta, "elapsed_s": elapsed})
        return snap

    def report(self):
        """Print a summary table sorted by worst offenders."""
        print_sep("SUMMARY — Memory by Step")
        rows = sorted(self.checkpoints[1:], key=lambda r: -r["delta_mb"])  # skip baseline
        print(f"  {'Step':<50}  {'Δ MB':>8}  {'Total RSS':>10}")
        print(f"  {'─' * 50}  {'─' * 8}  {'─' * 10}")
        for r in rows:
            bar = "█" * min(int(abs(r["delta_mb"]) // 5), 20)
            print(f"  {r['label']:<50}  {r['delta_mb']:>+8.1f}  {r['rss_mb']:>8.1f} MB  {bar}")

        if self.use_tracemalloc:
            tracemalloc.stop()


# ---------------------------------------------------------------------------
# Profile stages
# ---------------------------------------------------------------------------

def profile_imports(profiler: MemoryProfiler):
    """Stage 0 – Python interpreter + stdlib already loaded."""
    profiler.baseline("Python interpreter + baseline imports")


def profile_third_party_imports(profiler: MemoryProfiler):
    """Stage 1 – import heavy third-party libs."""
    import numpy as np          # noqa: F401
    import altair as alt        # noqa: F401
    import seaborn as sns       # noqa: F401
    profiler.mark("After numpy / altair / seaborn import")


def profile_harvest_imports(profiler: MemoryProfiler):
    """Stage 2 – import internal harvest modules."""
    import harvest.data as hd   # noqa: F401
    import harvest.plot as hp   # noqa: F401
    profiler.mark("After harvest.data + harvest.plot import")


def profile_redis_connect(profiler: MemoryProfiler):
    """Stage 3 – connect to Redis."""
    import redis
    redis_url = os.environ["REDIS_URL"]
    r = redis.from_url(redis_url)
    r.ping()
    profiler.mark("After Redis connection")
    return r


def profile_div_score_table(profiler: MemoryProfiler, r, market: str = "JKSE"):
    """Stage 4 – download + parse dividend score table from Redis."""
    import harvest.data as hd

    key = "div_score_jkse" if market == "JKSE" else "div_score_sp500"

    t0 = time.perf_counter()
    rjson = r.get(key)
    elapsed_redis = time.perf_counter() - t0

    if rjson is None:
        print(f"    ⚠️  Key '{key}' not found in Redis — skipping Redis load step.")
        return None

    print(f"    Redis GET took {elapsed_redis * 1000:.1f} ms  |  payload size = {len(rjson) / 1024:.1f} KB")

    div_score_json = json.loads(rjson)
    profiler.mark("After json.loads(redis payload)")

    raw_df = pd.DataFrame(json.loads(div_score_json["content"]))
    profiler.mark("After pd.DataFrame from JSON", show_diff=True)

    # Rename + merge company profile (same as the app)
    raw_df.rename(columns={"symbol": "stock"}, inplace=True)
    cp_df = hd.get_company_profile(raw_df["stock"].to_list())
    profiler.mark("After hd.get_company_profile()")

    raw_df.drop(columns=["price"], inplace=True)
    final_df = raw_df.merge(cp_df[["price", "changes"]], left_on="stock", right_on="symbol")
    final_df = final_df.set_index("stock")
    profiler.mark("After merge + set_index")

    # Report sizes
    raw_mb = df_mem_mb(raw_df, "raw_df")
    cp_mb = df_mem_mb(cp_df, "cp_df")
    final_mb = df_mem_mb(final_df, "final_df")
    print(f"\n    DataFrame sizes:")
    print(f"      raw_df    : {raw_mb:>7.2f} MB  ({len(raw_df):,} rows × {len(raw_df.columns)} cols)")
    print(f"      cp_df     : {cp_mb:>7.2f} MB  ({len(cp_df):,} rows × {len(cp_df.columns)} cols)")
    print(f"      final_df  : {final_mb:>7.2f} MB  ({len(final_df):,} rows × {len(final_df.columns)} cols)")

    # Per-column memory breakdown
    print(f"\n    Per-column memory (final_df, top 15 by size):")
    col_mem = (
        final_df.memory_usage(deep=True)
        .drop("Index", errors="ignore")
        .sort_values(ascending=False)
        / 1024
    )
    for col, kb in col_mem.head(15).items():
        print(f"      {col:<35} {kb:>8.1f} KB")

    return final_df


def profile_column_pruning(profiler: MemoryProfiler, final_df: pd.DataFrame):
    """Stage 5 – column pruning (mirrors what the app does)."""
    import numpy as np

    _KEEP_COLS = [
        "price", "changes", "sector", "industry", "mktCap", "ipoDate",
        "yield", "lastDiv", "avgFlatAnnualDivIncrease", "numDividendYear",
        "positiveYear", "numOfYear", "maximumCutPct", "max10CutPct",
        "peRatio", "psRatio", "revenueGrowth", "netIncomeGrowth",
        "medianProfitMargin", "earningTTM", "revenueTTM",
        "revenueGrowthTTM", "netIncomeGrowthTTM",
        "return_7d", "return_1m", "return_1y", "return_10y",
        "total_return_1y", "total_return_10y",
        "is_syariah",
    ]
    keep = [c for c in _KEEP_COLS if c in final_df.columns]
    pruned_df = final_df[keep]

    before_mb = df_mem_mb(final_df, "final_df")
    after_mb = df_mem_mb(pruned_df, "pruned_df")
    savings = before_mb - after_mb

    profiler.mark(f"After column pruning ({len(final_df.columns)} → {len(keep)} cols)")
    print(f"\n    Memory before pruning : {before_mb:.2f} MB")
    print(f"    Memory after  pruning : {after_mb:.2f} MB")
    print(f"    Savings               : {savings:.2f} MB  ({savings / before_mb * 100:.1f}%)")
    print(f"    Dropped columns       : {sorted(set(final_df.columns) - set(keep))}")

    return pruned_df


def profile_derived_columns(profiler: MemoryProfiler, pruned_df: pd.DataFrame):
    """Stage 6 – compute derived columns (get_processed_df equivalent)."""
    import numpy as np

    df = pruned_df.copy()

    df["marginTTM"] = df["earningTTM"] / df["revenueTTM"] * 100
    df["mc_penalty"] = df["mktCap"].apply(lambda x: 1 / (1 + np.exp(-2 * (x / 3_000_000_000_000 - 1))))
    df["maximumCutPct"] = df["maximumCutPct"].apply(lambda x: min(x, 0) * -1)
    df["max10CutPct"]   = df["max10CutPct"].apply(lambda x: min(x, 0) * -1)
    df["maxDivIncrease"]       = df.apply(lambda x: min(x["avgFlatAnnualDivIncrease"], x["lastDiv"] * 0.05), axis=1)
    df["maxRevGrowthDecrease"] = df.apply(lambda x: min(x["revenueGrowthTTM"], 0), axis=1)
    df["maxIncGrowthDecrease"] = df.apply(lambda x: min(x["netIncomeGrowthTTM"], 0), axis=1)

    return_cols = ["return_7d", "return_1m", "return_1y", "return_10y", "total_return_1y", "total_return_10y"]
    for col in return_cols:
        if col in df.columns:
            df[col] = df[col] * 100

    df["DScore"] = (
        (df["lastDiv"] + df["maxDivIncrease"] * 5 * (df["positiveYear"] / df["numOfYear"])) / df["price"]
    ) * 100 \
      * (df["numDividendYear"] / (df["numDividendYear"] + 25)) \
      * (1 - np.exp(-df["numDividendYear"] / 5)) \
      * (100 - df["max10CutPct"]) / 100 \
      * df["mc_penalty"] \
      * (1 + df["maxRevGrowthDecrease"] / 100) \
      * (1 + df["maxIncGrowthDecrease"] / 100)

    df = df.fillna(0).sort_values("DScore", ascending=False)

    profiler.mark("After get_processed_df (derived columns + sort)")
    print(f"\n    filtered_df size: {df_mem_mb(df, 'filtered_df'):.2f} MB  "
          f"({len(df):,} rows × {len(df.columns)} cols)")

    return df


def profile_display_df(profiler: MemoryProfiler, filtered_df: pd.DataFrame, sl: str = "JKSE"):
    """Stage 7 – build the display copy used in st.dataframe."""
    divisor = 1_000_000_000_000 if sl == "JKSE" else 1_000_000_000

    display_df = filtered_df.copy()
    mcap_pos    = display_df.columns.get_loc("mktCap")
    earning_pos = display_df.columns.get_loc("earningTTM")
    revenue_pos = display_df.columns.get_loc("revenueTTM")
    display_df.insert(mcap_pos,         "mktCapDisplay",    display_df["mktCap"]     / divisor)
    display_df.insert(earning_pos + 1,  "earningTTMDisplay", display_df["earningTTM"] / divisor)
    display_df.insert(revenue_pos + 2,  "revenueTTMDisplay", display_df["revenueTTM"] / divisor)

    profiler.mark("After display_df copy + insert (table view)")
    print(f"\n    display_df size: {df_mem_mb(display_df, 'display_df'):.2f} MB")

    return display_df


def profile_treemap_df(profiler: MemoryProfiler, filtered_df: pd.DataFrame):
    """Stage 8 – build the df_tree used in the Treemap view."""
    df_tree = pd.DataFrame({
        "sector":              filtered_df["sector"],
        "industry":            filtered_df["industry"],
        "Market Cap":          filtered_df["mktCap"] / 1_000_000_000,
        "Revenue":             filtered_df["revenueTTM"],
        "Net Income":          filtered_df["earningTTM"],
        "Dividend Yield":      filtered_df["yield"],
        "Median Profit Margin": filtered_df["medianProfitMargin"],
        "TTM Profit Margin":   filtered_df["marginTTM"],
        "Revenue Growth":      filtered_df["revenueGrowth"],
        "1D Price Return":     filtered_df["changes"] / filtered_df["price"] * 100,
        "7D Price Return":     filtered_df["return_7d"],
        "1M Price Return":     filtered_df["return_1m"],
        "1Y Price Return":     filtered_df["return_1y"],
        "10Y Price Return":    filtered_df["return_10y"],
        "Total 1Y Return":     filtered_df["total_return_1y"],
        "Total 10Y Return":    filtered_df["total_return_10y"],
        "PE Ratio":            filtered_df["peRatio"],
        "PS Ratio":            filtered_df["psRatio"],
    }, index=filtered_df.index).dropna()

    profiler.mark("After df_tree construction (treemap view)")
    print(f"\n    df_tree size: {df_mem_mb(df_tree, 'df_tree'):.2f} MB  "
          f"({len(df_tree):,} rows × {len(df_tree.columns)} cols)")

    return df_tree


def profile_stock_detail(profiler: MemoryProfiler, stock_name: str, sl: str = "JKSE"):
    """Stage 9 – fetch all single-stock detail data."""
    import harvest.data as hd

    print(f"\n  Fetching detail for {stock_name} ...")

    t0 = time.perf_counter()
    n_share = hd.get_shares_outstanding(stock_name)["outstandingShares"].tolist()[0]
    profiler.mark(f"  After get_shares_outstanding({stock_name})")

    fin = hd.get_financial_data(stock_name)
    profiler.mark(f"  After get_financial_data({stock_name})")

    cp_df = hd.get_company_profile([stock_name])
    profiler.mark(f"  After get_company_profile({stock_name})")

    price_df = hd.get_daily_stock_price(stock_name, start_from="2010-01-01")
    profiler.mark(f"  After get_daily_stock_price({stock_name})")

    if sl == "JKSE":
        sdf = hd.get_dividend_history_single_stock(stock_name, source="dag")
    else:
        sdf = hd.get_dividend_history_single_stock(stock_name, source="fmp")
    profiler.mark(f"  After get_dividend_history({stock_name})")

    elapsed = time.perf_counter() - t0
    print(f"\n    Total stock-detail fetch time: {elapsed:.2f}s")
    print(f"    fin DataFrame       : {df_mem_mb(fin):.2f} MB  "
          f"({len(fin):,} rows × {len(fin.columns)} cols)")
    print(f"    price_df DataFrame  : {df_mem_mb(price_df):.2f} MB  "
          f"({len(price_df):,} rows × {len(price_df.columns)} cols)")
    if sdf is not None:
        print(f"    sdf DataFrame       : {df_mem_mb(sdf):.2f} MB  "
              f"({len(sdf):,} rows × {len(sdf.columns)} cols)")

    # Per-column breakdown for fin (usually the biggest)
    print(f"\n    Per-column memory (fin, all cols):")
    col_mem = (
        fin.memory_usage(deep=True)
        .drop("Index", errors="ignore")
        .sort_values(ascending=False)
        / 1024
    )
    for col, kb in col_mem.items():
        print(f"      {col:<40} {kb:>8.1f} KB")

    return fin, cp_df, price_df, sdf, n_share


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Memory profiler for stock picker page")
    parser.add_argument("--market", choices=["jkse", "sp500"], default="jkse",
                        help="Which stock list to load (default: jkse)")
    parser.add_argument("--stock", default=None,
                        help="Optional stock ticker to profile single-stock fetch (e.g. BBCA.JK)")
    parser.add_argument("--tracemalloc-top", type=int, default=0,
                        help="Show top-N tracemalloc allocations in the final report (0 = skip)")
    parser.add_argument("--show-diff", action="store_true",
                        help="Show per-step allocation diffs from tracemalloc")
    args = parser.parse_args()

    market = args.market.upper().replace("500", "500")
    sl = "JKSE" if args.market == "jkse" else "S&P500"
    use_tracemalloc = args.tracemalloc_top > 0 or args.show_diff

    # Validate env vars early
    for var in ("FMP_API_KEY", "REDIS_URL"):
        if not os.environ.get(var):
            print(f"❌  Environment variable '{var}' is not set. Aborting.")
            sys.exit(1)

    print_sep("HARVEST — STOCK PICKER MEMORY PROFILER")
    print(f"  Market: {sl}  |  Stock: {args.stock or 'none'}  |  tracemalloc: {use_tracemalloc}")
    print_sep()

    profiler = MemoryProfiler(use_tracemalloc=use_tracemalloc)

    # ── Stage 0: baseline ───────────────────────────────────────────────────
    print_sep("Stage 0 · Baseline")
    profile_imports(profiler)

    # ── Stage 1: third-party imports ────────────────────────────────────────
    print_sep("Stage 1 · Third-party imports")
    profile_third_party_imports(profiler)

    # ── Stage 2: harvest imports ─────────────────────────────────────────────
    print_sep("Stage 2 · Internal harvest imports")
    profile_harvest_imports(profiler)

    # ── Stage 3: Redis connection ────────────────────────────────────────────
    print_sep("Stage 3 · Redis connection")
    r = profile_redis_connect(profiler)

    # ── Stage 4: Load dividend score table ──────────────────────────────────
    print_sep("Stage 4 · Dividend score table (Redis → DataFrame)")
    final_df = profile_div_score_table(profiler, r, market=sl)
    if final_df is None:
        print("  ⚠️  Cannot continue without dividend score table.")
        profiler.report()
        return

    # ── Stage 5: Column pruning ──────────────────────────────────────────────
    print_sep("Stage 5 · Column pruning")
    pruned_df = profile_column_pruning(profiler, final_df)

    # ── Stage 6: Derived columns / DScore ───────────────────────────────────
    print_sep("Stage 6 · Derived columns + DScore")
    filtered_df = profile_derived_columns(profiler, pruned_df)

    # ── Stage 7: Display DataFrame (table view) ──────────────────────────────
    print_sep("Stage 7 · Display DataFrame (table view)")
    _display_df = profile_display_df(profiler, filtered_df, sl=sl)
    del _display_df

    # ── Stage 8: Treemap DataFrame ───────────────────────────────────────────
    print_sep("Stage 8 · Treemap DataFrame")
    _tree_df = profile_treemap_df(profiler, filtered_df)
    del _tree_df

    # ── Stage 9 (optional): Single stock detail ──────────────────────────────
    if args.stock:
        stock_name = args.stock.upper()
        if sl == "JKSE" and ".JK" not in stock_name:
            stock_name += ".JK"
        print_sep(f"Stage 9 · Single stock detail: {stock_name}")
        profile_stock_detail(profiler, stock_name, sl=sl)

    # ── Final tracemalloc report ─────────────────────────────────────────────
    if args.tracemalloc_top > 0:
        print_sep("tracemalloc Top Allocations (full run)")
        snap = tracemalloc.take_snapshot()
        display_tracemalloc_top(snap, top_n=args.tracemalloc_top)

    # ── Summary table ────────────────────────────────────────────────────────
    profiler.report()
    print_sep()
    print(f"\n  Final process RSS: {rss_mb():.1f} MB\n")


if __name__ == "__main__":
    main()
