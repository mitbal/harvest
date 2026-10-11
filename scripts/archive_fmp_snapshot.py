"""Back up one ``data/global`` FMP snapshot to R2 as Parquet.

Converts the files ``refresh_global_data.py`` writes into three long tables and
uploads them under ``fmp-archive/<snapshot>/`` in the R2 bucket:

* ``universe.parquet`` — the stock list, as downloaded
* ``profiles.parquet`` — every exchange's profiles, plus ``source_exchange``
  (the exchange name the file was fetched for)
* ``dividends.parquet`` — one row per symbol and dividend, plus ``source_set``

Snapshots are never edited: an existing object is only replaced with ``--force``.
Every upload is read back and its row count checked.

Usage:
    python -m scripts.archive_fmp_snapshot --date 2026-10-10
    python -m scripts.archive_fmp_snapshot --date 2025-06-14 --skip dividends
    # The 2025 dividend files carry no date; they were downloaded on 2025-07-14.
    python -m scripts.archive_fmp_snapshot --date 2025-07-14 --undated-dividends --skip universe profiles
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harvest.prices as prices  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "global"
PREFIX = "fmp-archive"
TABLES = ("universe", "profiles", "dividends")


def _strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Mixed-type text columns (cik, zip, phone, …) as strings so Arrow accepts them."""
    for column in frame.columns:
        if frame[column].dtype == object or str(frame[column].dtype) in ("str", "string"):
            frame[column] = frame[column].astype("string")
    return frame


def build_universe(date: str) -> pd.DataFrame:
    return _strings(pd.read_csv(DATA_DIR / f"{date}_all_stocks_universe.csv", low_memory=False))


def build_profiles(date: str) -> pd.DataFrame:
    frames = []
    for path in sorted(DATA_DIR.glob(f"{date}_*_company_profiles.csv")):
        exchange = path.name[len(date) + 1:-len("_company_profiles.csv")]
        frame = pd.read_csv(path, dtype=str)
        frame.insert(0, "source_exchange", exchange)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No profile files for {date}")
    profiles = pd.concat(frames, ignore_index=True)
    # Read as text so no file's types clash with another's; restore the numeric fields.
    for column in ("price", "beta", "volAvg", "mktCap", "lastDiv", "changes", "dcfDiff", "dcf", "fullTimeEmployees"):
        if column in profiles.columns:
            profiles[column] = pd.to_numeric(profiles[column], errors="coerce")
    for column in ("defaultImage", "isEtf", "isActivelyTrading", "isAdr", "isFund"):
        if column in profiles.columns:
            profiles[column] = profiles[column].map({"True": True, "False": False}).astype("boolean")
    return _strings(profiles)


def build_dividends(date: str, undated: bool) -> pd.DataFrame:
    if undated:
        paths = [p for p in DATA_DIR.glob("*_dividends.pkl") if not re.match(r"\d{4}-\d{2}-\d{2}_", p.name)]
    else:
        paths = list(DATA_DIR.glob(f"{date}_*_dividends.pkl"))
    frames = []
    for path in sorted(paths):
        name = path.name.removesuffix("_dividends.pkl")
        name = name[len(date) + 1:] if not undated else name
        for symbol, frame in pd.read_pickle(path).items():
            frames.append(frame.assign(symbol=symbol, source_set=name))
    if not frames:
        raise FileNotFoundError(f"No dividend files for {date}")
    dividends = pd.concat(frames, ignore_index=True)
    leading = ["source_set", "symbol"]
    dividends = dividends[leading + [c for c in dividends.columns if c not in leading]]
    for column in ("adjDividend", "dividend"):
        dividends[column] = pd.to_numeric(dividends[column], errors="coerce")
    return _strings(dividends)


def upload(frame: pd.DataFrame, key: str, config: dict, force: bool) -> None:
    filesystem = prices._filesystem(config)
    target = f"{config['bucket']}/{key}"
    if filesystem.get_file_info(target).type.name != "NotFound" and not force:
        print(f"  {key}: exists, skipped (use --force to replace)")
        return
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), target, filesystem=filesystem,
                   compression="zstd")
    stored = pq.read_metadata(target, filesystem=filesystem)
    if stored.num_rows != len(frame):
        raise RuntimeError(f"{key}: wrote {len(frame)} rows but R2 has {stored.num_rows}")
    size = filesystem.get_file_info(target).size / 1e6
    print(f"  {key}: {stored.num_rows:,} rows, {size:.1f} MB — verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", required=True, help="snapshot date: the file-name prefix and the archive folder")
    parser.add_argument("--skip", nargs="*", default=[], choices=TABLES)
    parser.add_argument("--undated-dividends", action="store_true",
                        help="read the undated <name>_dividends.pkl files (the 2025 download)")
    parser.add_argument("--force", action="store_true", help="replace objects already in the archive")
    args = parser.parse_args()

    config = prices.r2_config()
    if config is None:
        sys.exit("R2 is not configured; set the R2_* settings or .streamlit/secrets.toml [r2]")

    builders = {
        "universe": lambda: build_universe(args.date),
        "profiles": lambda: build_profiles(args.date),
        "dividends": lambda: build_dividends(args.date, args.undated_dividends),
    }
    print(f"Archiving {args.date} to {config['bucket']}/{PREFIX}/{args.date}/")
    for table, build in builders.items():
        if table not in args.skip:
            upload(build(), f"{PREFIX}/{args.date}/{table}.parquet", config, args.force)


if __name__ == "__main__":
    main()
