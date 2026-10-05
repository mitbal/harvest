"""Daily price history, one Parquet file per exchange on Cloudflare R2.

The file lives at ``prices/<exchange>.parquet`` in the R2 bucket, with
``data/<exchange>/historical_prices.parquet`` as the local fallback for
development. Rows are ``symbol, date, close`` sorted by symbol and date, so a
filtered read only touches the row groups that hold the requested symbols.

This module stays Streamlit-free so the pipeline can use it; pages wrap
``load_prices`` in ``st.cache_data`` themselves.
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRICE_COLUMNS = ("close",)
INTEGER_COLUMNS = ("volume",)
# About 40 IDX symbols of full history per row group: small enough that a
# one-symbol read stays cheap, large enough to keep the file footer small.
ROW_GROUP_SIZE = 150_000
R2_FIELDS = ("account_id", "access_key_id", "secret_access_key", "bucket")


def local_path(exchange: str) -> Path:
    """Local fallback file for one exchange."""
    return REPOSITORY_ROOT / "data" / exchange / "historical_prices.parquet"


def r2_key(exchange: str) -> str:
    """Object key of one exchange's price file inside the R2 bucket."""
    return f"prices/{exchange}.parquet"


def r2_config() -> Optional[dict]:
    """R2 settings from ``st.secrets["r2"]`` (app) or ``R2_*`` env vars (pipeline); None when incomplete."""
    config = {}
    try:
        import streamlit as st
        config = dict(st.secrets["r2"])
    except Exception:
        pass
    if not config:
        config = {field: os.environ.get(f"R2_{field.upper()}") for field in R2_FIELDS}
    return config if all(config.get(field) for field in R2_FIELDS) else None


def _filesystem(config: Mapping):
    from pyarrow import fs

    return fs.S3FileSystem(
        access_key=config["access_key_id"],
        secret_key=config["secret_access_key"],
        endpoint_override=f"{config['account_id']}.r2.cloudflarestorage.com",
        scheme="https",
        region="auto",
    )


def _empty(columns: Iterable[str]) -> pd.DataFrame:
    frame = pd.DataFrame(columns=["symbol", "date", *columns])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _filters(symbols, start, end) -> list:
    filters = []
    if symbols is not None:
        filters.append(("symbol", "in", list(symbols)))
    if start is not None:
        filters.append(("date", ">=", pd.Timestamp(start).date()))
    if end is not None:
        filters.append(("date", "<=", pd.Timestamp(end).date()))
    return filters


def _to_frame(table: pa.Table) -> pd.DataFrame:
    frame = table.to_pandas()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    for column in frame.columns.difference(["symbol", "date"]):
        # Stored as float32 to halve the file; callers get float64 as before.
        frame[column] = frame[column].astype("int64" if column in INTEGER_COLUMNS else "float64")
    return frame.sort_values(["symbol", "date"], ignore_index=True)


def read_prices(
    exchange: str,
    symbols: Optional[Iterable[str]] = None,
    start=None,
    end=None,
    columns: Optional[Iterable[str]] = PRICE_COLUMNS,
    config: Optional[Mapping] = None,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Read prices from ``path`` if given, else from R2. Errors propagate.

    ``columns=None`` reads every stored column. Used by the pipeline, which
    must never mistake a failed read for an empty history.
    """
    if symbols is not None:
        symbols = list(dict.fromkeys(str(symbol) for symbol in symbols))
        if not symbols:
            return _empty(columns or PRICE_COLUMNS)
    read_columns = None if columns is None else ["symbol", "date", *columns]
    filters = _filters(symbols, start, end) or None

    if path is not None:
        table = pq.read_table(path, columns=read_columns, filters=filters)
    else:
        config = config or r2_config()
        if config is None:
            raise RuntimeError("R2 is not configured; set the R2_* settings")
        table = pq.read_table(
            f"{config['bucket']}/{r2_key(exchange)}",
            filesystem=_filesystem(config),
            columns=read_columns,
            filters=filters,
        )
    return _to_frame(table)


def load_prices(
    exchange: str,
    symbols: Optional[Iterable[str]] = None,
    start=None,
    end=None,
    columns: Iterable[str] = PRICE_COLUMNS,
) -> pd.DataFrame:
    """Long frame ``symbol, date, <columns>`` from R2, falling back to the local file.

    ``start`` and ``end`` are inclusive and accept anything ``pd.Timestamp``
    parses. Returns an empty frame when neither source is available.
    """
    columns = tuple(columns)
    config = r2_config()
    if config is not None:
        try:
            return read_prices(exchange, symbols, start, end, columns, config=config)
        except Exception as exc:
            logger.warning("R2 price read failed for %s, using local file: %s", exchange, exc)

    path = local_path(exchange)
    if not path.exists():
        logger.warning("No price history available for %s", exchange)
        return _empty(columns)
    return read_prices(exchange, symbols, start, end, columns, path=path)


def normalize_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Storage form: typed columns, one row per symbol and date (last wins), sorted."""
    value_columns = [column for column in frame.columns if column not in ("symbol", "date")]
    frame = frame[["symbol", "date", *value_columns]].copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in value_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["symbol", "date", "close"])
    for column in value_columns:
        if column in INTEGER_COLUMNS:
            frame[column] = frame[column].fillna(0).astype("int64")
        else:
            frame[column] = frame[column].astype("float32")
    frame = frame.drop_duplicates(subset=["symbol", "date"], keep="last")
    return frame.sort_values(["symbol", "date"], ignore_index=True)


def _to_table(frame: pd.DataFrame) -> pa.Table:
    frame = normalize_prices(frame)
    frame["date"] = frame["date"].dt.date
    table = pa.Table.from_pandas(frame, preserve_index=False)
    return table.cast(table.schema.set(
        table.schema.get_field_index("date"), pa.field("date", pa.date32())
    ))


def write_prices(
    exchange: str,
    frame: pd.DataFrame,
    config: Optional[Mapping] = None,
    path: Optional[Path] = None,
) -> str:
    """Replace the whole price file for ``exchange``; returns where it was written.

    Writes to ``path`` if given, else to R2 (which must be configured).
    """
    if frame.empty:
        raise ValueError(f"Refusing to write an empty price history for {exchange}")
    table = _to_table(frame)
    options = {"row_group_size": ROW_GROUP_SIZE, "compression": "zstd"}

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
            temporary_path = Path(handle.name)
        try:
            pq.write_table(table, temporary_path, **options)
            temporary_path.chmod(0o644)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
        destination = str(path)
    else:
        config = config or r2_config()
        if config is None:
            raise RuntimeError("R2 is not configured; set the R2_* settings")
        destination = f"{config['bucket']}/{r2_key(exchange)}"
        # An S3 upload only becomes visible once complete, so readers never see a partial file.
        pq.write_table(table, destination, filesystem=_filesystem(config), **options)

    logger.info("Wrote %s price rows for %s to %s", table.num_rows, exchange, destination)
    return destination


def merge_prices(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append ``new`` to ``existing``; rows in ``new`` replace same symbol and date."""
    if existing.empty:
        return normalize_prices(new)
    return normalize_prices(pd.concat([existing, new], ignore_index=True))
