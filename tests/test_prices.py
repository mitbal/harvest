from unittest.mock import patch

import pandas as pd
import pytest

import harvest.prices as prices


R2_ENV = {
    "R2_ACCOUNT_ID": "account",
    "R2_ACCESS_KEY_ID": "key",
    "R2_SECRET_ACCESS_KEY": "secret",
    "R2_BUCKET": "bucket",
}


def _frame():
    return pd.DataFrame(
        {
            "symbol": ["BBRI.JK", "BBRI.JK", "BBRI.JK", "TLKM.JK", "TLKM.JK"],
            "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-02", "2024-01-03"],
            "close": [5000.0, 5050.0, 5100.0, 3900.0, 3950.0],
            "volume": [100, 200, 300, 400, 500],
        }
    )


@pytest.fixture
def local_file(tmp_path, monkeypatch):
    """A local price file with R2 unconfigured, so load_prices uses the fallback."""
    path = tmp_path / "jkse" / "historical_prices.parquet"
    prices.write_prices("jkse", _frame(), path=path)
    monkeypatch.setattr(prices, "local_path", lambda exchange: path)
    for name in R2_ENV:
        monkeypatch.delenv(name, raising=False)
    return path


def test_load_prices_reads_the_local_fallback(local_file):
    frame = prices.load_prices("jkse")

    assert list(frame.columns) == ["symbol", "date", "close"]
    assert len(frame) == 5
    assert frame["date"].dtype == "datetime64[ns]"
    assert frame["close"].dtype == "float64"
    assert frame.loc[0, "close"] == 5000.0


def test_load_prices_filters_symbols(local_file):
    frame = prices.load_prices("jkse", symbols=("TLKM.JK",))

    assert frame["symbol"].unique().tolist() == ["TLKM.JK"]
    assert frame["close"].tolist() == [3900.0, 3950.0]


def test_load_prices_filters_dates_inclusively(local_file):
    frame = prices.load_prices("jkse", start="2024-01-03", end="2024-01-03")

    assert frame["date"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2024-01-03"]
    assert frame["symbol"].tolist() == ["BBRI.JK", "TLKM.JK"]


def test_load_prices_selects_columns(local_file):
    frame = prices.load_prices("jkse", symbols=["BBRI.JK"], columns=("close", "volume"))

    assert list(frame.columns) == ["symbol", "date", "close", "volume"]
    assert frame["volume"].dtype == "int64"
    assert frame["volume"].tolist() == [100, 200, 300]


def test_load_prices_with_no_symbols_is_empty(local_file):
    frame = prices.load_prices("jkse", symbols=[])

    assert frame.empty
    assert list(frame.columns) == ["symbol", "date", "close"]


def test_load_prices_unknown_symbol_is_empty(local_file):
    assert prices.load_prices("jkse", symbols=["NOPE.JK"]).empty


def test_load_prices_without_any_source_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "local_path", lambda exchange: tmp_path / "missing.parquet")
    with patch.dict("os.environ", {}, clear=True):
        frame = prices.load_prices("jkse")

    assert frame.empty
    assert list(frame.columns) == ["symbol", "date", "close"]


def test_load_prices_falls_back_when_r2_fails(local_file):
    with patch.dict("os.environ", R2_ENV), patch.object(
        prices, "_filesystem", side_effect=OSError("R2 unavailable")
    ) as filesystem:
        frame = prices.load_prices("jkse", symbols=["BBRI.JK"])

    filesystem.assert_called_once()
    assert len(frame) == 3


def test_load_prices_prefers_r2(local_file):
    from_r2 = pd.DataFrame({"symbol": ["X"], "date": pd.to_datetime(["2024-01-02"]), "close": [1.0]})
    with patch.dict("os.environ", R2_ENV), patch.object(
        prices, "read_prices", return_value=from_r2
    ) as read:
        frame = prices.load_prices("jkse", symbols=["X"], start="2024-01-01")

    assert frame is from_r2
    assert read.call_args.kwargs["config"]["bucket"] == "bucket"


def test_read_prices_requires_r2_without_a_path():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="R2 is not configured"):
            prices.read_prices("jkse")


def test_read_prices_with_no_columns_reads_everything(local_file):
    frame = prices.read_prices("jkse", columns=None, path=local_file)

    assert list(frame.columns) == ["symbol", "date", "close", "volume"]


def test_r2_config_needs_every_field():
    with patch.dict("os.environ", R2_ENV, clear=True):
        assert prices.r2_config() == {
            "account_id": "account",
            "access_key_id": "key",
            "secret_access_key": "secret",
            "bucket": "bucket",
        }
    with patch.dict("os.environ", {**R2_ENV, "R2_BUCKET": ""}, clear=True):
        assert prices.r2_config() is None


def test_write_prices_stores_compact_sorted_rows(tmp_path):
    path = tmp_path / "prices.parquet"
    shuffled = _frame()[["symbol", "date", "close"]].iloc[::-1]
    prices.write_prices("jkse", shuffled, path=path)

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    assert str(table.schema.field("date").type) == "date32[day]"
    assert str(table.schema.field("close").type) == "float"
    assert table.column("symbol").to_pylist() == ["BBRI.JK"] * 3 + ["TLKM.JK"] * 2


def test_write_prices_refuses_an_empty_frame(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        prices.write_prices("jkse", pd.DataFrame(), path=tmp_path / "prices.parquet")


def test_write_prices_requires_r2_without_a_path():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="R2 is not configured"):
            prices.write_prices("jkse", _frame())


def test_merge_prices_replaces_overlapping_rows():
    existing = _frame()[["symbol", "date", "close"]]
    new = pd.DataFrame(
        {"symbol": ["BBRI.JK", "BBRI.JK"], "date": ["2024-01-04", "2024-01-05"], "close": [5200.0, 5300.0]}
    )
    merged = prices.merge_prices(existing, new)

    bbri = merged[merged["symbol"] == "BBRI.JK"]
    assert bbri["close"].tolist() == [5000.0, 5050.0, 5200.0, 5300.0]
    assert len(merged) == 6


def test_normalize_prices_drops_unusable_rows():
    frame = pd.DataFrame(
        {"symbol": ["A", "A", "A"], "date": ["2024-01-02", "not a date", "2024-01-03"], "close": [1.0, 2.0, None]}
    )

    assert prices.normalize_prices(frame)["close"].tolist() == [1.0]
