from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import harvest.prices as prices
from pipeline import historical_prices_core


def _stocks(*symbols):
    return pd.DataFrame({"symbol": list(symbols)})


def _prices(symbol, close=100.0):
    return pd.DataFrame(
        {"symbol": [symbol], "date": ["2026-08-14"], "close": [close]}
    )


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        historical_prices_core.run_historical_pipeline(mode="unknown")


def test_rejects_unknown_target():
    with pytest.raises(ValueError, match="target"):
        historical_prices_core.run_historical_pipeline(target="unknown")


def test_r2_target_merges_new_rows_into_stored_history():
    stored = pd.DataFrame(
        {
            "symbol": ["TEST", "TEST", "DELISTED"],
            "date": pd.to_datetime(["2026-08-13", "2026-08-14", "2020-01-02"]),
            "close": [80.0, 90.0, 50.0],
        }
    )

    with patch.object(
        historical_prices_core.hd, "get_all_idx_stocks", return_value=_stocks("TEST")
    ), patch.object(
        historical_prices_core, "_fetch_with_retries", return_value=_prices("TEST", close=100.0)
    ), patch.object(
        historical_prices_core.prices, "read_prices", return_value=stored
    ), patch.object(
        historical_prices_core.prices, "write_prices"
    ) as write, patch.object(historical_prices_core, "upsert_to_db") as upsert:
        summary = historical_prices_core.run_historical_pipeline(max_concurrency=1)

    upsert.assert_not_called()
    written = write.call_args.args[1]
    # The corrected close replaces the stored one; other symbols keep their history.
    assert written[["symbol", "close"]].values.tolist() == [
        ["DELISTED", 50.0],
        ["TEST", 80.0],
        ["TEST", 100.0],
    ]
    assert summary["target"] == "r2"
    assert summary["rows_stored"] == 3


def test_r2_target_refuses_to_overwrite_an_empty_history():
    with patch.object(
        historical_prices_core.hd, "get_all_idx_stocks", return_value=_stocks("TEST")
    ), patch.object(
        historical_prices_core, "_fetch_with_retries", return_value=_prices("TEST")
    ), patch.object(
        historical_prices_core.prices, "read_prices", return_value=pd.DataFrame()
    ), patch.object(historical_prices_core.prices, "write_prices") as write:
        with pytest.raises(RuntimeError, match="backfill"):
            historical_prices_core.run_historical_pipeline(max_concurrency=1)

    write.assert_not_called()


def test_r2_read_failure_stops_the_run():
    with patch.object(
        historical_prices_core.hd, "get_all_idx_stocks", return_value=_stocks("TEST")
    ), patch.object(
        historical_prices_core, "_fetch_with_retries", return_value=_prices("TEST")
    ), patch.object(
        historical_prices_core.prices, "read_prices", side_effect=OSError("R2 unavailable")
    ), patch.object(historical_prices_core.prices, "write_prices") as write:
        with pytest.raises(OSError, match="R2 unavailable"):
            historical_prices_core.run_historical_pipeline(max_concurrency=1)

    write.assert_not_called()


def test_r2_backfill_does_not_read_stored_history():
    with patch.object(
        historical_prices_core.hd, "get_all_idx_stocks", return_value=_stocks("TEST")
    ), patch.object(
        historical_prices_core, "_fetch_with_retries", return_value=_prices("TEST")
    ) as fetch, patch.object(
        historical_prices_core.prices, "read_prices"
    ) as read, patch.object(historical_prices_core.prices, "write_prices") as write:
        historical_prices_core.run_historical_pipeline(mode="backfill", max_concurrency=1)

    read.assert_not_called()
    write.assert_called_once()
    assert fetch.call_args.args[1] == historical_prices_core.BACKFILL_START_DATE


def test_partial_price_downloads_are_upserted():
    client = MagicMock()

    def fetch(symbol, _start_date):
        if symbol == "FAIL":
            raise RuntimeError("failed")
        return _prices(symbol)

    with patch.object(
        historical_prices_core.hd, "get_all_idx_stocks", return_value=_stocks("OK", "FAIL")
    ), patch.object(
        historical_prices_core, "_fetch_with_retries", side_effect=fetch
    ), patch.object(
        historical_prices_core, "get_supabase_client", return_value=client
    ), patch.object(historical_prices_core, "upsert_to_db") as upsert:
        summary = historical_prices_core.run_historical_pipeline(
            max_concurrency=2, target="supabase"
        )

    assert summary["symbols_succeeded"] == 1
    assert summary["failed_symbols"] == ["FAIL"]
    assert summary["rows_upserted"] == 1
    upsert.assert_called_once()


def test_empty_price_stage_fails():
    with patch.object(
        historical_prices_core.hd, "get_all_idx_stocks", return_value=_stocks("FAIL")
    ), patch.object(historical_prices_core, "_fetch_with_retries", return_value=None):
        with pytest.raises(RuntimeError, match="No historical prices"):
            historical_prices_core.run_historical_pipeline(max_concurrency=1)


def test_supplied_symbols_define_the_historical_price_universe():
    client = MagicMock()

    with patch.object(
        historical_prices_core.hd, "get_all_idx_stocks"
    ) as get_stocks, patch.object(
        historical_prices_core, "_fetch_with_retries", return_value=_prices("ACTIVE")
    ) as fetch, patch.object(
        historical_prices_core, "get_supabase_client", return_value=client
    ), patch.object(historical_prices_core, "upsert_to_db"):
        summary = historical_prices_core.run_historical_pipeline(
            symbols=["ACTIVE", "ACTIVE"], max_concurrency=1, target="supabase"
        )

    get_stocks.assert_not_called()
    assert fetch.call_args.args[0] == "ACTIVE"
    assert summary["symbols_total"] == 1


def test_local_cache_keeps_new_corrections(tmp_path):
    cache_path = tmp_path / "jkse" / "historical_prices.parquet"
    prices.write_prices(
        "jkse", pd.concat([_prices("TEST", close=90.0), _prices("INACTIVE")]), path=cache_path
    )
    client = MagicMock()

    with patch.object(
        historical_prices_core.hd, "get_all_idx_stocks", return_value=_stocks("TEST")
    ), patch.object(
        historical_prices_core, "_fetch_with_retries", return_value=_prices("TEST", close=100.0)
    ), patch.object(
        historical_prices_core, "get_supabase_client", return_value=client
    ), patch.object(historical_prices_core, "upsert_to_db"):
        historical_prices_core.run_historical_pipeline(
            max_concurrency=1,
            use_local_cache=True,
            cache_path=cache_path,
            symbols=["TEST"],
            target="supabase",
        )

    cached = prices.read_prices("jkse", path=cache_path)
    assert cached.loc[0, "close"] == 100.0
    assert cached["symbol"].tolist() == ["TEST"]


def test_database_failure_does_not_replace_local_cache(tmp_path):
    cache_path = tmp_path / "jkse" / "historical_prices.parquet"
    prices.write_prices("jkse", _prices("TEST", close=90.0), path=cache_path)
    original = prices.read_prices("jkse", path=cache_path)

    with patch.object(
        historical_prices_core.hd, "get_all_idx_stocks", return_value=_stocks("TEST")
    ), patch.object(
        historical_prices_core, "_fetch_with_retries", return_value=_prices("TEST", close=100.0)
    ), patch.object(
        historical_prices_core, "get_supabase_client", return_value=MagicMock()
    ), patch.object(
        historical_prices_core, "upsert_to_db", side_effect=RuntimeError("database failed")
    ):
        with pytest.raises(RuntimeError, match="database failed"):
            historical_prices_core.run_historical_pipeline(
                max_concurrency=1,
                use_local_cache=True,
                cache_path=cache_path,
                target="supabase",
            )

    pd.testing.assert_frame_equal(prices.read_prices("jkse", path=cache_path), original)


def test_prefect_historical_wrapper_delegates_to_plain_core():
    from pipeline import historical_prices

    with patch.object(
        historical_prices.core,
        "run_historical_pipeline",
        return_value={"rows_upserted": 1},
    ) as run_core:
        result = historical_prices.run_historical_pipeline.fn(
            exch="jkse",
            mode="incremental",
            max_concurrency=3,
            use_local_cache=True,
        )

    assert result == {"rows_upserted": 1}
    run_core.assert_called_once_with(
        exchange="jkse",
        mode="incremental",
        max_concurrency=3,
        use_local_cache=True,
        cache_path=None,
    )


def test_cli_passes_target_and_start_date():
    with patch.object(
        historical_prices_core, "run_historical_pipeline", return_value={}
    ) as run:
        code = historical_prices_core.main(
            ["--exchange", "jkse", "--target", "supabase", "--start-date", "2026-04-01"]
        )

    assert code == 0
    assert run.call_args.kwargs["target"] == "supabase"
    assert run.call_args.kwargs["start_date"] == "2026-04-01"


def test_cli_defaults_to_r2():
    args = historical_prices_core.build_parser().parse_args(["--exchange", "jkse"])

    assert args.target == "r2"
