import pickle
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

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
        summary = historical_prices_core.run_historical_pipeline(max_concurrency=2)

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


def test_local_cache_keeps_new_corrections(tmp_path):
    cache_path = tmp_path / "historical_prices.pkl"
    with cache_path.open("wb") as cache_file:
        pickle.dump(_prices("TEST", close=90.0), cache_file)
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
        )

    with cache_path.open("rb") as cache_file:
        cached = pickle.load(cache_file)
    assert cached.loc[0, "close"] == 100.0


def test_database_failure_does_not_replace_local_cache(tmp_path):
    cache_path = tmp_path / "historical_prices.pkl"
    original = _prices("TEST", close=90.0)
    with cache_path.open("wb") as cache_file:
        pickle.dump(original, cache_file)

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
            )

    with cache_path.open("rb") as cache_file:
        cached = pickle.load(cache_file)
    pd.testing.assert_frame_equal(cached, original)


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
