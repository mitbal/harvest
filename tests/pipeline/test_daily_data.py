from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from pipeline import daily_data


PIPELINE_ENV = {
    "FMP_API_KEY": "fmp-test",
    "REDIS_URL": "redis://test",
    "SUPABASE_URL": "https://supabase.test",
    "SUPABASE_KEY": "supabase-test",
}


def test_validate_environment_reports_all_missing_variables():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(EnvironmentError) as error:
            daily_data.validate_environment()

    for variable in daily_data.REQUIRED_ENVIRONMENT:
        assert variable in str(error.value)


def test_parser_uses_current_year_and_exchange_defaults():
    args = daily_data.build_parser().parse_args(["--exchange", "jkse"])

    assert args.start_year == 2020
    assert args.end_year == datetime.now().year
    assert args.mcap_filter is None


def test_main_returns_nonzero_when_pipeline_fails():
    with patch.object(daily_data, "run_daily", side_effect=RuntimeError("failed")):
        assert daily_data.main(["--exchange", "jkse"]) == 1


def test_download_data_allows_partial_results():
    def download(symbol):
        if symbol == "FAIL":
            raise RuntimeError("unavailable")
        return {"symbol": symbol}

    with patch.object(daily_data, "DEFAULT_RETRIES", 0):
        result, summary = daily_data.download_data(
            ["OK", "FAIL"], download, "financial", max_concurrency=2
        )

    assert result == {"OK": {"symbol": "OK"}}
    assert summary["succeeded"] == 1
    assert summary["failed_symbols"] == ["FAIL"]


def test_download_data_fails_when_stage_is_empty():
    with patch.object(daily_data, "DEFAULT_RETRIES", 0):
        with pytest.raises(RuntimeError, match="No dividend data"):
            daily_data.download_data(
                ["FAIL"],
                lambda _symbol: None,
                "dividend",
                max_concurrency=1,
            )


def test_download_data_treats_empty_dataframes_as_failures():
    with patch.object(daily_data, "DEFAULT_RETRIES", 0):
        with pytest.raises(RuntimeError, match="No financial data"):
            daily_data.download_data(
                ["EMPTY"],
                lambda _symbol: pd.DataFrame(),
                "financial",
                max_concurrency=1,
            )


def test_store_frames_to_redis_uses_one_transaction():
    client = MagicMock()
    transaction = client.pipeline.return_value.__enter__.return_value
    frames = {
        "first": pd.DataFrame({"value": [1]}),
        "second": pd.DataFrame({"value": [2]}),
    }

    with patch.dict("os.environ", PIPELINE_ENV, clear=True), patch.object(
        daily_data.redis, "from_url", return_value=client
    ), patch.object(daily_data, "_serialize_dataframe", side_effect=[b"one", b"two"]):
        daily_data.store_frames_to_redis(frames)

    assert transaction.set.call_args_list == [call("first", b"one"), call("second", b"two")]
    transaction.execute.assert_called_once_with()
    client.close.assert_called_once_with()


def test_run_daily_uses_exchange_defaults_and_current_calendar_year():
    profiles = pd.DataFrame(
        {
            "isActivelyTrading": [True],
            "mktCap": [200_000_000_000],
            "price": [100.0],
        },
        index=pd.Index(["TEST.JK"], name="symbol"),
    )
    score = pd.DataFrame(
        {"DScore": [1.0]}, index=pd.Index(["TEST.JK"], name="symbol")
    )
    download_summary = {
        "type": "test",
        "total": 1,
        "succeeded": 1,
        "failed": 0,
        "failed_symbols": [],
        "success_rate": 100.0,
    }
    calendar = pd.DataFrame({"date": ["2026-01-01"], "symbol": ["TEST.JK"]})
    supabase = MagicMock()

    with patch.dict("os.environ", PIPELINE_ENV, clear=True), patch.object(
        daily_data, "_load_company_profiles", return_value=(profiles, ["TEST.JK"])
    ), patch.object(
        daily_data,
        "download_data",
        side_effect=[({"TEST.JK": pd.DataFrame()}, download_summary)] * 2,
    ), patch.object(
        daily_data,
        "run_historical_pipeline",
        return_value={"rows_upserted": 1},
    ), patch.object(
        daily_data, "get_supabase_client", return_value=supabase
    ), patch.object(
        daily_data, "refresh_returns_view", return_value=None
    ), patch.object(
        daily_data, "_add_syariah_status", side_effect=lambda frame: frame.assign(is_syariah=True)
    ), patch.object(
        daily_data,
        "get_latest_returns_from_db",
        return_value=pd.DataFrame(
            {
                "return_7d": [0.01],
                "return_1m": [0.02],
                "return_1y": [0.1],
                "return_10y": [1.0],
            },
            index=pd.Index(["TEST.JK"], name="symbol"),
        ),
    ), patch.object(
        daily_data, "compute_div_score", return_value=score
    ), patch.object(
        daily_data, "prepare_dividend_calendar", return_value=calendar
    ) as prepare_calendar, patch.object(
        daily_data, "store_frames_to_redis"
    ) as store_redis, patch.object(
        daily_data, "store_pickle_to_supabase_storage"
    ) as store_storage:
        summary = daily_data.run_daily(exchange="jkse")

    years = list(range(2020, datetime.now().year + 1))
    assert prepare_calendar.call_count == len(years)
    assert prepare_calendar.call_args_list[-1].args[-1] == years[-1]
    assert all(
        args.args[2] == daily_data.DEFAULT_MCAP_FILTERS["jkse"]
        for args in prepare_calendar.call_args_list
    )
    assert summary["calendar_years"] == years
    assert f"div_cal_jkse_{years[-1]}" in store_redis.call_args.args[0]
    assert store_storage.call_count == 2


def test_storage_upload_errors_propagate():
    client = MagicMock()
    client.storage.from_.return_value.upload.side_effect = RuntimeError("storage failed")

    with pytest.raises(RuntimeError, match="storage failed"):
        daily_data.store_pickle_to_supabase_storage("data/test.pkl", {}, client=client)


def test_returns_view_read_errors_propagate():
    client = MagicMock()
    client.table.return_value.select.return_value.range.return_value.execute.side_effect = (
        RuntimeError("database unavailable")
    )

    with pytest.raises(RuntimeError, match="Unable to fetch returns"):
        daily_data.get_latest_returns_from_db(client)


def test_prefect_daily_wrapper_delegates_to_plain_core():
    from pipeline import daily_panen_data

    summary = {"exchange": "sp500"}
    with patch.object(
        daily_panen_data, "run_daily_core", return_value=summary
    ) as run_core, patch.object(daily_panen_data, "create_markdown_artifact"):
        result = daily_panen_data.run_daily.fn(
            exch="sp500",
            mcap_filter=10_000_000_000,
            dividend_years=[2026],
        )

    assert result == summary
    run_core.assert_called_once_with(
        exchange="sp500",
        mcap_filter=10_000_000_000,
        dividend_years=[2026],
        use_local_price_cache=True,
    )
