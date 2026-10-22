from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from pipeline import daily_data


PIPELINE_ENV = {
    "FMP_API_KEY": "fmp-test",
    "REDIS_URL": "redis://test",
    "SUPABASE_URL": "https://supabase.test",
    "SUPABASE_KEY": "supabase-test",
    "R2_ACCOUNT_ID": "account",
    "R2_ACCESS_KEY_ID": "key",
    "R2_SECRET_ACCESS_KEY": "secret",
    "R2_BUCKET": "bucket",
}


def test_validate_environment_reports_all_missing_variables():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(EnvironmentError) as error:
            daily_data.validate_environment()

    for variable in daily_data.REQUIRED_ENVIRONMENT + daily_data.R2_ENVIRONMENT:
        assert variable in str(error.value)


def test_validate_environment_requires_r2():
    env = {name: "x" for name in daily_data.REQUIRED_ENVIRONMENT}
    with patch.dict("os.environ", env, clear=True):
        with pytest.raises(EnvironmentError, match="R2_BUCKET"):
            daily_data.validate_environment()


def _history(symbol, closes):
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": pd.bdate_range("2015-01-01", periods=len(closes)),
            "close": closes,
        }
    )


def test_compute_latest_returns_counts_sessions_back_from_latest_close():
    long = _history("LONG", np.arange(1.0, 2522.0))  # 2521 sessions, enough for 10 years
    short = _history("SHORT", [100.0] * 5 + [110.0])  # exactly one 7-day window
    returns = daily_data.compute_latest_returns(pd.concat([short, long]).sample(frac=1, random_state=0))

    assert returns.loc["LONG", "return_7d"] == pytest.approx(2521 / 2516 - 1)
    assert returns.loc["LONG", "return_1m"] == pytest.approx(2521 / 2500 - 1)
    assert returns.loc["LONG", "return_1y"] == pytest.approx(2521 / 2269 - 1)
    assert returns.loc["LONG", "return_10y"] == pytest.approx(2521 / 1 - 1)
    assert returns.loc["SHORT", "return_7d"] == pytest.approx(0.1)
    assert returns.loc["SHORT", ["return_1m", "return_1y", "return_10y"]].isna().all()


def test_compute_latest_returns_handles_empty_history():
    returns = daily_data.compute_latest_returns(pd.DataFrame())

    assert returns.empty
    assert list(returns.columns) == list(daily_data.RETURN_WINDOWS)


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


def test_download_data_separates_errors_from_missing_data():
    def download(symbol):
        if symbol == "ERROR":
            raise RuntimeError("503 Service Unavailable")
        if symbol == "MISSING":
            return None
        return pd.DataFrame({"value": [1]})

    with patch.object(daily_data, "DEFAULT_RETRIES", 0):
        _data, summary = daily_data.download_data(
            ["OK", "MISSING", "ERROR"], download, "dividend", max_concurrency=1
        )

    assert summary["failed_symbols"] == ["ERROR", "MISSING"]
    assert summary["errors"] == 1
    assert summary["error_symbols"] == ["ERROR"]


def _write_snapshot(root):
    (root / "jkse" / "dividends").mkdir(parents=True)
    (root / "jkse" / "income_statements").mkdir(parents=True)
    (root / "jkse" / "dividends" / "PAYS.csv").write_text(
        "ex_date,dividend\n2026-06-09,170\n"
    )
    (root / "jkse" / "income_statements" / "PAYS.csv").write_text(
        "date,revenue,net_income\n2026-03-31,100,10\n2026-06-30,120,12\n"
    )


def test_download_fundamentals_reads_jkse_from_one_snapshot(tmp_path):
    _write_snapshot(tmp_path)

    with patch.object(
        daily_data.hd, "daguerreo_snapshot", return_value=nullcontext(tmp_path)
    ) as snapshot, patch.object(daily_data.hd.requests, "get") as get:
        dividends, dividend_summary, financials, financial_summary = (
            daily_data.download_fundamentals("jkse", ["PAYS.JK", "NODATA.JK"], 2)
        )

    snapshot.assert_called_once_with()
    get.assert_not_called()
    assert dividends["PAYS.JK"]["adjDividend"].tolist() == [170]
    assert financials["PAYS.JK"]["netIncome"].tolist() == [12, 10]
    # A stock missing from the snapshot is "no data", not an error.
    assert dividend_summary["failed_symbols"] == ["NODATA.JK"]
    assert dividend_summary["errors"] == 0
    assert financial_summary["failed_symbols"] == ["NODATA.JK"]


def test_download_fundamentals_skips_snapshot_for_sp500():
    with patch.object(daily_data.hd, "daguerreo_snapshot") as snapshot, patch.object(
        daily_data, "download_single_us_dividend", return_value=pd.DataFrame({"x": [1]})
    ), patch.object(
        daily_data, "download_single_us_financial", return_value=pd.DataFrame({"x": [1]})
    ):
        dividends, _, financials, _ = daily_data.download_fundamentals("sp500", ["AAPL"], 1)

    snapshot.assert_not_called()
    assert list(dividends) == list(financials) == ["AAPL"]


def _summary(total, error_symbols):
    return {
        "type": "dividend",
        "total": total,
        "errors": len(error_symbols),
        "error_symbols": error_symbols,
    }


def test_check_download_errors_allows_missing_data_and_few_errors():
    daily_data.check_download_errors(_summary(100, ["A.JK"] * 5), max_error_rate=0.05)
    daily_data.check_download_errors({"type": "dividend", "total": 0}, max_error_rate=0.05)


def test_check_download_errors_rejects_widespread_outage():
    with pytest.raises(RuntimeError, match="6 of 100 dividend downloads errored"):
        daily_data.check_download_errors(_summary(100, ["A.JK"] * 6), max_error_rate=0.05)


def _score_profiles(symbols):
    return pd.DataFrame(
        {
            "price": [1000.0] * len(symbols),
            "mktCap": [200_000_000_000] * len(symbols),
            "ipoDate": ["2010-01-04"] * len(symbols),
            "lastDiv": [0.0] * len(symbols),
            "sector": ["Industrials"] * len(symbols),
            "industry": ["Conglomerates"] * len(symbols),
            "is_syariah": [True] * len(symbols),
            "state_owned": ["Swasta"] * len(symbols),
        },
        index=pd.Index(symbols, name="symbol"),
    )


def test_compute_div_score_keeps_dividends_without_financials():
    """A missing income statement used to skip the dividend metrics too."""
    dividends = {
        "PAYER.JK": pd.DataFrame(
            {
                "date": ["2025-06-11", "2026-06-09"],
                "adjDividend": [120.0, 170.0],
                "fiscal_year": [2024, 2025],
                "dividend_type": ["final", "final"],
            }
        )
    }

    with patch.object(
        daily_data.hd, "calc_latest_finalized_dividend_sum", return_value=170.0
    ):
        frame = daily_data.compute_div_score(
            _score_profiles(["PAYER.JK", "NONPAYER.JK"]), {}, dividends
        )

    assert frame.loc["PAYER.JK", "lastDiv"] == 170.0
    assert frame.loc["PAYER.JK", "scoreDiv"] == 170.0
    assert frame.loc["PAYER.JK", "yield"] == pytest.approx(17.0)
    assert frame.loc["PAYER.JK", "numDividendYear"] > 0
    assert np.isnan(frame.loc["PAYER.JK", "earningTTM"])
    assert frame.loc["NONPAYER.JK", "lastDiv"] == 0.0
    assert frame["numOfYear"].tolist() == [datetime.now().year - 2010] * 2
    assert frame["state_owned"].tolist() == ["Swasta", "Swasta"]


def test_add_state_owned_status_defaults_unlisted_stocks_to_swasta():
    profiles = pd.DataFrame(
        {"price": [1.0, 2.0, 3.0]},
        index=pd.Index(["BBRI.JK", "PTBA.JK", "ASII.JK"], name="symbol"),
    )

    tagged = daily_data._add_state_owned_status(profiles)

    assert tagged["state_owned"].to_dict() == {
        "BBRI.JK": "BUMN",
        "PTBA.JK": "Anak BUMN",
        "ASII.JK": "Swasta",
    }
    assert tagged.index.name == "symbol"


def test_load_company_profiles_keeps_only_listed_active_stocks():
    listed = pd.DataFrame({"symbol": ["ACTIVE", "INACTIVE", "ACTIVE"]})
    profiles = pd.DataFrame(
        {
            "symbol": ["ACTIVE", "INACTIVE", "UNLISTED"],
            "isActivelyTrading": [True, False, True],
        }
    )

    with patch.object(
        daily_data.hd, "get_all_idx_stocks", return_value=listed
    ), patch.object(
        daily_data.hd, "get_company_profile", return_value=profiles
    ) as get_profiles:
        result, symbols = daily_data._load_company_profiles("jkse")

    get_profiles.assert_called_once_with(["ACTIVE", "INACTIVE"])
    assert result.index.tolist() == ["ACTIVE"]
    assert symbols == ["ACTIVE"]


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
        daily_data.hd, "daguerreo_snapshot", return_value=nullcontext(Path("/snapshot"))
    ), patch.object(
        daily_data,
        "run_historical_pipeline",
        return_value={"rows_upserted": 1},
    ) as run_historical, patch.object(
        daily_data, "get_supabase_client", return_value=supabase
    ), patch.object(
        daily_data, "_add_syariah_status", side_effect=lambda frame: frame.assign(is_syariah=True)
    ), patch.object(
        daily_data,
        "_add_state_owned_status",
        side_effect=lambda frame: frame.assign(state_owned="Swasta"),
    ), patch.object(
        daily_data,
        "get_latest_returns_from_prices",
        return_value=pd.DataFrame(
            {
                "return_7d": [0.01, 0.02],
                "return_1m": [0.02, 0.03],
                "return_1y": [0.1, 0.2],
                "return_10y": [1.0, 2.0],
            },
            index=pd.Index(["TEST.JK", "INACTIVE.JK"], name="symbol"),
        ),
    ) as get_returns, patch.object(
        daily_data, "compute_div_score", return_value=score
    ) as compute_score, patch.object(
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
    run_historical.assert_called_once_with(
        exchange="jkse",
        mode="incremental",
        max_concurrency=daily_data.DEFAULT_MAX_CONCURRENCY,
        use_local_cache=False,
        symbols=["TEST.JK"],
    )
    get_returns.assert_called_once_with("jkse", ["TEST.JK"])
    assert compute_score.call_args.kwargs["latest_returns"].index.tolist() == ["TEST.JK"]
    assert summary["returns_rows"] == 1
    redis_frames = store_redis.call_args.args[0]
    assert f"div_cal_jkse_{years[-1]}" in redis_frames
    assert redis_frames["latest_returns_jkse"]["symbol"].tolist() == ["TEST.JK"]
    assert redis_frames["latest_returns_jkse"]["return_1y"].tolist() == [0.1]
    assert store_storage.call_count == 2


def test_run_daily_does_not_publish_when_dividend_downloads_error():
    profiles = pd.DataFrame(
        {"isActivelyTrading": [True], "mktCap": [200_000_000_000], "price": [100.0]},
        index=pd.Index(["TEST.JK"], name="symbol"),
    )
    dividend_summary = {
        "type": "dividend",
        "total": 10,
        "succeeded": 4,
        "failed": 6,
        "failed_symbols": [],
        "errors": 6,
        "error_symbols": ["A.JK"] * 6,
        "success_rate": 40.0,
    }

    with patch.dict("os.environ", PIPELINE_ENV, clear=True), patch.object(
        daily_data, "_load_company_profiles", return_value=(profiles, ["TEST.JK"])
    ), patch.object(
        daily_data,
        "download_data",
        side_effect=[({}, dividend_summary), ({}, dict(dividend_summary, type="financial"))],
    ), patch.object(
        daily_data.hd, "daguerreo_snapshot", return_value=nullcontext(Path("/snapshot"))
    ), patch.object(
        daily_data, "run_historical_pipeline", return_value={}
    ) as run_historical, patch.object(
        daily_data, "get_supabase_client"
    ), patch.object(
        daily_data, "store_frames_to_redis"
    ) as store_redis, patch.object(
        daily_data, "store_pickle_to_supabase_storage"
    ) as store_storage:
        with pytest.raises(RuntimeError, match="not publishing"):
            daily_data.run_daily(exchange="jkse")

    run_historical.assert_called_once()
    store_redis.assert_not_called()
    store_storage.assert_not_called()


def test_storage_upload_errors_propagate():
    client = MagicMock()
    client.storage.from_.return_value.upload.side_effect = RuntimeError("storage failed")

    with pytest.raises(RuntimeError, match="storage failed"):
        daily_data.store_pickle_to_supabase_storage("data/test.pkl", {}, client=client)


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
