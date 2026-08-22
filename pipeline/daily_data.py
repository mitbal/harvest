import argparse
import json
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import redis
from supabase import Client, create_client

import harvest.data as hd
from pipeline.historical_prices_core import run_historical_pipeline


logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 30
DEFAULT_MAX_CONCURRENCY = 5
DEFAULT_MCAP_FILTERS = {
    "jkse": 100_000_000_000,
    "sp500": 10_000_000_000,
}
REQUIRED_ENVIRONMENT = ("FMP_API_KEY", "REDIS_URL", "SUPABASE_URL", "SUPABASE_KEY")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def validate_environment() -> None:
    missing = [name for name in REQUIRED_ENVIRONMENT if not os.environ.get(name)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")


def get_supabase_client() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def _serialize_dataframe(frame: pd.DataFrame) -> bytes:
    return frame.to_parquet(engine="pyarrow", compression="snappy")


def store_frames_to_redis(frames: dict[str, pd.DataFrame]) -> None:
    """Publish all Redis DataFrame outputs in one transaction."""
    payloads = {key: _serialize_dataframe(frame) for key, frame in frames.items()}
    client = redis.from_url(
        os.environ["REDIS_URL"],
        socket_connect_timeout=10,
        socket_timeout=30,
        socket_keepalive=True,
        retry_on_timeout=True,
    )
    try:
        with client.pipeline(transaction=True) as transaction:
            for key, payload in payloads.items():
                transaction.set(key, payload)
            transaction.execute()
    finally:
        client.close()


def store_pickle_to_supabase_storage(
    filename: str,
    content: dict,
    client: Optional[Client] = None,
) -> None:
    """Upsert a pickle payload using the existing storage format and path."""
    supabase = client or get_supabase_client()
    supabase.storage.from_("harvest_dividend").upload(
        path=filename,
        file=pickle.dumps(content),
        file_options={"contentType": "application/octet-stream", "upsert": "true"},
    )


def refresh_returns_view(client: Optional[Client] = None) -> Optional[str]:
    """Request a materialized-view refresh, tolerating remote timeout behavior."""
    try:
        (client or get_supabase_client()).rpc("refresh_latest_returns").execute()
        return None
    except Exception as exc:
        message = str(exc)
        logger.warning(
            "Returns-view refresh returned an error; Postgres may still be refreshing: %s",
            message,
        )
        return message


def get_latest_returns_from_db(client: Optional[Client] = None) -> pd.DataFrame:
    """Fetch all precalculated returns from the materialized view."""
    supabase = client or get_supabase_client()
    records = []
    page_size = 1000
    offset = 0
    try:
        while True:
            response = (
                supabase.table("mat_view_latest_returns")
                .select("*")
                .range(offset, offset + page_size - 1)
                .execute()
            )
            page = response.data or []
            if not page:
                break
            records.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
    except Exception as exc:
        raise RuntimeError("Unable to fetch returns from materialized view") from exc

    frame = pd.DataFrame(records)
    if not frame.empty and "symbol" in frame.columns:
        frame = frame.set_index("symbol")
    return frame


def _download_with_retries(
    download_func: Callable[[str], object],
    stock: str,
    data_type: str,
):
    for attempt in range(DEFAULT_RETRIES + 1):
        try:
            return download_func(stock)
        except Exception as exc:
            if attempt == DEFAULT_RETRIES:
                raise
            logger.warning(
                "%s download failed for %s: %s; retrying in %ss (%s/%s)",
                data_type.capitalize(),
                stock,
                exc,
                DEFAULT_RETRY_DELAY,
                attempt + 1,
                DEFAULT_RETRIES,
            )
            time.sleep(DEFAULT_RETRY_DELAY)
    return None


def download_data(
    stock_list: list[str],
    download_func: Callable[[str], object],
    data_type: str,
    max_concurrency: int,
) -> tuple[dict, dict]:
    """Download one data type in parallel and return data plus completeness summary."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")

    data = {}
    failed_symbols = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(_download_with_retries, download_func, stock, data_type): stock
            for stock in stock_list
        }
        for future in as_completed(futures):
            stock = futures[future]
            try:
                result = future.result()
                if result is not None and not (
                    isinstance(result, pd.DataFrame) and result.empty
                ):
                    data[stock] = result
                else:
                    failed_symbols.append(stock)
            except Exception as exc:
                failed_symbols.append(stock)
                logger.error("%s download exhausted retries for %s: %s", data_type, stock, exc)

    if stock_list and not data:
        raise RuntimeError(f"No {data_type} data was downloaded")

    summary = {
        "type": data_type,
        "total": len(stock_list),
        "succeeded": len(data),
        "failed": len(failed_symbols),
        "failed_symbols": sorted(failed_symbols),
        "success_rate": (len(data) / len(stock_list) * 100) if stock_list else 0.0,
    }
    logger.info("Download summary: %s", summary)
    return data, summary


def download_single_financial(stock: str):
    return hd.get_financial_data(stock, period="quarter")


def download_single_dividend(stock: str):
    return hd.get_dividend_history_single_stock(stock, source="dag")


def download_single_us_dividend(stock: str):
    return hd.get_dividend_history_single_stock(stock, source="fmp")


def prepare_dividend_calendar(
    company_profiles: pd.DataFrame,
    dividends: dict,
    mcap_filter: int,
    year: int,
) -> pd.DataFrame:
    filtered_profiles = company_profiles[company_profiles["mktCap"] >= mcap_filter].copy()
    filtered_profiles.reset_index(drop=False, inplace=True)
    return hd.prep_div_cal(dividends, filtered_profiles, year=year)


def _safe_dividend_sums(dividend_frame: pd.DataFrame) -> tuple[float, float]:
    if dividend_frame.empty or not {"date", "adjDividend"}.issubset(dividend_frame.columns):
        return 0.0, 0.0
    dates = pd.to_datetime(dividend_frame["date"], errors="coerce")
    values = pd.to_numeric(dividend_frame["adjDividend"], errors="coerce").fillna(0)
    now = pd.Timestamp.now()
    return (
        float(values[dates >= now - pd.Timedelta(days=365)].sum()),
        float(values[dates >= now - pd.Timedelta(days=3650)].sum()),
    )


def compute_div_score(
    company_profiles: pd.DataFrame,
    financials: dict,
    dividends: dict,
    exchange: str = "jkse",
    latest_returns: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute dividend and return metrics for each company profile."""
    frame = company_profiles.copy()
    initialized_columns = {
        "yield": 0.0,
        "lastDiv": 0.0,
        "revenueGrowth": np.nan,
        "revenueCAGR5Y": np.nan,
        "revenueGrowthTTM": np.nan,
        "netIncomeGrowth": np.nan,
        "netIncomeGrowthTTM": np.nan,
        "medianProfitMargin": np.nan,
        "earningTTM": np.nan,
        "revenueTTM": np.nan,
        "avgFlatAnnualDivIncrease": np.nan,
        "avgPctAnnualDivIncrease": np.nan,
        "numDividendYear": np.nan,
        "positiveYear": np.nan,
        "numOfYear": np.nan,
        "maximumCutPct": 0.0,
        "max10CutPct": 0.0,
        "div_sum_1y": 0.0,
        "div_sum_10y": 0.0,
    }
    for column, value in initialized_columns.items():
        frame[column] = value
    frame["mktCap"] = pd.to_numeric(frame["mktCap"], errors="raise").astype("int64")

    for symbol in frame.index.tolist():
        try:
            financial_frame = financials[symbol]
            currency = "IDR" if exchange == "jkse" else "USD"
            financial_stats = hd.calc_fin_stats(financial_frame, target_currency=currency)
            frame.loc[symbol, "revenueGrowth"] = financial_stats["trim_mean_10y_revenue_growth"]
            frame.loc[symbol, "revenueCAGR5Y"] = financial_stats["cagr_5y_revenue"]
            frame.loc[symbol, "revenueGrowthTTM"] = financial_stats["revenue_growth_TTM"]
            frame.loc[symbol, "netIncomeGrowth"] = financial_stats["trim_mean_10y_netIncome_growth"]
            frame.loc[symbol, "netIncomeGrowthTTM"] = financial_stats["netIncome_growth_TTM"]
            frame.loc[symbol, "medianProfitMargin"] = financial_stats["median_profit_margin"]
            frame.loc[symbol, "revenueTTM"] = financial_stats["revenueTTM"]
            frame.loc[symbol, "earningTTM"] = financial_stats["earningTTM"]

            dividend_frame = dividends[symbol].copy()
            if exchange == "jkse":
                ordinary = dividend_frame[dividend_frame["dividend_type"] != "special"]
                aggregate = ordinary.groupby("fiscal_year")["adjDividend"].sum()
                final_years = dividend_frame.loc[
                    dividend_frame["dividend_type"] == "final", "fiscal_year"
                ]
                final_year = final_years.iloc[0] if not final_years.empty else None
                last_dividend = (
                    float(aggregate.get(final_year, 0))
                    if final_year is not None and final_year >= datetime.now().year - 2
                    else 0.0
                )
            else:
                last_dividend = float(company_profiles.loc[symbol, "lastDiv"])

            one_year_sum, ten_year_sum = _safe_dividend_sums(dividend_frame)
            frame.loc[symbol, "div_sum_1y"] = one_year_sum
            frame.loc[symbol, "div_sum_10y"] = ten_year_sum

            dividend_stats = hd.calc_div_stats(hd.preprocess_div(dividend_frame))
            dividend_increases = np.nan_to_num(
                np.array(
                    [
                        dividend_stats["historical_mean_flat"],
                        dividend_stats["div_inc_5y_mean_flat"],
                    ]
                ),
                nan=0.0,
            )
            price = float(frame.loc[symbol, "price"])
            frame.loc[symbol, "lastDiv"] = last_dividend
            frame.loc[symbol, "yield"] = last_dividend / price * 100 if price else np.nan
            frame.loc[symbol, "avgFlatAnnualDivIncrease"] = np.min(dividend_increases)
            frame.loc[symbol, "avgPctAnnualDivIncrease"] = dividend_stats["historical_mean_pct"]
            frame.loc[symbol, "maximumCutPct"] = dividend_stats["maximum_cut_pct"]
            frame.loc[symbol, "max10CutPct"] = dividend_stats["max_10y_cut_pct"]
            frame.loc[symbol, "numDividendYear"] = dividend_stats["num_dividend_year"]
            frame.loc[symbol, "positiveYear"] = dividend_stats["num_positive_year"]
            ipo_year = datetime.strptime(frame.loc[symbol, "ipoDate"], "%Y-%m-%d").year
            frame.loc[symbol, "numOfYear"] = datetime.now().year - ipo_year
        except Exception as exc:
            logger.warning("Unable to compute dividend metrics for %s: %s", symbol, exc)

    frame["peRatio"] = frame["mktCap"] / frame["earningTTM"]
    frame["psRatio"] = frame["mktCap"] / frame["revenueTTM"]
    frame["DScore"] = hd.calc_div_score(frame)

    return_columns = ["return_7d", "return_1m", "return_1y", "return_10y"]
    if latest_returns is not None and not latest_returns.empty:
        available = [column for column in return_columns if column in latest_returns.columns]
        frame = frame.join(latest_returns[available])
    for column in return_columns:
        if column not in frame.columns:
            frame[column] = np.nan

    one_year_start_price = frame["price"] / (1 + frame["return_1y"])
    ten_year_start_price = frame["price"] / (1 + frame["return_10y"])
    frame["total_return_1y"] = frame["return_1y"] + frame["div_sum_1y"] / one_year_start_price
    frame["total_return_10y"] = frame["return_10y"] + frame["div_sum_10y"] / ten_year_start_price
    frame[["total_return_1y", "total_return_10y"]] = frame[
        ["total_return_1y", "total_return_10y"]
    ].replace([np.inf, -np.inf], np.nan)

    features = [
        "price",
        "lastDiv",
        "yield",
        "sector",
        "industry",
        "mktCap",
        "ipoDate",
        "revenueGrowth",
        "revenueCAGR5Y",
        "revenueGrowthTTM",
        "netIncomeGrowth",
        "netIncomeGrowthTTM",
        "medianProfitMargin",
        "earningTTM",
        "revenueTTM",
        "peRatio",
        "psRatio",
        "avgFlatAnnualDivIncrease",
        "numDividendYear",
        "positiveYear",
        "maximumCutPct",
        "max10CutPct",
        "numOfYear",
        "DScore",
        "return_7d",
        "return_1m",
        "return_1y",
        "return_10y",
        "total_return_1y",
        "total_return_10y",
    ]
    if exchange == "jkse":
        features.insert(0, "is_syariah")
    return frame[features]


def _load_company_profiles(exchange: str) -> tuple[pd.DataFrame, list[str]]:
    stocks = hd.get_all_idx_stocks() if exchange == "jkse" else hd.get_all_sp500_stocks()
    if stocks.empty or "symbol" not in stocks.columns:
        raise RuntimeError(f"No stock symbols found for {exchange}")
    stock_list = stocks["symbol"].dropna().astype(str).tolist()
    profiles = hd.get_company_profile(stock_list)
    if profiles.empty:
        raise RuntimeError(f"No company profiles found for {exchange}")
    if "symbol" in profiles.columns:
        profiles = profiles.set_index("symbol")
    profiles.index.name = "symbol"
    active = profiles[profiles["isActivelyTrading"].fillna(False)].index.astype(str).tolist()
    if not active:
        raise RuntimeError(f"No actively traded stocks found for {exchange}")
    return profiles, active


def _add_syariah_status(profiles: pd.DataFrame) -> pd.DataFrame:
    syariah_path = REPOSITORY_ROOT / "data" / "jkse" / "syariah.csv"
    syariah = pd.read_csv(syariah_path, sep=";")
    syariah["symbol"] = syariah["Kode"].astype(str) + ".JK"
    merged = profiles.reset_index().merge(syariah, on="symbol", how="left")
    merged["is_syariah"] = merged["Kode"].notna()
    return merged.set_index("symbol")


def run_daily(
    exchange: str = "jkse",
    mcap_filter: Optional[int] = None,
    dividend_years=None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    use_local_price_cache: bool = False,
) -> dict:
    """Run the complete daily data pipeline and return a serializable summary."""
    validate_environment()
    if exchange not in DEFAULT_MCAP_FILTERS:
        raise ValueError("exchange must be either 'jkse' or 'sp500'")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    selected_mcap_filter = (
        DEFAULT_MCAP_FILTERS[exchange] if mcap_filter is None else mcap_filter
    )
    years = list(dividend_years) if dividend_years is not None else list(range(2020, datetime.now().year + 1))
    if not years:
        raise ValueError("dividend_years must not be empty")

    started_at = datetime.now()
    logger.info("Starting daily pipeline for %s", exchange)
    profiles, active_stocks = _load_company_profiles(exchange)

    dividend_function = download_single_dividend if exchange == "jkse" else download_single_us_dividend
    dividends, dividend_summary = download_data(
        active_stocks, dividend_function, "dividend", max_concurrency
    )
    financials, financial_summary = download_data(
        active_stocks, download_single_financial, "financial", max_concurrency
    )

    historical_summary = run_historical_pipeline(
        exchange=exchange,
        mode="incremental",
        max_concurrency=max_concurrency,
        use_local_cache=use_local_price_cache,
    )

    supabase = get_supabase_client()
    refresh_warning = refresh_returns_view(supabase)
    if exchange == "jkse":
        profiles = _add_syariah_status(profiles)

    latest_returns = get_latest_returns_from_db(supabase)
    if latest_returns.empty:
        raise RuntimeError("The latest-returns view returned no rows")
    dividend_scores = compute_div_score(
        profiles,
        financials,
        dividends,
        exchange=exchange,
        latest_returns=latest_returns,
    )
    if dividend_scores.empty:
        raise RuntimeError(f"Dividend score calculation produced no rows for {exchange}")

    redis_frames = {f"div_score_{exchange}": dividend_scores.reset_index()}
    available_years = []
    for year in years:
        calendar = prepare_dividend_calendar(profiles, dividends, selected_mcap_filter, year)
        if calendar.empty:
            continue
        redis_frames[f"div_cal_{exchange}_{year}"] = calendar
        available_years.append(year)
    redis_frames[f"div_cal_years_{exchange}"] = pd.DataFrame(
        {"year": sorted(available_years)}, dtype="int64"
    )
    store_pickle_to_supabase_storage(
        f"data/{exchange}/dividends.pkl", dividends, client=supabase
    )
    store_pickle_to_supabase_storage(
        f"data/{exchange}/financials.pkl", financials, client=supabase
    )
    store_frames_to_redis(redis_frames)

    finished_at = datetime.now()
    summary = {
        "exchange": exchange,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "active_stocks": len(active_stocks),
        "downloads": {
            "dividends": dividend_summary,
            "financials": financial_summary,
        },
        "historical_prices": historical_summary,
        "returns_rows": len(latest_returns),
        "returns_refresh_warning": refresh_warning,
        "score_rows": len(dividend_scores),
        "score_non_null": int(dividend_scores["DScore"].notna().sum()),
        "calendar_years": sorted(available_years),
        "redis_keys": sorted(redis_frames),
    }
    logger.info("Daily pipeline summary: %s", json.dumps(summary, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Harvest daily data pipeline")
    parser.add_argument("--exchange", choices=sorted(DEFAULT_MCAP_FILTERS), required=True)
    parser.add_argument("--mcap-filter", type=int)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=datetime.now().year)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start_year > args.end_year:
        parser.error("--start-year must not be after --end-year")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be at least 1")

    try:
        run_daily(
            exchange=args.exchange,
            mcap_filter=args.mcap_filter,
            dividend_years=range(args.start_year, args.end_year + 1),
            max_concurrency=args.max_concurrency,
            use_local_price_cache=False,
        )
        return 0
    except Exception:
        logger.exception("Daily pipeline failed")
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    sys.exit(main())
