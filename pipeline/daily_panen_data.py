import json
from datetime import datetime

from prefect import flow
from prefect.artifacts import create_markdown_artifact

from pipeline.daily_data import run_daily as run_daily_core


@flow(name="Daily Panen Data", log_prints=True)
def run_daily(
    exch: str = "jkse",
    mcap_filter: int = 100_000_000_000,
    dividend_years=None,
):
    """Prefect compatibility wrapper around the plain daily pipeline."""
    summary = run_daily_core(
        exchange=exch,
        mcap_filter=mcap_filter,
        dividend_years=dividend_years,
        use_local_price_cache=True,
    )
    create_markdown_artifact(
        key=f"daily-panen-{exch}-summary",
        markdown=f"# Daily Panen {exch.upper()} Summary\n```json\n{json.dumps(summary, indent=2)}\n```",
        description=f"Daily pipeline summary for {exch}",
    )
    return summary


if __name__ == "__main__":
    run_daily(
        exch="jkse",
        mcap_filter=100_000_000_000,
        dividend_years=range(2020, datetime.now().year + 1),
    )
