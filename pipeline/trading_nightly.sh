#!/bin/sh
# Nightly JKSE job: prices, then every trading scan, then the track record.
# A failed price update skips the rest, since the scans need fresh prices.
# One failed scan doesn't stop the others; the job still exits non-zero.

run() {
    limit=$1
    shift
    timeout --signal=TERM --kill-after=60s "$limit" python -m "$@"
}

run 6h pipeline.daily_data --exchange jkse || exit 1

status=0
run 1h pipeline.trading_scan --tool dividend_play || status=1
run 1h pipeline.trading_scan --tool swing || status=1
run 1h pipeline.trading_scan --tool long_term || status=1
run 1h pipeline.trading_track --tool all || status=1
exit $status
