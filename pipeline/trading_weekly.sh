#!/bin/sh
# Weekly setup search for every trading tool. One failed tool doesn't stop the others;
# the job still exits non-zero.

run() {
    limit=$1
    shift
    timeout --signal=TERM --kill-after=60s "$limit" python -m "$@"
}

status=0
run 1h pipeline.trading_optimize --tool dividend_play || status=1
run 3h pipeline.trading_optimize --tool swing || status=1
run 1h pipeline.trading_optimize --tool long_term || status=1
exit $status
