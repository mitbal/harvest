"""Replaying published cards into realised outcomes (harvest/track_record.py)."""
import pandas as pd
import pytest

import harvest.dividend_play as dp
import harvest.track_record as tr


DAYS = pd.bdate_range("2026-03-02", periods=30)


def _bars(close, open_=None, low=None, high=None, days=DAYS) -> pd.DataFrame:
    close = pd.Series(close, dtype=float)
    days = days[:len(close)]
    frame = pd.DataFrame({
        "open": open_ if open_ is not None else close.to_numpy(),
        "high": high if high is not None else close.to_numpy(),
        "low": low if low is not None else close.to_numpy(),
        "close": close.to_numpy(),
    }, index=days)
    return frame


def _dp_signal(day, limit, exit_by, symbol="BBRI.JK", **params):
    return {
        "symbol": symbol, "as_of": DAYS[day].date(), "entry_price": limit, "stop_price": None, "target_price": None,
        "exit_by": DAYS[exit_by].date(), "key_date": DAYS[exit_by].date(), "key_date_estimated": False,
        "detail": {"setup": "Buy 30-5d before ex", "holdout": {"avg_return": 3.0},
                   "params": {"take_profit_pct": 0.0, "stop_loss_pct": 0.0, **params}},
    }


def _signals(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


COSTS = dp.trade_costs()


def _net(entry, exit_price, dividend=0.0):
    keep = 1 - COSTS["sell_fee_pct"] / 100
    return ((exit_price * keep + dividend * (1 - COSTS["dividend_tax_pct"] / 100))
            / (entry * (1 + COSTS["buy_fee_pct"] / 100)) - 1) * 100


# ── Dividend Play ────────────────────────────────────────────────────────── #

def test_dividend_play_fills_at_the_limit_and_sells_on_the_sell_by_close_with_the_dividend():
    close = [1000.0] * 12
    close[10] = 1080.0
    lows = list(close)
    lows[1] = 985.0                      # the session after the card dips through the 990 limit
    bars = {"BBRI.JK": _bars(close, open_=[1000.0] * 12, low=lows)}
    divs = {"BBRI.JK": pd.DataFrame({"date": [DAYS[8]], "dividend": [50.0]})}

    [trade] = tr.dividend_play_outcomes(_signals(_dp_signal(0, 991.0, 10)), DAYS[:12], bars, divs)

    assert trade["status"] == "closed"
    assert trade["entry_date"] == DAYS[1].date().isoformat() and trade["entry_price"] == 990.0
    assert trade["exit_date"] == DAYS[10].date().isoformat() and trade["exit_reason"] == tr.REASON_SELL_BY
    assert trade["return_pct"] == pytest.approx(_net(990.0, 1080.0, 50.0))
    assert trade["detail"]["dividend"] == 50.0 and trade["detail"]["expected"] == 3.0


def test_dividend_play_fills_at_the_open_when_it_gaps_below_the_limit():
    bars = {"BBRI.JK": _bars([1000.0] * 6, open_=[1000.0, 970.0, 1000, 1000, 1000, 1000], low=[1000, 960, 1000, 1000, 1000, 1000])}

    [trade] = tr.dividend_play_outcomes(_signals(_dp_signal(0, 990.0, 4)), DAYS[:6], bars, {})

    assert trade["entry_price"] == 970.0


def test_dividend_play_retries_with_the_next_card_and_dates_the_trade_from_the_first():
    lows = [1000.0, 1000.0, 975.0, 1000.0, 1000.0, 1000.0]
    bars = {"BBRI.JK": _bars([1000.0] * 6, low=lows)}
    signals = _signals(_dp_signal(0, 990.0, 5), _dp_signal(1, 980.0, 5))

    [trade] = tr.dividend_play_outcomes(signals, DAYS[:6], bars, {})

    assert trade["signal_date"] == DAYS[0].date().isoformat()
    assert trade["entry_date"] == DAYS[2].date().isoformat() and trade["entry_price"] == 980.0
    assert trade["detail"]["cards"] == 2


def test_dividend_play_counts_an_unfilled_streak_once_it_ends():
    bars = {"BBRI.JK": _bars([1000.0] * 6)}
    signals = _signals(_dp_signal(0, 990.0, 5), _dp_signal(1, 990.0, 5))

    [missed] = tr.dividend_play_outcomes(signals, DAYS[:6], bars, {})

    assert missed["status"] == "not_filled" and missed["detail"]["cards"] == 2
    assert missed["return_pct"] is None


def test_dividend_play_waits_while_the_streak_is_still_running_or_the_card_is_new():
    bars = {"BBRI.JK": _bars([1000.0] * 3)}
    # Day 1's card is the latest scan: its streak may continue tomorrow.
    assert tr.dividend_play_outcomes(_signals(_dp_signal(0, 990.0, 5), _dp_signal(1, 990.0, 5)), DAYS[:2], bars, {}) == []
    # Day 2's card was published after the latest close.
    assert tr.dividend_play_outcomes(_signals(_dp_signal(2, 1000.0, 5)), DAYS[:3], bars, {}) == []


def test_dividend_play_open_trade_is_marked_to_the_latest_close():
    bars = {"BBRI.JK": _bars([1000.0, 1000.0, 1010.0, 1020.0], low=[1000.0, 990.0, 1010.0, 1020.0])}

    [trade] = tr.dividend_play_outcomes(_signals(_dp_signal(0, 995.0, 10)), DAYS[:4], bars, {})

    assert trade["status"] == "open" and trade["exit_date"] is None and trade["exit_price"] is None
    assert trade["return_pct"] == pytest.approx(_net(995.0, 1020.0))
    assert trade["detail"]["marked_at"] == DAYS[3].date().isoformat()


def test_dividend_play_cards_during_a_held_trade_belong_to_it():
    lows = [1000.0] * 10
    lows[1] = 980.0
    bars = {"BBRI.JK": _bars([1000.0] * 10, low=lows)}
    signals = _signals(_dp_signal(0, 990.0, 5), _dp_signal(2, 990.0, 5), _dp_signal(4, 990.0, 5))

    outcomes = tr.dividend_play_outcomes(signals, DAYS[:10], bars, {})

    assert [o["status"] for o in outcomes] == ["closed"]


def test_dividend_play_take_profit_exits_early():
    close = [1000.0, 1000.0, 1020.0, 1060.0, 1000.0, 1000.0]
    bars = {"BBRI.JK": _bars(close, low=[1000.0, 990.0, 1020.0, 1060.0, 1000.0, 1000.0])}

    [trade] = tr.dividend_play_outcomes(_signals(_dp_signal(0, 995.0, 5, take_profit_pct=5.0)), DAYS[:6], bars, {})

    assert trade["exit_reason"] == "Take profit" and trade["exit_date"] == DAYS[3].date().isoformat()


# ── Swing ────────────────────────────────────────────────────────────────── #

def _swing_signal(day, symbol="BMRI.JK", stop=2.0, target=5.0, hold=3):
    return {"symbol": symbol, "as_of": DAYS[day].date(), "exit_by": DAYS[day + hold + 1].date(),
            "detail": {"stop_pct": stop, "target_pct": target, "max_hold": hold, "holdout": {"avg_return": 0.8}}}


def test_swing_buys_the_next_open_and_stops_out():
    bars = {"BMRI.JK": _bars([1000.0, 1000.0, 970.0, 970.0, 970.0], open_=[1000.0, 1000.0, 990.0, 970.0, 970.0],
                             low=[1000.0, 995.0, 960.0, 970.0, 970.0], high=[1000.0] * 5)}

    [trade] = tr.swing_outcomes(_signals(_swing_signal(0)), bars)

    assert trade["entry_date"] == DAYS[1].date().isoformat() and trade["entry_price"] == 1000.0
    assert trade["exit_reason"] == "Stop" and trade["exit_price"] == pytest.approx(980.0)
    assert trade["return_pct"] < -2.0 and trade["detail"]["sessions"] == 1


def test_swing_sells_after_the_maximum_hold_and_then_takes_the_next_card():
    bars = {"BMRI.JK": _bars([1000.0] * 12)}
    signals = _signals(_swing_signal(0), _swing_signal(2), _swing_signal(4))

    outcomes = tr.swing_outcomes(signals, bars)

    # Bought day 1, sold at day 4's close after 3 sessions; day 2's card falls inside it.
    assert [(o["signal_date"], o["exit_reason"]) for o in outcomes] == [
        (DAYS[0].date().isoformat(), "Max hold"), (DAYS[4].date().isoformat(), "Max hold"),
    ]


def test_swing_open_trade_and_new_card():
    bars = {"BMRI.JK": _bars([1000.0, 1000.0, 1010.0])}

    [trade] = tr.swing_outcomes(_signals(_swing_signal(0), _swing_signal(2)), bars)

    assert trade["status"] == "open" and trade["detail"]["last_close"] == 1010.0


def test_swing_open_from_the_latest_session():
    bars = {"BMRI.JK": _bars([1000.0, 1004.0], open_=[1000.0, 1001.0])}

    [trade] = tr.swing_outcomes(_signals(_swing_signal(0)), bars)

    assert trade["status"] == "open" and trade["entry_price"] == 1001.0


# ── Long Term ────────────────────────────────────────────────────────────── #

def _lt_signal(day, price, target, symbol="CMRY.JK"):
    return {"symbol": symbol, "as_of": DAYS[day].date(), "entry_price": price, "target_price": target}


def test_long_term_closes_when_the_target_is_reached_and_compares_with_the_jkse():
    bars = {"CMRY.JK": _bars([100.0, 105.0, 121.0, 130.0])}
    jkse = pd.Series([7000.0, 7100.0, 7140.0, 7200.0], index=DAYS[:4])
    signals = _signals(_lt_signal(0, 100.0, 120.0), _lt_signal(1, 105.0, 125.0))

    [pick] = tr.long_term_outcomes(signals, bars, jkse)

    assert pick["status"] == "closed" and pick["exit_date"] == DAYS[2].date().isoformat()
    assert pick["exit_reason"] == tr.REASON_TARGET_VALUE
    assert pick["benchmark_pct"] == pytest.approx(2.0)


def test_long_term_open_pick_is_marked_to_market():
    bars = {"CMRY.JK": _bars([100.0, 95.0, 90.0])}

    [pick] = tr.long_term_outcomes(_signals(_lt_signal(0, 100.0, 140.0)), bars)

    assert pick["status"] == "open" and pick["return_pct"] == pytest.approx(_net(100.0, 90.0))
    assert pick["benchmark_pct"] is None


# ── Summary ──────────────────────────────────────────────────────────────── #

def test_summarize_counts_only_closed_trades_in_the_averages():
    frame = pd.DataFrame([
        {"status": "closed", "return_pct": 4.0, "benchmark_pct": 1.0, "detail": {"expected": 3.0}},
        {"status": "closed", "return_pct": -2.0, "benchmark_pct": 0.0, "detail": {"expected": 3.0}},
        {"status": "open", "return_pct": 9.0, "benchmark_pct": None, "detail": {}},
        {"status": "not_filled", "return_pct": None, "benchmark_pct": None, "detail": {}},
    ])

    summary = tr.summarize(frame)

    assert summary == {"closed": 2, "wins": 1, "win_rate": 50.0, "avg_return": 1.0, "expected": 3.0,
                       "avg_benchmark": 0.5, "open": 1, "not_filled": 1}
    assert tr.summarize(frame.iloc[:0])["closed"] == 0
