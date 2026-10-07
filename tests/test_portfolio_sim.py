"""One-account replay of backtest trades (harvest/portfolio_sim.py)."""
import pandas as pd
import pytest

import harvest.portfolio_sim as ps


DAYS = pd.bdate_range("2025-01-01", periods=40)


def _bars(close=1000.0, volume=10_000_000, days=DAYS) -> pd.DataFrame:
    return pd.DataFrame({"close": close, "volume": volume}, index=days)


def _trade(symbol, entry, exit_, ret, price=1000.0):
    return {"symbol": symbol, "entry_date": DAYS[entry], "exit_date": DAYS[exit_], "entry_price": price,
            "return_pct": ret}


def _run(trades, bars, **kwargs):
    return ps.simulate_portfolio(pd.DataFrame(trades), bars, **kwargs)


def test_one_trade_uses_one_slot_in_whole_lots_and_books_its_return():
    result = _run([_trade("A", 0, 5, 10.0, price=1010.0)], {"A": _bars()}, capital=10_000_000, max_positions=2)

    # Half the account is Rp 5,000,000: 49 lots of Rp 101,000.
    cost = 49 * 101_000
    assert result["stats"]["final_equity"] == pytest.approx(10_000_000 + cost * 0.10)
    assert result["stats"]["trades_taken"] == 1 and result["stats"]["wins"] == 1


def test_full_slots_and_held_stocks_skip_trades():
    trades = [_trade("A", 0, 10, 1.0), _trade("B", 0, 10, 1.0), _trade("C", 1, 5, 1.0), _trade("A", 2, 6, 1.0)]
    bars = {s: _bars() for s in "ABC"}

    stats = _run(trades, bars, capital=10_000_000, max_positions=2)["stats"]

    assert stats["trades_taken"] == 2
    assert stats["skipped"] == {"held": 1, "full": 1, "too_small": 0}


def test_a_freed_slot_is_used_from_the_next_session():
    trades = [_trade("A", 0, 3, 0.0), _trade("B", 3, 6, 0.0), _trade("C", 4, 8, 0.0)]
    bars = {s: _bars() for s in "ABC"}

    stats = _run(trades, bars, capital=10_000_000, max_positions=1)["stats"]

    # B starts the day A exits (entries come first), so it is skipped; C takes the slot.
    assert stats["trades_taken"] == 2 and stats["skipped"]["full"] == 1


def test_liquidity_caps_the_position_and_tiny_stocks_are_skipped():
    bars = {"A": _bars(volume=2_000), "B": _bars(volume=100)}
    trades = [_trade("A", 25, 30, 5.0), _trade("B", 25, 30, 5.0)]

    stats = _run(trades, bars, capital=100_000_000, max_positions=2, liquidity_pct=5.0)["stats"]

    # A trades Rp 2,000,000 a day: 5% buys 1 lot. B's 5% is Rp 5,000, less than a lot.
    assert stats["liquidity_capped"] == 2 and stats["trades_taken"] == 1
    assert stats["skipped"]["too_small"] == 1
    assert stats["final_equity"] == pytest.approx(100_000_000 + 100_000 * 0.05)


def test_positions_are_marked_at_the_close_and_drawdown_measured():
    closes = pd.Series(1000.0, index=DAYS)
    closes.iloc[3:6] = 800.0
    bars = {"A": pd.DataFrame({"close": closes, "volume": 10_000_000})}

    result = _run([_trade("A", 0, 8, 0.0)], bars, capital=1_000_000, max_positions=1)

    assert result["stats"]["max_drawdown"] == pytest.approx(-20.0)
    # Invested every session but the exit day.
    assert result["stats"]["avg_invested_pct"] == pytest.approx(8 / 9 * 100)


def test_benchmark_and_weekly_curve_are_indexed_to_100():
    jkse = pd.Series([7000.0 + 10 * i for i in range(len(DAYS))], index=DAYS)

    result = _run([_trade("A", 0, 30, 4.0)], {"A": _bars()}, capital=10_000_000, max_positions=1, benchmark=jkse)

    assert result["curve"][0]["account"] == pytest.approx(100.0)
    assert result["curve"][0]["jkse"] == pytest.approx(100.0 + 0.0, abs=1.0)
    assert result["stats"]["benchmark"]["total_return"] == pytest.approx((7300 / 7000 - 1) * 100, abs=0.2)
    assert all(set(point) == {"date", "account", "jkse"} for point in result["curve"])


def test_no_trades():
    result = _run([], {}, capital=1_000_000)

    assert result["stats"] is None and result["curve"] == []
    assert result["settings"]["max_positions"] == ps.DEFAULT_MAX_POSITIONS
