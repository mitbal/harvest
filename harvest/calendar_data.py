"""Dividend calendar data helpers shared by the Payout Calendar page and the share image route.

Nothing here imports Streamlit, so server.py can build the same calendar outside a session.
"""
import io
import json
import logging

import holidays
import numpy as np
import pandas as pd


logger = logging.getLogger('calendar')


def decode_frame(raw):
    """DataFrame from a Redis value: parquet bytes, a {'date', 'content'} JSON envelope, or plain JSON records."""
    if isinstance(raw, bytes) and raw.startswith(b'PAR1'):
        return pd.read_parquet(io.BytesIO(raw))

    payload = json.loads(raw)
    if isinstance(payload, dict) and 'date' in payload and 'content' in payload:
        return pd.DataFrame(json.loads(payload['content']))
    return pd.DataFrame(payload)


def market_holidays(market, years):
    """{date: holiday name} for days the exchange is closed: IDX (market 'ID') or NYSE (market 'US')."""
    if market == 'US':
        closed = dict(holidays.financial_holidays('NYSE', years=years))
    else:
        # IDX closes on public holidays, cuti bersama (government category) and the 31 Dec bourse holiday
        closed = dict(holidays.Indonesia(years=years, categories=('public', 'government'), language='en_US'))
        for y in years:
            closed.setdefault(pd.Timestamp(year=y, month=12, day=31).date(), 'IDX Year-End Holiday')
    return dict(sorted(closed.items()))


def project_to_next_year(dates, market='ID'):
    """Shift ex-dates one year forward so the calendar grid uses the projected year's weekdays.

    DateOffset maps 29 Feb onto 28 Feb instead of raising like Timestamp.replace.
    Ex-dates are trading days, so a projection landing on a weekend or exchange holiday
    moves to the next trading day, or back to the previous one if that would leave the month.
    """
    projected = pd.Series(pd.to_datetime(dates)) + pd.DateOffset(years=1)
    if projected.empty:
        return projected

    years = sorted(projected.dt.year.unique().tolist())
    closed = np.array(list(market_holidays(market, years)), dtype='datetime64[D]')
    days = projected.values.astype('datetime64[D]')

    forward = np.busday_offset(days, 0, roll='forward', holidays=closed)
    backward = np.busday_offset(days, 0, roll='backward', holidays=closed)

    rolled = pd.Series(pd.to_datetime(forward), index=projected.index)
    leaves_month = rolled.dt.month != projected.dt.month
    rolled[leaves_month] = pd.to_datetime(backward[leaves_month.to_numpy()])
    return rolled


def add_yield(df):
    """Add yield (%) and drop rows that are not a real payout or have no usable price.

    A zero or missing price gives an inf/NaN yield that skews the averages and breaks idxmax;
    a zero dividend (e.g. a non-payer with a 0.0 row) is not a payout.
    """
    df = df.copy()
    adj = pd.to_numeric(df['adjDividend'], errors='coerce')
    price = pd.to_numeric(df['price'], errors='coerce')
    valid = (adj > 0) & (price > 0)

    df['yield'] = (adj / price.where(valid)) * 100
    valid &= np.isfinite(df['yield'])

    dropped = int((~valid).sum())
    if dropped:
        logger.warning(f'Dropped {dropped} dividend rows without a positive dividend and price: '
                       f'{sorted(df.loc[~valid, "symbol"].astype(str).unique().tolist())}')
    return df[valid].reset_index(drop=True)


def syariah_symbols(div_score_df):
    """Set of Syariah-compliant symbols, or None when the score data does not carry the flag."""
    if not {'symbol', 'is_syariah'}.issubset(div_score_df.columns):
        return None
    return set(div_score_df.loc[div_score_df['is_syariah'].eq(True), 'symbol'])
