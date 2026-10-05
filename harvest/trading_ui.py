"""Small Streamlit pieces shared by the trading pages' Today and Evidence views."""
import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

import harvest.swing as sw
import harvest.trading_store as store


WIB = ZoneInfo('Asia/Jakarta')
VIEW_TODAY, VIEW_EVIDENCE, VIEW_LAB = 'Today', 'Evidence', 'Lab'


def short(symbol: str) -> str:
    return symbol.removesuffix('.JK')


def day(value, estimated: bool = False) -> str:
    """'Mon 3 Nov', with '~' in front of estimated dates."""
    date = pd.Timestamp(value)
    return f"{'~' if estimated else ''}{date:%a} {date.day} {date:%b}"


def idr(value) -> str:
    return f'Rp {value:,.0f}'


def tick_floor(price: float) -> float:
    """Round an 'at or below' price down to a valid IDX tick."""
    tick = float(sw.idx_tick_size([price])[0])
    return float(np.floor(price / tick) * tick)


def next_session(value) -> datetime.date:
    """The first IDX trading day after ``value``."""
    start = pd.Timestamp(value) + pd.Timedelta(days=1)
    return store.idx_trading_days(start, start + pd.Timedelta(days=14))[0]


def first_session(value) -> datetime.date:
    """The first IDX trading day on or after ``value``."""
    start = pd.Timestamp(value)
    return store.idx_trading_days(start, start + pd.Timedelta(days=14))[0]


def open_view(target: str, stock: str | None = None) -> None:
    """Button callback: switch the page's view (and the Evidence stock)."""
    st.session_state['view'] = target
    if stock:
        st.session_state['stock'] = stock


def status_line(run: dict) -> None:
    """'Updated <day>, <time> WIB after close. Next update <day> after close.'"""
    as_of = pd.Timestamp(run['as_of'])
    finished = pd.Timestamp(run.get('finished_at') or run['started_at']).tz_convert(WIB)
    st.caption(
        f"Updated {day(as_of)}, {finished:%H:%M} WIB after close. "
        f"Next update {day(next_session(as_of))} after close."
    )


def unlock_prompt(hidden: int) -> None:
    with st.container(border=True):
        st.markdown(
            f"**{hidden} more {'setup' if hidden == 1 else 'setups'} today.** Supporters see every card, "
            "the evidence for any stock and the research lab."
        )
        st.button(
            'Unlock the trading lab', type='primary', icon=':material/lock_open:',
            on_click=open_view, args=(VIEW_LAB,), key='today_unlock',
        )
