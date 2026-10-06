"""Small Streamlit pieces shared by the trading pages' Today and Evidence views."""
import datetime
import html
import re
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


def status_line(run: dict, stale: bool = False) -> None:
    """'Updated <day>, <time> WIB after close · Next update <day> after close', as a live pill."""
    as_of = pd.Timestamp(run['as_of'])
    finished = pd.Timestamp(run.get('finished_at') or run['started_at']).tz_convert(WIB)
    text = (f"Updated {day(as_of)}, {finished:%H:%M} WIB after close · "
            f"Next update {day(next_session(as_of))} after close")
    if stale:
        text += ' · Missed at least one nightly update, so this may be out of date'
    render(f'<span class="hv-status{" stale" if stale else ""}"><span class="dot"></span>{esc(text)}</span>')


def unlock_prompt(hidden: list[str], noun: str = 'setup', lead: str = 'Also today') -> None:
    """Dark call-to-action card naming what is locked, with the unlock button inside."""
    count = len(hidden)
    with st.container(key='hv-unlock'):
        render(
            f'<div class="hv-unlock"><div class="lock">{icon("lock")}</div><div>'
            f'<h4>{count} more {noun if count == 1 else noun + "s"} today</h4>'
            '<p>Supporters see every card, the evidence for any stock and the research lab.</p>'
            f'<div class="blur"><span>{esc(lead)}:</span>'
            + ''.join(f'<span>{esc(short(s))}</span>' for s in hidden) + '</div></div></div>'
        )
        st.button(
            'Unlock the trading lab', type='primary', icon=':material/lock_open:',
            on_click=open_view, args=(VIEW_LAB,), key='today_unlock',
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN KIT
# Plain HTML pieces rendered with st.html. Every dynamic string goes through esc().
# Cards are st.container(key='hv-card-...') so native buttons sit inside them.
# ═══════════════════════════════════════════════════════════════════════════════

STYLES = """
<style>
:root {
  --hv-ink: #10231c; --hv-muted: #5c6d66; --hv-faint: #8a9a93; --hv-line: #dbe3df;
  --hv-card: #ffffff; --hv-soft: #f5f8f6; --hv-brand: #064e3b; --hv-teal: #0f766e;
  --hv-mint: #e3f1eb; --hv-up: #15803d; --hv-up-soft: #dcf2e3; --hv-down: #c2410c;
  --hv-down-soft: #fde8dc; --hv-amber: #b45309; --hv-amber-soft: #fdf0d5;
}
.hv { color: var(--hv-ink); font-variant-numeric: tabular-nums; line-height: 1.45; container-type: inline-size; }
.hv b { font-weight: 650; }
.hv .icon { font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal;
  font-size: 1.1em; line-height: 1; vertical-align: -0.18em; display: inline-block; }

/* Hero */
.hv-hero { position: relative; overflow: hidden; border-radius: 18px; padding: 1.6rem 1.8rem 1.5rem;
  color: #ecfdf5; background:
    radial-gradient(120% 140% at 100% 0%, rgba(52, 211, 153, .28) 0%, transparent 55%),
    linear-gradient(135deg, #064e3b 0%, #0b5f4a 55%, #0f766e 100%); }
.hv-hero::after { content: ''; position: absolute; inset: 0; pointer-events: none; opacity: .14;
  background-image: linear-gradient(rgba(255,255,255,.5) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.5) 1px, transparent 1px);
  background-size: 28px 28px; mask-image: linear-gradient(90deg, transparent 35%, #000 100%);
  -webkit-mask-image: linear-gradient(90deg, transparent 35%, #000 100%); }
.hv-hero .eyebrow { font-size: .74rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
  color: #a7f3d0; display: flex; gap: .4rem; align-items: center; }
.hv-hero h1 { color: #fff; font-size: 2.15rem; font-weight: 750; letter-spacing: -.02em;
  margin: .3rem 0 .35rem; padding: 0; line-height: 1.15; }
.hv-hero p { margin: 0; max-width: 62ch; color: #d1fae5; font-size: .98rem; }

/* Status pill */
.hv-status { display: inline-flex; align-items: center; gap: .55rem; padding: .32rem .8rem .32rem .65rem;
  border-radius: 999px; background: var(--hv-card); border: 1px solid var(--hv-line);
  font-size: .82rem; color: var(--hv-muted); }
.hv-status .dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e;
  box-shadow: 0 0 0 0 rgba(34,197,94,.55); animation: hv-pulse 2.4s infinite; }
.hv-status.stale .dot { background: #f59e0b; animation: none; }
.hv-status.stale { border-color: #f5d28a; background: var(--hv-amber-soft); color: #7c4a03; }
@keyframes hv-pulse { 70% { box-shadow: 0 0 0 7px rgba(34,197,94,0); } 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); } }
@media (prefers-reduced-motion: reduce) { .hv-status .dot { animation: none; } }

/* Section header */
.hv-section { display: flex; flex-wrap: wrap; align-items: baseline; gap: .2rem .6rem; margin: .9rem 0 .1rem; }
.hv-section h3 { white-space: nowrap; font-size: 1.3rem; font-weight: 700; margin: 0; padding: 0; color: var(--hv-ink); }
.hv-section .count { font-size: .78rem; font-weight: 700; color: var(--hv-teal); background: var(--hv-mint);
  padding: .1rem .55rem; border-radius: 999px; }
.hv-section .note { font-size: .85rem; color: var(--hv-muted); }

/* Metric tiles (native st.metric) */
[data-testid="stMetric"] { background: var(--hv-card); border: 1px solid var(--hv-line); border-radius: 14px;
  padding: .8rem 1rem .75rem; box-shadow: 0 1px 2px rgba(16,35,28,.04); position: relative; overflow: hidden; }
[data-testid="stMetric"]::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: linear-gradient(#10b981, #0f766e); }
[data-testid="stMetricLabel"] { color: var(--hv-muted); }
[data-testid="stMetricValue"] { color: var(--hv-ink); font-weight: 700; }

/* Cards */
[class*="st-key-hv-card"] { background: var(--hv-card); border-radius: 16px !important;
  border: 1px solid var(--hv-line) !important; box-shadow: 0 1px 2px rgba(16,35,28,.05), 0 8px 24px -12px rgba(16,35,28,.14);
  padding: 1.1rem 1.15rem 1rem !important; gap: .7rem; transition: box-shadow .2s, transform .2s; }
[class*="st-key-hv-card"]:hover { box-shadow: 0 1px 2px rgba(16,35,28,.05), 0 14px 32px -14px rgba(16,35,28,.25); }
[class*="st-key-hv-card-urgent"] { border-color: #f5c08a !important; }
.hv-head { display: flex; align-items: flex-start; justify-content: space-between; gap: .8rem; }
.hv-ticker { display: flex; align-items: center; gap: .7rem; min-width: 0; }
.hv-avatar { flex: none; width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center;
  font-size: 1.35rem; color: var(--hv-brand);
  background: linear-gradient(140deg, #d1fae5, #a7f3d0); }
.hv-symbol { font-size: 1.22rem; font-weight: 800; letter-spacing: .01em; line-height: 1.1; }
.hv-company { font-size: .85rem; color: var(--hv-muted); }
.hv-chips { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .55rem; }
.hv-chip { display: inline-flex; align-items: center; gap: .25rem; font-size: .76rem; font-weight: 600;
  padding: .16rem .55rem; border-radius: 999px; background: var(--hv-soft); color: var(--hv-muted);
  border: 1px solid var(--hv-line); white-space: nowrap; }
.hv-chip.up { background: var(--hv-up-soft); color: var(--hv-up); border-color: transparent; }
.hv-chip.down { background: var(--hv-down-soft); color: var(--hv-down); border-color: transparent; }
.hv-chip.warn { background: var(--hv-amber-soft); color: var(--hv-amber); border-color: transparent; }
.hv-chip.brand { background: var(--hv-mint); color: var(--hv-teal); border-color: transparent; }
.hv-chip.lg { font-size: .85rem; padding: .25rem .7rem; }
.hv-rank { font-size: .74rem; font-weight: 800; color: #fff; background: var(--hv-brand); border-radius: 8px; margin-right: .4rem;
  padding: .12rem .45rem; vertical-align: .15em; }

.hv-cells { display: grid; grid-template-columns: repeat(var(--n, 4), minmax(0, 1fr)); gap: .45rem; margin-top: .85rem; }
@container (max-width: 430px) { .hv-cells { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.hv-cell { background: var(--hv-soft); border-radius: 11px; padding: .5rem .6rem; min-width: 0; }
.hv-cell .k { font-size: .7rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--hv-faint); }
.hv-cell .v { font-size: 1.05rem; font-weight: 750; margin-top: .1rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hv-cell .s { font-size: .78rem; color: var(--hv-muted); }
.hv-cell.up .v { color: var(--hv-up); } .hv-cell.down .v { color: var(--hv-down); }
.hv-cell.up { background: linear-gradient(180deg, #f0faf3, var(--hv-soft)); }
.hv-cell.down { background: linear-gradient(180deg, #fff4ed, var(--hv-soft)); }

/* Price ladder */
.hv-ladder { position: relative; height: 62px; margin: 1rem .4rem .2rem; }
.hv-ladder .track { position: absolute; left: 0; right: 0; top: 27px; height: 8px; border-radius: 99px; background: #e8eeeb; }
.hv-ladder .seg { position: absolute; top: 27px; height: 8px; }
.hv-ladder .seg.down { background: linear-gradient(90deg, #fb923c, #fed7aa); border-radius: 99px 0 0 99px; }
.hv-ladder .seg.up { background: linear-gradient(90deg, #bbf7d0, #22c55e); border-radius: 0 99px 99px 0; }
.hv-ladder .mark { position: absolute; top: 22px; width: 3px; height: 18px; margin-left: -1.5px; border-radius: 2px; background: var(--hv-ink); }
.hv-ladder .mark.entry { width: 14px; height: 14px; top: 24px; margin-left: -7px; border-radius: 50%;
  background: #fff; border: 3px solid var(--hv-brand); }
.hv-ladder .lbl { position: absolute; transform: translateX(-50%); font-size: .72rem; color: var(--hv-muted); white-space: nowrap; }
.hv-ladder .lbl b { color: var(--hv-ink); }
.hv-ladder .lbl.top { top: 0; } .hv-ladder .lbl.bot { top: 44px; }
.hv-ladder .lbl.left { transform: none; } .hv-ladder .lbl.right { transform: translateX(-100%); }

/* Progress */
.hv-progress { margin-top: .85rem; }
.hv-progress .bar { height: 8px; border-radius: 99px; background: #e8eeeb; overflow: hidden; }
.hv-progress .fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #34d399, #0f766e); }
.hv-progress.warn .fill { background: linear-gradient(90deg, #fbbf24, #d97706); }
.hv-progress.down .fill { background: linear-gradient(90deg, #fb923c, #c2410c); }
.hv-progress .legend { display: flex; justify-content: space-between; gap: .5rem; font-size: .78rem; color: var(--hv-muted); margin-top: .3rem; }

/* Callout strip and footnote */
.hv-strip { display: flex; align-items: center; gap: .55rem; margin-top: .85rem; padding: .55rem .75rem;
  border-radius: 11px; background: var(--hv-mint); color: #0b4d3c; font-size: .9rem; }
.hv-strip.plain { background: var(--hv-soft); color: var(--hv-muted); }
.hv-strip.warn { background: var(--hv-amber-soft); color: #7c4a03; }
.hv-foot { margin-top: .75rem; padding-top: .65rem; border-top: 1px dashed var(--hv-line); font-size: .8rem; color: var(--hv-muted); }
.hv-foot + .hv-foot { border-top: 0; margin-top: .25rem; padding-top: 0; }
.hv-winbar { display: inline-flex; width: 64px; height: 6px; border-radius: 99px; overflow: hidden;
  background: #fed7aa; vertical-align: middle; margin-right: .45rem; }
.hv-winbar span { background: #22c55e; }

/* Score ring */
.hv-ring { flex: none; width: 52px; height: 52px; border-radius: 50%; display: grid; place-items: center;
  background: conic-gradient(var(--hv-teal) calc(var(--p) * 1%), #e5ece9 0); }
.hv-ring span { width: 40px; height: 40px; border-radius: 50%; background: #fff; display: grid; place-items: center;
  font-weight: 800; font-size: .95rem; color: var(--hv-brand); }

/* Valuation meters */
.hv-meters { display: grid; gap: .55rem; margin-top: .85rem; }
.hv-meter .row { display: flex; justify-content: space-between; font-size: .82rem; color: var(--hv-muted); }
.hv-meter .row b { color: var(--hv-ink); }
.hv-meter .bar { position: relative; height: 8px; border-radius: 99px; background: #e8eeeb; margin-top: .25rem; }
.hv-meter .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px; background: linear-gradient(90deg, #6ee7b7, #0f766e); }
.hv-meter .median { position: absolute; top: -3px; width: 2px; height: 14px; background: var(--hv-ink); border-radius: 2px; }

/* Watch chips, unlock and empty states */
.hv-tickers { display: flex; flex-wrap: wrap; gap: .45rem; }
.hv-tickers .t { font-weight: 750; font-size: .85rem; padding: .32rem .7rem; border-radius: 10px;
  background: var(--hv-card); border: 1px solid var(--hv-line); }
.hv-tickers .t small { font-weight: 500; color: var(--hv-muted); margin-left: .3rem; }
[class*="st-key-hv-unlock"] { border-radius: 16px; padding: 1.2rem 1.3rem !important; color: #ecfdf5;
  background: radial-gradient(100% 160% at 0% 0%, rgba(52,211,153,.25), transparent 60%),
              linear-gradient(120deg, #052e25, #064e3b 60%, #0f766e); }
[class*="st-key-hv-unlock"] .stButton { padding-left: calc(46px + 1rem); }
[class*="st-key-hv-unlock"] [data-testid="stBaseButton-primary"] { background: #6ee7b7; border-color: #6ee7b7; color: #052e25; font-weight: 700; }
[class*="st-key-hv-unlock"] [data-testid="stBaseButton-primary"]:hover { background: #a7f3d0; border-color: #a7f3d0; color: #052e25; }
.hv-unlock { color: #ecfdf5; display: flex; gap: 1rem; align-items: center; }
.hv-unlock .lock { flex: none; width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center;
  background: rgba(255,255,255,.12); font-size: 1.5rem; color: #6ee7b7; }
.hv-unlock h4 { margin: 0; padding: 0; color: #fff; font-size: 1.08rem; font-weight: 700; }
.hv-unlock p { margin: .15rem 0 0; color: #a7f3d0; font-size: .88rem; }
.hv-unlock .blur { display: flex; gap: .4rem; flex-wrap: wrap; margin-top: .55rem; }
.hv-unlock .blur span { font-size: .78rem; font-weight: 700; padding: .15rem .55rem; border-radius: 8px;
  background: rgba(255,255,255,.1); color: #d1fae5; }
.hv-empty { display: flex; gap: .9rem; align-items: flex-start; padding: 1rem 1.1rem; border-radius: 14px;
  background: var(--hv-card); border: 1px dashed #c6d3cd; }
.hv-empty .icon { font-size: 1.6rem; color: var(--hv-teal); }
.hv-empty h4 { margin: 0; padding: 0; font-size: 1rem; font-weight: 700; color: var(--hv-ink); }
.hv-empty p { margin: .2rem 0 0; font-size: .88rem; color: var(--hv-muted); }

.hv-setup { display: flex; gap: .8rem; align-items: flex-start; padding: .85rem 1rem; border-radius: 14px;
  background: var(--hv-card); border: 1px solid var(--hv-line); border-left: 4px solid var(--hv-teal); }
.hv-setup > .icon { font-size: 1.5rem; color: var(--hv-teal); margin-top: .15rem; }
.hv-setup .k { font-size: .7rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--hv-faint); }
.hv-setup .t { font-size: 1.02rem; font-weight: 700; margin: .1rem 0; }
.hv-setup .n { font-size: .85rem; color: var(--hv-muted); }

@media (max-width: 640px) {
  .hv-hero { padding: 1.2rem 1.1rem; border-radius: 14px; } .hv-hero h1 { font-size: 1.6rem; }
  .hv-unlock { align-items: flex-start; }
  [class*="st-key-hv-unlock"] .stButton { padding-left: 0; }
  .hv-ladder .lbl { font-size: .66rem; }
}
</style>
"""


def esc(value) -> str:
    return html.escape('' if value is None else str(value))


def bold(markdown: str) -> str:
    """Escape a one-line Markdown string, keeping its **bold** runs."""
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', esc(markdown))


def inject_styles() -> None:
    st.html(STYLES)


def render(markup: str) -> None:
    st.html(f'<div class="hv">{markup}</div>')


def icon(name: str) -> str:
    return f'<span class="icon" aria-hidden="true">{name}</span>'


def hero(eyebrow: str, title: str, subtitle: str, icon_name: str = 'monitoring') -> None:
    render(f'<div class="hv-hero"><div class="eyebrow">{icon(icon_name)}{esc(eyebrow)}</div>'
           f'<h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>')


def section(title: str, count: int | None = None, note: str | None = None) -> None:
    badge = f'<span class="count">{count}</span>' if count is not None else ''
    extra = f'<span class="note">{esc(note)}</span>' if note else ''
    render(f'<div class="hv-section"><h3>{esc(title)}</h3>{badge}{extra}</div>')


def chip(text: str, tone: str = '', icon_name: str | None = None) -> str:
    return f'<span class="hv-chip {tone}">{icon(icon_name) if icon_name else ""}{esc(text)}</span>'


def card_head(symbol: str, company: str | None, right: str = '', rank: int | None = None,
              icon_name: str = 'candlestick_chart') -> str:
    code = short(symbol)
    rank_html = f'<span class="hv-rank">#{rank}</span>' if rank is not None else ''
    return (
        f'<div class="hv-head"><div class="hv-ticker"><div class="hv-avatar">{icon(icon_name)}</div>'
        f'<div style="min-width:0"><div class="hv-symbol">{rank_html}{esc(code)}</div>'
        f'<div class="hv-company">{esc(company or symbol)}</div></div></div>{right}</div>'
    )


def cells(items: list[tuple]) -> str:
    """(label, value, sub[, tone]) tiles; value and sub are trusted HTML from this module's callers."""
    out = []
    for item in items:
        label, value, sub, tone = (*item, '')[:4]
        out.append(f'<div class="hv-cell {tone}"><div class="k">{esc(label)}</div><div class="v">{value}</div>'
                   + (f'<div class="s">{sub}</div>' if sub else '') + '</div>')
    return f'<div class="hv-cells" style="--n:{len(out)}">{"".join(out)}</div>'


def ladder(entry: float, stop: float | None, target: float | None) -> str:
    """Stop, entry and target on one line, with the loss and gain zones shaded."""
    if stop is None and target is None:
        return ''
    low, high = (stop if stop is not None else entry), (target if target is not None else entry)
    pad = max(high - low, entry * 0.01) * 0.3
    lo, hi = low - pad, high + pad
    at = lambda price: (price - lo) / (hi - lo) * 100
    parts = ['<div class="hv-ladder"><div class="track"></div>']
    if stop is not None:
        parts.append(f'<div class="seg down" style="left:{at(stop):.1f}%;width:{at(entry) - at(stop):.1f}%"></div>'
                     f'<div class="mark" style="left:{at(stop):.1f}%"></div>'
                     f'<div class="lbl bot" style="left:{at(stop):.1f}%">Stop <b>{esc(idr(stop))}</b></div>')
    if target is not None:
        parts.append(f'<div class="seg up" style="left:{at(entry):.1f}%;width:{at(target) - at(entry):.1f}%"></div>'
                     f'<div class="mark" style="left:{at(target):.1f}%"></div>'
                     f'<div class="lbl bot" style="left:{at(target):.1f}%">Target <b>{esc(idr(target))}</b></div>')
    parts.append(f'<div class="mark entry" style="left:{at(entry):.1f}%"></div>'
                 f'<div class="lbl top" style="left:{at(entry):.1f}%">Close <b>{esc(idr(entry))}</b></div></div>')
    return ''.join(parts)


def progress(fraction: float, left: str, right: str = '', tone: str = '') -> str:
    width = min(max(fraction, 0.0), 1.0) * 100
    return (f'<div class="hv-progress {tone}"><div class="bar"><div class="fill" style="width:{width:.1f}%"></div></div>'
            f'<div class="legend"><span>{left}</span><span>{right}</span></div></div>')


def winbar(wins: int | None, trades: int | None) -> str:
    if not trades or wins is None:
        return ''
    return f'<span class="hv-winbar" title="{wins} of {trades} won"><span style="width:{wins / trades * 100:.0f}%"></span></span>'


def strip(markup: str, tone: str = '', icon_name: str | None = None) -> str:
    return f'<div class="hv-strip {tone}">{icon(icon_name) if icon_name else ""}<span>{markup}</span></div>'


def foot(markup: str) -> str:
    return f'<div class="hv-foot">{markup}</div>'


def tickers(symbols, notes: dict | None = None) -> None:
    notes = notes or {}
    render('<div class="hv-tickers">' + ''.join(
        f'<span class="t">{esc(short(s))}' + (f'<small>{esc(notes[s])}</small>' if s in notes else '') + '</span>'
        for s in symbols) + '</div>')


def empty_state(title: str, body: str = '', icon_name: str = 'search_off') -> None:
    render(f'<div class="hv-empty">{icon(icon_name)}<div><h4>{esc(title)}</h4>'
           + (f'<p>{esc(body)}</p>' if body else '') + '</div></div>')


def card(key: str, urgent: bool = False):
    """A styled card container; native widgets placed inside it stay interactive."""
    return st.container(key=f"hv-card{'-urgent' if urgent else ''}-{key}")


def setup_banner(title: str, note: str, eyebrow: str = 'Published setup') -> None:
    """The setup the Evidence view backtests, as a headline above the chart."""
    render(f'<div class="hv-setup">{icon("tune")}<div><div class="k">{esc(eyebrow)}</div>'
           f'<div class="t">{esc(title)}</div><div class="n">{esc(note)}</div></div></div>')
