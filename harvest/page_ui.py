"""Shared look for the custom-styled pages (Payout Calendar month view, Compounding Simulator).

Plain HTML pieces rendered with ``st.html``. Every dynamic string goes through ``esc()``.
Cards are ``st.container(key='pdx-card-...')`` so native widgets and charts can sit inside them.
Each page calls ``inject_styles()`` itself, so pages that do not opt in look unchanged.
"""
import html

import streamlit as st

from harvest import chart_style as cs


TOKENS = {
    'canvas': cs.CHART_SURFACE,  # '#EFF1EF'
    'ink': '#10251d',
    'ink_soft': '#3c5148',
    'muted': '#5c6d66',
    'line': '#dbe3df',
    'soft': '#f5f8f6',
    'forest': '#073f2e',
    'deep': '#052d22',
    'lime': '#d7f35a',
    'mint': '#e3f1eb',
    'up': '#15803d',
    'down': '#c2410c',
    'amber': '#b45309',
}

CHIP_TONES = ('', 'up', 'down', 'warn', 'brand', 'lime')
NOTE_TONES = ('', 'up', 'down', 'warn', 'muted')

FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');"
)

_T = TOKENS

STYLES = f"""
<style>
{FONT_IMPORT}

:root {{
  --pdx-canvas: {_T['canvas']}; --pdx-ink: {_T['ink']}; --pdx-ink-soft: {_T['ink_soft']};
  --pdx-muted: {_T['muted']}; --pdx-line: {_T['line']}; --pdx-soft: {_T['soft']};
  --pdx-forest: {_T['forest']}; --pdx-deep: {_T['deep']}; --pdx-lime: {_T['lime']};
  --pdx-mint: {_T['mint']}; --pdx-up: {_T['up']}; --pdx-down: {_T['down']}; --pdx-amber: {_T['amber']};
}}

/* Page font. Spans are left out on purpose so Material icon glyphs keep their own font. */
[data-testid="stMainBlockContainer"],
[data-testid="stMainBlockContainer"] :is(h1, h2, h3, h4, p, li, label, input, textarea, button, small, b, td, th) {{
  font-family: 'Manrope', system-ui, sans-serif;
}}

.pdx {{ color: var(--pdx-ink); font-variant-numeric: tabular-nums; line-height: 1.45; }}
.pdx *, .pdx *::before, .pdx *::after {{ box-sizing: border-box; }}
.pdx h2, .pdx h3, .pdx p {{ margin: 0; padding: 0; }}
.pdx .pdx-h2 {{ font-size: 16px; font-weight: 800; letter-spacing: -.01em; line-height: 1.3; color: var(--pdx-ink); }}
.pdx .pdx-sub {{ color: var(--pdx-muted); font-size: 13.5px; margin-top: -.6rem; }}
.pdx .pdx-note {{ font-size: 12.5px; color: var(--pdx-muted); }}
.pdx .pdx-hd {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px 12px; }}
.pdx .pdx-legend {{ display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12.5px; font-weight: 700; color: var(--pdx-ink-soft); }}
.pdx .pdx-legend > span {{ display: inline-flex; align-items: center; gap: 6px; }}
.pdx .pdx-swatch {{ width: 11px; height: 11px; border-radius: 3px; display: inline-block; flex: none; }}

/* Tone text */
.pdx .pdx-tone-up {{ color: var(--pdx-up); }}
.pdx .pdx-tone-down {{ color: var(--pdx-down); }}
.pdx .pdx-tone-warn {{ color: var(--pdx-amber); }}
.pdx .pdx-tone-muted {{ color: var(--pdx-muted); }}

/* Cards: keyed Streamlit containers, so native widgets sit inside them */
[class*="st-key-pdx-card-"] {{
  background: #fff; border: 1px solid var(--pdx-line); border-radius: 16px;
  padding: 18px 20px; min-width: 0;
}}

/* Two-pane split (see split()): the pair wraps, assumptions on top, whenever the main block
   cannot fit both bases. Streamlit wraps each child container of a horizontal container in a
   layout wrapper that carries its flex; :has() reaches that wrapper by the child's key class. */
[class*="st-key-pdx-split-"] > :has(> [class*="st-key-pdx-pane-left-"]) {{
  flex: 1 1 300px; width: auto; min-width: 0; max-width: 100%;
}}
[class*="st-key-pdx-split-"] > :has(> [class*="st-key-pdx-pane-right-"]) {{
  flex: 3 1 560px; width: auto; min-width: 0; max-width: 100%;
}}

/* Metric tiles */
.pdx-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 12px; }}
.pdx-kpi {{
  background: #fff; border: 1px solid var(--pdx-line); border-radius: 14px; padding: 13px 16px;
  display: flex; flex-direction: column; gap: 2px; min-width: 0;
}}
.pdx-kpi .pdx-kpi-label {{ font-size: 12px; font-weight: 700; color: var(--pdx-muted); }}
.pdx-kpi .pdx-kpi-value {{
  font-size: 20px; font-weight: 800; letter-spacing: -.02em; line-height: 1.2; color: var(--pdx-ink);
  overflow-wrap: break-word;
}}
.pdx-kpi .pdx-kpi-note {{ font-size: 12px; font-weight: 700; }}

/* Chips */
.pdx-chip {{
  display: inline-flex; align-items: center; gap: 5px; font-size: 12px; font-weight: 700;
  padding: 3px 10px; border-radius: 99px; background: #f1f5f3; color: var(--pdx-ink-soft);
  border: 1px solid var(--pdx-line); white-space: nowrap;
}}
.pdx-chip-up {{ background: #dcf2e3; color: #166534; border-color: transparent; }}
.pdx-chip-down {{ background: #fde8dc; color: #9a3412; border-color: transparent; }}
.pdx-chip-warn {{ background: #fdf0d5; color: #8a4207; border-color: transparent; }}
.pdx-chip-brand {{ background: var(--pdx-mint); color: #0b5c55; border-color: transparent; }}
.pdx-chip-lime {{ background: var(--pdx-lime); color: var(--pdx-deep); border-color: transparent; }}

/* Calendar month grid */
.pdx-cal {{ display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 6px; }}
.pdx-dow {{
  font-size: 11.5px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase;
  color: var(--pdx-muted); padding: 0 6px 2px;
}}
.pdx-day {{
  background: var(--pdx-soft); border-radius: 10px; min-height: 92px; padding: 7px;
  display: flex; flex-direction: column; gap: 4px; min-width: 0;
}}
.pdx-day b {{ font-size: 12px; color: var(--pdx-ink-soft); }}
.pdx-day-closed b {{ color: var(--pdx-down); }}
.pdx-day-off {{ background: transparent; }}
.pdx-day-today {{ outline: 2px solid var(--pdx-forest); background: #fff; }}
.pdx-ev {{
  display: block; font-size: 11.5px; font-weight: 800; border-radius: 6px; padding: 2px 6px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-decoration: none;
}}

@media (max-width: 640px) {{
  .pdx-kpi .pdx-kpi-value {{ font-size: 18px; }}
  [class*="st-key-pdx-card-"] {{ padding: 14px 14px; }}
  .pdx-cal {{ gap: 3px; }}
  .pdx-day {{ min-height: 64px; padding: 4px; }}
  .pdx-ev {{ font-size: 10.5px; padding: 1px 3px; }}
}}
</style>
"""


def esc(value) -> str:
    """HTML-escape any value (None becomes an empty string), quotes included."""
    return html.escape('' if value is None else str(value), quote=True)


def inject_styles() -> None:
    st.html(STYLES)


def render(markup: str) -> None:
    st.html(f'<div class="pdx">{markup}</div>')


def chip(text, tone: str = '') -> str:
    """A pill. tone: '', 'up', 'down', 'warn', 'brand' or 'lime'; anything else renders plain."""
    cls = 'pdx-chip' + (f' pdx-chip-{tone}' if tone and tone in CHIP_TONES else '')
    return f'<span class="{cls}">{esc(text)}</span>'


def kpi(label, value, note='', tone: str = '') -> str:
    """One metric tile. tone colours the note: '', 'up', 'down', 'warn' or 'muted'."""
    note_html = ''
    if note:
        tone_cls = f' pdx-tone-{tone}' if tone and tone in NOTE_TONES else ''
        note_html = f'<span class="pdx-kpi-note{tone_cls}">{esc(note)}</span>'
    return (
        f'<div class="pdx-kpi"><small class="pdx-kpi-label">{esc(label)}</small>'
        f'<b class="pdx-kpi-value">{esc(value)}</b>{note_html}</div>'
    )


def kpi_row_html(items) -> str:
    """Markup for a wrapping grid of tiles. items: markup strings from kpi()."""
    return f'<div class="pdx-kpis">{"".join(items)}</div>'


def kpi_row(items) -> None:
    render(kpi_row_html(items))


def split(key: str):
    """(left, right) panes that sit side by side and wrap, left on top, when space runs out.

    Left has a 300px basis and right 560px, growing 1:3 (the mock-up's flex values).
    """
    outer = st.container(horizontal=True, gap='medium', key=f'pdx-split-{key}')
    with outer:
        left = st.container(key=f'pdx-pane-left-{key}')
        right = st.container(key=f'pdx-pane-right-{key}')
    return left, right


def card(key: str):
    """A white bordered card that native Streamlit elements can be placed in."""
    return st.container(key=f'pdx-card-{key}')
