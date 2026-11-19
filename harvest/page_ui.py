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
/* Wide-left split (split(..., wide_left=True)): main card 3 1 560px, side column 2 1 360px */
[class*="st-key-pdx-wsplit-"] > :has(> [class*="st-key-pdx-wpane-left-"]) {{
  flex: 3 1 560px; width: auto; min-width: 0; max-width: 100%;
}}
[class*="st-key-pdx-wsplit-"] > :has(> [class*="st-key-pdx-wpane-right-"]) {{
  flex: 2 1 360px; width: auto; min-width: 0; max-width: 100%;
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
.pdx-cal {{ display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 4px; }}
.pdx-dow {{
  font-size: 11.5px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase;
  color: var(--pdx-muted); padding: 0 6px 2px;
}}
/* Row height follows the window so a whole month fits the first screen: 372px is everything on
   that screen that is not a cell row. On the Single Month page the card starts at ~209px and has
   74px above its first row (border, padding, 32px heading, 8px gap, weekday row) = 283px; below the
   last row come 43px (8px gap, 20px legend, padding, border), the 36px footer and 10px of slack.
   The rest is shared by --pdx-weeks rows, less the 4px grid gap each row brings. Tall windows keep
   roomy 92px cells (from ~964px tall); short ones go down to 52px. A busy cell (number line and
   three chips) needs ~73px, which six-week months get from ~850px tall. */
.pdx-day {{
  background: var(--pdx-soft); border-radius: 10px; padding: 3px 5px;
  min-height: clamp(52px, calc((100dvh - 372px) / var(--pdx-weeks, 6) - 4px), 92px);
  display: flex; flex-direction: column; gap: 2px; min-width: 0;
}}
.pdx-day-hd {{ display: flex; align-items: center; justify-content: space-between; gap: 3px; min-height: 15px; min-width: 0; }}
.pdx-day b {{ font-size: 12px; line-height: 1.25; color: var(--pdx-ink-soft); flex: none; }}
.pdx-day-closed b {{ color: var(--pdx-down); }}
.pdx-day-off {{ background: transparent; }}
.pdx-day-today {{ outline: 2px solid var(--pdx-forest); background: #fff; }}
.pdx-ev {{
  display: block; font-size: 11px; font-weight: 800; line-height: 1.2; border-radius: 6px; padding: 1px 6px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-decoration: none;
}}

.pdx a.pdx-ev:hover {{ text-decoration: none; box-shadow: inset 0 0 0 2px rgba(16, 37, 29, .35); }}
.pdx-ev-more {{
  background: #e4ebe7; color: var(--pdx-ink-soft); cursor: default;
  font-size: 10.5px; padding: 1px 4px; letter-spacing: -.01em;
}}
.pdx-day-hd .pdx-ev-more {{ flex: 0 1 auto; min-width: 0; }}
/* On phones the grid stops shrinking at 300px (see the media query) and scrolls inside its card */
.pdx-cal-wrap {{ overflow-x: auto; container: pdx-cal / inline-size; }}
/* The day-number line holds a two-digit day (~14px), a 3px gap and "+99 more" (~55px with its
   padding): 72px of content, so an 82px cell (5px padding each side). Seven cells and six 4px gaps
   make 7 * 82 + 24 = 598px of grid; narrower grids show "+99" rather than cut it off */
@container pdx-cal (max-width: 597px) {{
  .pdx-ev-more-word {{ display: none; }}
}}
.pdx-cal-legend {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-start; gap: 6px; }}
.pdx-cal-legend .pdx-note {{ margin-right: 2px; }}
.pdx-cal-legend .pdx-legend-key {{ min-width: 38px; text-align: center; }}

@media (max-width: 640px) {{
  .pdx-kpi .pdx-kpi-value {{ font-size: 18px; }}
  [class*="st-key-pdx-card-"] {{ padding: 14px 14px; }}
  .pdx-cal {{ gap: 3px; min-width: 300px; }}
  .pdx-dow {{ font-size: 10.5px; letter-spacing: .02em; padding: 0 2px 2px; }}
  .pdx-day {{ min-height: 52px; padding: 4px 3px; gap: 2px; }}
  .pdx-day b {{ font-size: 11px; }}
  .pdx-ev {{ font-size: 10px; padding: 1px 2px; border-radius: 4px; letter-spacing: -.01em; }}
  .pdx-ev-more-word {{ display: none; }}
  .pdx-cal-legend {{ justify-content: flex-start; }}
  .pdx-cal-legend .pdx-legend-key {{ min-width: 0; }}
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


def kpi_row_html(items, columns=None) -> str:
    """Markup for a grid of tiles. items: markup strings from kpi().

    By default tiles wrap at a 175px minimum; columns=n keeps n equal columns at every width.
    """
    style = f' style="grid-template-columns:repeat({int(columns)},minmax(0,1fr))"' if columns else ''
    return f'<div class="pdx-kpis"{style}>{"".join(items)}</div>'


def kpi_row(items, columns=None) -> None:
    render(kpi_row_html(items, columns))


def split_keys(key: str, wide_left: bool = False):
    """Container keys (outer, left, right) that split() uses; each prefix has its own CSS sizes."""
    if wide_left:
        return f'pdx-wsplit-{key}', f'pdx-wpane-left-{key}', f'pdx-wpane-right-{key}'
    return f'pdx-split-{key}', f'pdx-pane-left-{key}', f'pdx-pane-right-{key}'


def split(key: str, wide_left: bool = False):
    """(left, right) panes that sit side by side and wrap, left on top, when space runs out.

    Default (simulator): left has a 300px basis and right 560px, growing 1:3.
    wide_left (calendar): left 560px and right 360px, growing 3:2.
    """
    outer_key, left_key, right_key = split_keys(key, wide_left)
    outer = st.container(horizontal=True, gap='medium', key=outer_key)
    with outer:
        left = st.container(key=left_key)
        right = st.container(key=right_key)
    return left, right


def card(key: str, gap='small'):
    """A white bordered card that native Streamlit elements can be placed in.

    gap is st.container's gap between the card's elements ('small', 1rem, by default).
    """
    return st.container(key=f'pdx-card-{key}', gap=gap)


# ---------------------------------------------------------------------------
# Share bar: Copy image, Copy link, Share (where the browser can share files) and Download
# ---------------------------------------------------------------------------

_SHARE_BAR_HTML = """
<div class="pdx-share" role="group" aria-label="Share this calendar">
  <span class="pdx-share-label">Share</span>
  <button type="button" class="pdx-share-btn" data-act="copy-image">Copy image</button>
  <button type="button" class="pdx-share-btn" data-act="copy-link">Copy link</button>
  <button type="button" class="pdx-share-btn" data-act="share" hidden>Share</button>
  <a class="pdx-share-btn" data-act="download" download>Download</a>
  <span class="pdx-share-status" role="status" aria-live="polite"></span>
</div>
"""

# The page's --pdx-* tokens reach into the component's shadow root; the fallbacks keep it styled without them
_SHARE_BAR_CSS = """
.pdx-share {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px;
  font-family: 'Manrope', system-ui, sans-serif; color: var(--pdx-ink, #10251d);
}
.pdx-share-label { font-size: 12.5px; font-weight: 700; color: var(--pdx-muted, #5c6d66); margin-right: 2px; }
.pdx-share-btn {
  display: inline-flex; align-items: center; min-height: 32px; padding: 0 12px; box-sizing: border-box;
  font: inherit; font-size: 12.5px; font-weight: 700; line-height: 1; text-decoration: none; cursor: pointer;
  color: var(--pdx-ink-soft, #3c5148); background: #fff;
  border: 1px solid var(--pdx-line, #dbe3df); border-radius: 99px; white-space: nowrap;
}
.pdx-share-btn:hover { background: var(--pdx-mint, #e3f1eb); color: var(--pdx-deep, #052d22); }
.pdx-share-btn:focus-visible { outline: 2px solid var(--pdx-forest, #073f2e); outline-offset: 2px; }
.pdx-share-btn[hidden] { display: none; }
.pdx-share-status { font-size: 12.5px; font-weight: 700; color: var(--pdx-up, #15803d); }
.pdx-share-status.is-error { color: var(--pdx-down, #c2410c); }
@media (max-width: 640px) {
  .pdx-share-label { flex-basis: 100%; }
  .pdx-share-btn { min-height: 36px; }
  .pdx-share-status { flex-basis: 100%; }
}
"""

_SHARE_BAR_JS = """
// Per-instance state, keyed by the mount point so two share bars never mix up their images
const instances = new WeakMap()

const BLOCKED = "Your browser blocked copying the image. Use Download instead."
const MISSING = "The image is not available right now."

export default function (component) {
  const { data, parentElement } = component
  const root = parentElement.querySelector(".pdx-share")
  if (!root || !data) return

  let st = instances.get(parentElement)
  if (!st) {
    st = { url: null, png: null, blob: null, failed: false, timer: null, observer: null }
    instances.set(parentElement, st)
  }
  // A new selection (year, stock list, Syariah) means a different image
  if (st.url !== data.image_url) {
    st.url = data.image_url
    st.png = null
    st.blob = null
    st.failed = false
  }

  const status = root.querySelector(".pdx-share-status")
  const copyImage = root.querySelector('[data-act="copy-image"]')
  const copyLink = root.querySelector('[data-act="copy-link"]')
  const shareBtn = root.querySelector('[data-act="share"]')
  const download = root.querySelector('[data-act="download"]')

  const say = (text, isError) => {
    status.textContent = text
    status.classList.toggle("is-error", !!isError)
    clearTimeout(st.timer)
    st.timer = setTimeout(() => { status.textContent = "" }, isError ? 8000 : 3000)
  }

  // One fetch per image; a failure is remembered and allows a retry on the next click
  const fetchPng = () => {
    if (!st.png) {
      const url = st.url
      st.png = fetch(url, { credentials: "same-origin" }).then((r) => {
        if (!r.ok) throw new Error("HTTP " + r.status)
        return r.blob()
      }).then((b) => {
        const blob = b.type === "image/png" ? b : new Blob([b], { type: "image/png" })
        if (st.url === url) { st.blob = blob; st.failed = false }
        return blob
      })
      st.png.catch(() => {
        if (st.url === url) { st.png = null; st.failed = true }
      })
    }
    return st.png
  }
  const prefetch = () => { fetchPng().catch(() => {}) }

  const canShareFiles = (() => {
    try {
      return !!(navigator.canShare && navigator.share &&
        navigator.canShare({ files: [new File([""], "x.png", { type: "image/png" })] }))
    } catch (e) { return false }
  })()
  shareBtn.hidden = !canShareFiles

  download.setAttribute("href", data.image_url)
  download.setAttribute("download", data.filename)
  download.setAttribute("aria-label", "Download the calendar image")
  copyImage.setAttribute("aria-label", "Copy the calendar image")
  copyLink.setAttribute("aria-label", "Copy a link to this calendar")
  shareBtn.setAttribute("aria-label", "Share the calendar image")

  // Safari only allows clipboard writes inside the click, so the ClipboardItem is made right here
  // with a promise for the image instead of awaiting the download first
  copyImage.onclick = () => {
    if (!navigator.clipboard || !navigator.clipboard.write || typeof ClipboardItem === "undefined") {
      say(BLOCKED, true)
      return
    }
    const png = fetchPng()
    let item
    try {
      item = new ClipboardItem({ "image/png": png })
    } catch (e) {
      png.catch(() => {})
      say(BLOCKED, true)
      return
    }
    navigator.clipboard.write([item]).then(
      () => say("Image copied"),
      () => png.then(() => say(BLOCKED, true), () => say(MISSING, true)),
    )
  }

  const copyText = (text) => {
    if (navigator.clipboard && navigator.clipboard.writeText) return navigator.clipboard.writeText(text)
    return new Promise((resolve, reject) => {
      const area = document.createElement("textarea")
      area.value = text
      area.setAttribute("readonly", "")
      area.style.position = "fixed"
      area.style.opacity = "0"
      root.appendChild(area)
      area.select()
      let ok = false
      try { ok = document.execCommand("copy") } catch (e) { ok = false }
      area.remove()
      ok ? resolve() : reject(new Error("copy failed"))
    })
  }

  // share_url is a path; resolve it on the host the page is served from, so a copied link from
  // localhost or staging stays there. The component runs in the main document, not an iframe
  const link = new URL(data.share_url, window.location.origin).href

  copyLink.onclick = () => {
    copyText(link).then(
      () => say("Link copied"),
      () => say("Could not copy. The link is " + link, true),
    )
  }

  shareBtn.onclick = () => {
    const send = (blob) => navigator.share({
      files: [new File([blob], data.filename, { type: "image/png" })],
      title: data.title, text: data.text, url: link,
    })
    // With the image already here the share sheet opens inside the tap, which Safari insists on
    const shared = st.blob ? send(st.blob) : fetchPng().then(send)
    shared.catch((e) => {
      if (e && e.name === "AbortError") return
      if (st.failed) say(MISSING, true)
      else if (e && e.name === "NotAllowedError") say("Tap Share again to open the share sheet.", true)
      else say("Sharing did not work here. Use Download instead.", true)
    })
  }

  // Native download, except when the image is known to be missing or is already in memory
  download.onclick = (e) => {
    if (st.failed) {
      e.preventDefault()
      say(MISSING, true)
      prefetch()
      return
    }
    if (st.blob) {
      e.preventDefault()
      const href = URL.createObjectURL(st.blob)
      const a = document.createElement("a")
      a.href = href
      a.download = data.filename
      root.appendChild(a)
      a.click()
      a.remove()
      setTimeout(() => URL.revokeObjectURL(href), 10000)
    }
  }

  // Fetch the image once someone shows interest, so Share and Copy are quick; never on page load
  root.onpointerenter = prefetch
  for (const el of [copyImage, copyLink, shareBtn, download]) el.onfocus = prefetch
  if (canShareFiles && !st.observer && "IntersectionObserver" in window) {
    // Phones have no hover: start when the bar scrolls into view, so the share sheet opens on the first tap
    st.observer = new IntersectionObserver((entries) => {
      if (entries.some((en) => en.isIntersecting)) prefetch()
    })
    st.observer.observe(root)
  }

  return () => {
    clearTimeout(st.timer)
    if (st.observer) { st.observer.disconnect(); st.observer = null }
  }
}
"""

def _share_bar_component():
    # Registered on every call, like an inline component at the top of a script: the registry belongs to
    # the Streamlit runtime, so a registration made when this module was first imported can be gone
    # (a new runtime in tests or after a dev reload). Registering the same name again is cheap.
    return st.components.v2.component(
        'pdx_share_bar', html=_SHARE_BAR_HTML, css=_SHARE_BAR_CSS, js=_SHARE_BAR_JS,
    )


def share_bar_data(image_url, share_url, filename, text, title='') -> dict:
    """What the share bar's script gets. image_url and share_url should be same-origin paths;
    the script resolves share_url against the page's origin (an absolute URL is kept as is)."""
    return {
        'image_url': str(image_url),
        'share_url': str(share_url),
        'filename': str(filename),
        'text': str(text),
        'title': str(title or text),
    }


def share_bar(image_url, share_url, filename, text, key, title=''):
    """Copy image, Copy link, Share (phones) and Download buttons for a server-rendered PNG.

    image_url is fetched and downloaded from the same origin, so it works on localhost; share_url is the
    path of the link people paste, which Copy link and Share turn into a full URL on the page's own host
    (so a link copied on localhost or staging stays there). When the image is missing the buttons say so
    instead of failing silently.
    """
    return _share_bar_component()(data=share_bar_data(image_url, share_url, filename, text, title), key=key)
