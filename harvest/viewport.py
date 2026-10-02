"""Measure how much of the browser viewport is left below a point on the page."""
import streamlit as st

# Reports window.innerHeight minus this element's page offset, so a chart
# rendered right after it can fill the rest of the first screen. It re-measures
# once the page settles (spinners above it disappear) and on window resize, and
# only reports changes bigger than a few pixels to avoid rerun loops.
_JS = """
export default function ({ parentElement, data, setStateValue }) {
  const reserve = data?.reserve ?? 0
  const known = data?.current ?? null
  const measure = () => {
    // parentElement is a shadow root, which has no box; measure our own node
    const anchor = parentElement.querySelector('#probe')
    if (!anchor) return
    const top = anchor.getBoundingClientRect().top + window.scrollY
    const available = Math.round(window.innerHeight - top - reserve)
    if (known === null || Math.abs(available - known) > 24) {
      setStateValue('available', available)
    }
  }
  const timers = [setTimeout(measure, 50), setTimeout(measure, 1200)]
  let pending
  const onResize = () => { clearTimeout(pending); pending = setTimeout(measure, 250) }
  window.addEventListener('resize', onResize)
  return () => {
    timers.forEach(clearTimeout)
    clearTimeout(pending)
    window.removeEventListener('resize', onResize)
  }
}
"""

_VIEWPORT_PROBE = st.components.v2.component('viewport_probe', html='<div id="probe"></div>', js=_JS)


def remaining_viewport_height(key: str, *, reserve: int = 0, default: int = 640, minimum: int = 420) -> int:
    """
    Pixels between this call's position and the bottom of the first screen,
    minus ``reserve`` for content that should stay visible below the chart.
    Returns ``default`` until the browser has reported a measurement.
    """
    current = (st.session_state.get(key) or {}).get('available')
    _VIEWPORT_PROBE(
        key=key,
        data={'reserve': reserve, 'current': current},
        on_available_change=lambda: None,
    )
    return max(minimum, int(current)) if current else default
