"""Supporter gate for the trading pages.

``require_premium()`` at the top of a page (or view) blocks everyone without an
active row in ``premium_users`` once ``PREMIUM_ENFORCED`` is on. The check runs
on the server and ends the script with ``st.stop()``, so nothing below it is
computed or sent to the browser.

Every supporter message in the app comes from here or from the two small pieces
in ``trading_ui`` (``unlock_prompt`` and ``locked_note``), so the wording and the
look stay the same on every page:

- ``require_premium(title, subtitle)``: the full paywall card, in place of a view.
- ``paywall_dialog()``: the same card in a dialog, opened by every Unlock button.
- ``trading_ui.unlock_prompt``: the dark card naming the signals that are hidden.
- ``trading_ui.locked_note``: one line under a table or list with fewer columns.

The CSS for the ``hv-paywall`` markup lives in ``trading_ui.STYLES``.
"""
import html
import logging
from typing import Optional

import streamlit as st

from harvest.donations import TIERS, is_active, is_enforced  # noqa: F401  (re-exported)


logger = logging.getLogger(__name__)

PREMIUM_CACHE_TTL = 10 * 60
TRAKTEER_URL = 'https://trakteer.id/mitbal'
BMC_URL = 'https://buymeacoffee.com/mitbal'
COMMUNITY_URL = 'https://reddit.com/r/panendividen'

# One vocabulary everywhere: people who donate are "supporters", what they get is
# "supporter access", and every button that leads to the paywall says UNLOCK_LABEL.
UNLOCK_LABEL = 'Unlock full access'
UNLOCK_ICON = ':material/lock_open:'
TAGLINE = 'Supporters see every signal, open the evidence for any stock and use the research Lab.'
WHY = ('Panen Dividen stays free and ad-free. Supporters cover the data and servers behind '
       'the trading tools, and get full access in return.')
BENEFITS = [
    'Every signal, candidate and entry window, not only the top one',
    'The verdict, best exit and past record on every table',
    'Evidence and backtests for any stock',
    'The research Lab on Dividend, Swing and Value Play',
    'A trade journal with sell-by reminders and Telegram alerts',
]
LAB_TITLE = 'Unlock the research Lab'
LAB_SUBTITLE = 'Search for setups and backtest any stock with your own parameters.'


@st.cache_resource(show_spinner=False)
def get_connection():
    """The app's signed-in Supabase connection, for premium and trading tables."""
    from st_supabase_connection import SupabaseConnection

    conn = st.connection('supabase', type=SupabaseConnection)
    conn.auth.sign_in_with_password({
        'email': st.secrets['connections']['supabase']['EMAIL_ADDRESS'],
        'password': st.secrets['connections']['supabase']['PASSWORD'],
    })
    return conn


@st.cache_data(ttl=PREMIUM_CACHE_TTL, show_spinner=False)
def _premium_rows(email: str, _client=None) -> list[dict]:
    client = _client or get_connection()
    return client.table('premium_users').select('expires_at').eq('email', email).execute().data or []


def is_premium(email: Optional[str], client=None) -> bool:
    """Whether ``email`` has an active ``premium_users`` row; cached for 10 minutes.

    Lookup errors count as not premium and are not cached, so the next rerun retries.
    """
    if not email:
        return False
    try:
        rows = _premium_rows(email.strip().lower(), _client=client)
    except Exception:
        logger.exception('Premium lookup failed')
        return False
    return any(is_active(row.get('expires_at')) for row in rows)


def current_email() -> Optional[str]:
    """The logged-in visitor's email, or None."""
    return st.user.email if st.user.is_logged_in else None


def has_access() -> bool:
    """Whether the visitor gets the full trading tools (always, while the gate is off)."""
    return not is_enforced() or is_premium(current_email())


# ═══════════════════════════════════════════════════════════════════════════════
# PAYWALL
# ═══════════════════════════════════════════════════════════════════════════════

def _esc(value) -> str:
    return html.escape('' if value is None else str(value))


def _icon(name: str) -> str:
    return f'<span class="icon" aria-hidden="true">{name}</span>'


def _tier_cards() -> str:
    monthly = TIERS[0]['min_idr']
    cards = []
    for tier in TIERS:
        months = max(1, round(tier['days'] / 30.5))
        per_month = tier['min_idr'] / months
        saving = 1 - per_month / monthly
        best = tier is TIERS[-1] and saving > 0
        note = (f'≈ Rp {per_month / 1000:.0f}rb a month · save {saving:.0%}' if saving > 0.005
                else 'Billed once, no auto-renew')
        cards.append(
            f'<div class="tier{" best" if best else ""}">'
            + ('<span class="flag">Best value</span>' if best else '')
            + f'<div class="name">{_esc(tier["name"])}</div><div class="price">{_esc(tier["price"])}</div>'
            f'<div class="dur">{_esc(tier["duration"])} of access</div><div class="note">{_esc(note)}</div></div>'
        )
    return f'<div class="tiers">{"".join(cards)}</div>'


def _paywall_markup(title: str, subtitle: str, email: Optional[str]) -> str:
    perks = ''.join(f'<li>{_icon("check_circle")}<span>{_esc(b)}</span></li>' for b in BENEFITS)
    email_step = (f'Donate with <b>{_esc(email)}</b>, the email you are logged in with.' if email
                  else 'Log in with Google, then donate with that same email.')
    steps = [
        email_step,
        'On Trakteer, also write that email in your support message.',
        'Access unlocks within minutes, on every device you log in from.',
    ]
    return (
        '<div class="hv"><div class="hv-paywall">'
        f'<div class="head"><div class="lock">{_icon("lock")}</div><div>'
        f'<div class="eyebrow">Supporter access</div><h2>{_esc(title)}</h2>'
        f'<p>{_esc(subtitle)}</p></div></div>'
        f'<div class="body"><div><ul class="perks">{perks}</ul><p class="why">{_esc(WHY)}</p></div>'
        f'<div>{_tier_cards()}<ol class="steps">'
        + ''.join(f'<li>{s}</li>' for s in steps) + '</ol></div></div>'
        '</div></div>'
    )


def _render_paywall(title: str, subtitle: str, key: str) -> None:
    """The paywall card with its actions inside; ``key`` keeps the widgets unique."""
    email = current_email()
    with st.container(key=f'hv-paywall-{key}'):
        st.html(_paywall_markup(title, subtitle, email))
        with st.container(horizontal=True):
            if not email:
                st.button(
                    'Log in with Google', icon=':material/login:', type='primary',
                    on_click=lambda: st.login('google'), key=f'premium_login_{key}',
                )
            st.link_button('Donate on Trakteer', TRAKTEER_URL, icon=':material/favorite:',
                           type='secondary' if not email else 'primary')
            st.link_button('Buy Me a Coffee', BMC_URL, icon=':material/coffee:')
    st.caption(f'Donated but still locked? Ask on [r/panendividen]({COMMUNITY_URL}) and we will sort it out.')


@st.dialog('Supporter access', width='large', icon=':material/workspace_premium:')
def paywall_dialog() -> None:
    """The paywall in a dialog; every Unlock button in the app opens this."""
    _render_paywall(UNLOCK_LABEL, TAGLINE, key='dialog')


def require_premium(title: str = LAB_TITLE, subtitle: str = LAB_SUBTITLE) -> None:
    """Stop the page with the paywall unless the visitor is an active supporter.

    Does nothing while ``PREMIUM_ENFORCED`` is off. ``title`` and ``subtitle`` say
    what this view unlocks; the benefits, prices and steps are the same everywhere.
    """
    if not is_enforced():
        return
    email = current_email()
    if email and is_premium(email):
        return

    _render_paywall(title, subtitle, key='gate')
    st.stop()
