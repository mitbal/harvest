"""Supporter gate for the trading pages.

``require_premium()`` at the top of a page blocks everyone without an active
row in ``premium_users`` once ``PREMIUM_ENFORCED`` is on. The check runs on the
server and ends the script with ``st.stop()``, so nothing below it is computed
or sent to the browser.
"""
import logging
from typing import Callable, Optional

import streamlit as st

from harvest.donations import TIERS, is_active, is_enforced  # noqa: F401  (re-exported)


logger = logging.getLogger(__name__)

PREMIUM_CACHE_TTL = 10 * 60
TRAKTEER_URL = 'https://trakteer.id/mitbal'
BMC_URL = 'https://buymeacoffee.com/mitbal'
COMMUNITY_URL = 'https://reddit.com/r/panendividen'

BENEFITS = [
    'Full scanner with every stock that has a setup today',
    'Backtest any ticker with your own parameters',
    'Swing trading and long-term position labs',
]


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


def _render_paywall(email: Optional[str]) -> None:
    st.title('Unlock the trading lab')
    st.markdown(
        'Panen Dividen stays free and ad-free. Supporters cover the data and servers '
        'that run these tools, and get full access in return.'
    )
    st.markdown('\n'.join(f'- {benefit}' for benefit in BENEFITS))

    for column, tier in zip(st.columns(len(TIERS)), TIERS):
        with column.container(border=True):
            st.markdown(f"**{tier['name']}**")
            st.markdown(f"### {tier['price']}")
            st.caption(f"Access for {tier['duration']}")

    st.markdown('#### How to unlock')
    st.markdown(
        '1. Log in with Google.\n'
        '2. Donate on Trakteer or Buy Me a Coffee using the same email. On Trakteer, '
        'write that email in your support message.\n'
        '3. Access unlocks within minutes.'
    )
    if email:
        st.info(f'Logged in as **{email}**. Use this email when you donate.', icon=':material/person:')

    with st.container(horizontal=True):
        if not email:
            st.button(
                'Log in with Google', icon=':material/login:', type='primary',
                on_click=lambda: st.login('google'), key='premium_login',
            )
        st.link_button('Donate on Trakteer', TRAKTEER_URL, icon=':material/favorite:')
        st.link_button('Buy Me a Coffee', BMC_URL, icon=':material/coffee:')

    st.caption(f'Donated but still locked? Ask for help on [r/panendividen]({COMMUNITY_URL}).')


def require_premium(teaser: Optional[Callable[[], None]] = None) -> None:
    """Stop the page with the paywall unless the visitor is an active supporter.

    Does nothing while ``PREMIUM_ENFORCED`` is off. ``teaser`` is rendered above
    the paywall for visitors who are blocked.
    """
    if not is_enforced():
        return
    email = current_email()
    if email and is_premium(email):
        return

    if teaser is not None:
        teaser()
    _render_paywall(email)
    st.stop()
