"""Donation webhooks: verify, parse and turn a donation into supporter access.

Buy Me a Coffee signs the raw body with HMAC-SHA256 (hex) in
``x-signature-sha256``. Trakteer sends a shared token in ``X-Webhook-Token``
and no supporter email, so the email is read from the supporter's message
when present; otherwise the donation is stored for manual matching.

Streamlit-free: used by the webhook routes in ``server.py``.
"""
import datetime
import hashlib
import hmac
import os
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd


# Supporter tiers, shown on the paywall and used to turn a donation into days of
# access. A donation unlocks the largest tier it covers. Placeholder prices.
TIERS = [
    {'name': 'Monthly', 'price': 'Rp 25rb', 'duration': '1 month', 'min_idr': 25_000, 'days': 31},
    {'name': 'Quarterly', 'price': 'Rp 60rb', 'duration': '3 months', 'min_idr': 60_000, 'days': 92},
    {'name': 'Yearly', 'price': 'Rp 200rb', 'duration': '12 months', 'min_idr': 200_000, 'days': 366},
]
# Buy Me a Coffee pays in foreign currency; placeholder conversion rates to IDR.
IDR_PER_UNIT = {'IDR': 1, 'USD': 16_000}

EMAIL_PATTERN = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')


def is_enforced() -> bool:
    """True when env ``PREMIUM_ENFORCED`` is set to a truthy value (default off)."""
    return os.environ.get('PREMIUM_ENFORCED', '').strip().lower() in {'1', 'true', 'yes', 'on'}


def is_active(expires_at, now: Optional[datetime.datetime] = None) -> bool:
    """A supporter is active with no expiry (lifetime) or an expiry in the future."""
    if expires_at is None or pd.isna(expires_at):
        return True
    expiry = pd.Timestamp(expires_at)
    if expiry.tzinfo is None:
        expiry = expiry.tz_localize('UTC')
    return expiry > pd.Timestamp(now or datetime.datetime.now(datetime.timezone.utc))


def active_supporters(client, now: Optional[datetime.datetime] = None) -> Optional[set[str]]:
    """Emails with access right now; None while the gate is off (everyone has access)."""
    if not is_enforced():
        return None
    rows = client.table('premium_users').select('email,expires_at').execute().data or []
    return {row['email'].lower() for row in rows if is_active(row.get('expires_at'), now)}


@dataclass
class Donation:
    provider: str
    provider_tx_id: str
    email: Optional[str]
    amount_idr: Optional[float]
    payload: dict


def verify_bmc_signature(body: bytes, signature: Optional[str], secret: str) -> bool:
    """Whether ``signature`` is the hex HMAC-SHA256 of the raw body under ``secret``."""
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.strip().lower())


def verify_token(received: Optional[str], expected: str) -> bool:
    """Constant-time comparison of a shared webhook token."""
    if not expected or not received:
        return False
    return hmac.compare_digest(received.strip().encode(), expected.encode())


def email_in(text: Optional[str]) -> Optional[str]:
    """The first email address in free text, lower-cased."""
    match = EMAIL_PATTERN.search(text or '')
    return match.group(0).lower().rstrip('.') if match else None


def to_idr(amount, currency: Optional[str]) -> Optional[float]:
    rate = IDR_PER_UNIT.get((currency or 'IDR').upper())
    try:
        return float(amount) * rate if rate is not None else None
    except (TypeError, ValueError):
        return None


def parse_bmc(payload: dict) -> Optional[Donation]:
    """A donation or membership payment from a Buy Me a Coffee event; None for other events."""
    event_type = str(payload.get('type', ''))
    if not (event_type.startswith('donation.') or event_type.startswith('membership.')) \
            or event_type.endswith(('.refunded', '.cancelled', '.canceled')):
        return None
    data = payload.get('data') or {}
    tx_id = data.get('transaction_id') or data.get('id') or payload.get('event_id')
    if tx_id is None:
        return None
    email = (data.get('supporter_email') or '').strip().lower() or email_in(data.get('support_note'))
    return Donation(
        provider='bmc',
        provider_tx_id=str(tx_id),
        email=email or None,
        amount_idr=to_idr(data.get('amount'), data.get('currency')),
        payload=payload,
    )


def parse_trakteer(payload: dict) -> Optional[Donation]:
    """A Trakteer tip; the email can only come from the supporter's message."""
    if payload.get('type', 'tip') != 'tip' or not payload.get('transaction_id'):
        return None
    return Donation(
        provider='trakteer',
        provider_tx_id=str(payload['transaction_id']),
        email=email_in(payload.get('supporter_message')),
        amount_idr=to_idr(payload.get('price'), 'IDR'),
        payload=payload,
    )


def days_for(amount_idr: Optional[float]) -> int:
    """Days of access a donation buys: the largest tier it covers, 0 below the first tier."""
    days = 0
    for tier in TIERS:
        if amount_idr is not None and amount_idr >= tier['min_idr']:
            days = tier['days']
    return days


def extended_expiry(current, days: int, now: datetime.datetime) -> Optional[datetime.datetime]:
    """New expiry: ``days`` after the later of now and the current expiry; lifetime stays lifetime."""
    if days <= 0:
        return None if current is None else pd.Timestamp(current).to_pydatetime()
    start = now
    if current is not None:
        current = pd.Timestamp(current)
        if current.tzinfo is None:
            current = current.tz_localize('UTC')
        start = max(now, current.to_pydatetime())
    return start + datetime.timedelta(days=days)


def record_donation(client, donation: Donation, now: Optional[datetime.datetime] = None) -> dict:
    """Store the donation once and grant access; returns what happened.

    A repeated delivery of the same transaction changes nothing. Without an
    email, or below the first tier, the donation is stored without access.
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)
    inserted = client.table('donations').upsert({
        'provider': donation.provider,
        'provider_tx_id': donation.provider_tx_id,
        'email': donation.email,
        'amount_idr': donation.amount_idr,
        'payload': donation.payload,
    }, on_conflict='provider,provider_tx_id', ignore_duplicates=True).execute().data
    if not inserted:
        return {'status': 'duplicate'}
    if not donation.email:
        return {'status': 'unmatched'}
    days = days_for(donation.amount_idr)
    if days == 0:
        return {'status': 'below_tier', 'email': donation.email}

    rows = client.table('premium_users').select('expires_at').eq('email', donation.email).execute().data
    if rows and rows[0].get('expires_at') is None:
        return {'status': 'lifetime', 'email': donation.email}
    current = rows[0]['expires_at'] if rows else None
    expires = extended_expiry(current, days, now)
    client.table('premium_users').upsert({
        'email': donation.email,
        'source': donation.provider,
        'expires_at': expires.isoformat(),
        'updated_at': now.isoformat(),
    }, on_conflict='email').execute()
    return {'status': 'granted', 'email': donation.email, 'expires_at': expires.isoformat(), 'days': days}
