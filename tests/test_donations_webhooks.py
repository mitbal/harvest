import datetime
import pandas as pd
import hashlib
import hmac
import json
from unittest.mock import patch

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

import harvest.alerts as alerts
import harvest.donations as donations
import harvest.trading_store as store
import harvest.webhooks as webhooks
from tests.fake_supabase import FakeSupabase


NOW = datetime.datetime(2026, 10, 5, 12, tzinfo=datetime.timezone.utc)
SECRETS = {
    "BMC_WEBHOOK_SECRET": "bmc-secret",
    "TRAKTEER_WEBHOOK_TOKEN": "trakteer-token",
    "TELEGRAM_WEBHOOK_SECRET": "telegram-secret",
}


def _bmc_event(tx="pi_1", email="Fan@Example.com", amount=5, currency="USD", event="donation.created"):
    return {
        "event_id": 99, "type": event, "live_mode": True, "created": 1759600000, "attempt": 1,
        "data": {"id": 7, "transaction_id": tx, "amount": amount, "currency": currency,
                 "supporter_email": email, "supporter_name": "Fan", "support_note": "thanks"},
    }


def _trakteer_tip(tx="trx-1", message="semangat! fan@example.com", price=60_000):
    return {
        "created_at": "2026-10-05T14:50:10+07:00", "transaction_id": tx, "type": "tip",
        "supporter_name": "Fan", "supporter_message": message, "unit": "Kopi",
        "quantity": 2, "price": price, "net_amount": price * 0.95,
    }


# ── Pure helpers ────────────────────────────────────────────────────────── #

def test_bmc_signature_is_hex_hmac_of_the_raw_body():
    body = b'{"type": "donation.created"}'
    signature = hmac.new(b"secret", body, hashlib.sha256).hexdigest()

    assert donations.verify_bmc_signature(body, signature, "secret")
    assert donations.verify_bmc_signature(body, signature.upper(), "secret")
    assert not donations.verify_bmc_signature(body + b" ", signature, "secret")
    assert not donations.verify_bmc_signature(body, signature, "")
    assert not donations.verify_bmc_signature(body, None, "secret")


def test_parse_bmc_donation_and_ignored_events():
    donation = donations.parse_bmc(_bmc_event())

    assert (donation.provider, donation.provider_tx_id, donation.email) == ("bmc", "pi_1", "fan@example.com")
    assert donation.amount_idr == 80_000
    assert donations.parse_bmc(_bmc_event(event="donation.refunded")) is None
    assert donations.parse_bmc(_bmc_event(event="extra_purchase.created")) is None


def test_parse_trakteer_reads_email_from_the_message():
    assert donations.parse_trakteer(_trakteer_tip()).email == "fan@example.com"
    assert donations.parse_trakteer(_trakteer_tip(message="Selalu berkarya")).email is None
    assert donations.parse_trakteer({"type": "tip"}) is None


@pytest.mark.parametrize("amount, days", [(None, 0), (10_000, 0), (25_000, 31), (59_999, 31), (60_000, 92), (500_000, 366)])
def test_days_for_picks_the_largest_tier_covered(amount, days):
    assert donations.days_for(amount) == days


def test_extended_expiry_starts_from_the_later_of_now_and_current():
    later = NOW + datetime.timedelta(days=10)
    assert donations.extended_expiry(None, 31, NOW) == NOW + datetime.timedelta(days=31)
    assert donations.extended_expiry(later.isoformat(), 31, NOW) == later + datetime.timedelta(days=31)
    expired = NOW - datetime.timedelta(days=100)
    assert donations.extended_expiry(expired.isoformat(), 31, NOW) == NOW + datetime.timedelta(days=31)


# ── Recording donations ─────────────────────────────────────────────────── #

def test_new_donation_grants_access():
    client = FakeSupabase()

    result = donations.record_donation(client, donations.parse_trakteer(_trakteer_tip()), now=NOW)

    assert result["status"] == "granted" and result["days"] == 92
    user = client.tables["premium_users"][0]
    assert user["email"] == "fan@example.com" and user["source"] == "trakteer"
    assert donations.is_active(user["expires_at"], NOW)
    assert len(client.tables["donations"]) == 1


def test_duplicate_delivery_changes_nothing():
    client = FakeSupabase()
    donation = donations.parse_trakteer(_trakteer_tip())
    donations.record_donation(client, donation, now=NOW)
    first_expiry = client.tables["premium_users"][0]["expires_at"]

    assert donations.record_donation(client, donation, now=NOW)["status"] == "duplicate"
    assert client.tables["premium_users"][0]["expires_at"] == first_expiry
    assert len(client.tables["donations"]) == 1


def test_second_donation_extends_the_current_expiry():
    client = FakeSupabase()
    donations.record_donation(client, donations.parse_trakteer(_trakteer_tip(tx="a", price=25_000)), now=NOW)
    donations.record_donation(client, donations.parse_trakteer(_trakteer_tip(tx="b", price=25_000)), now=NOW)

    expires = datetime.datetime.fromisoformat(client.tables["premium_users"][0]["expires_at"])
    assert expires == NOW + datetime.timedelta(days=62)


def test_unknown_email_is_stored_for_manual_matching():
    client = FakeSupabase()

    result = donations.record_donation(client, donations.parse_trakteer(_trakteer_tip(message="hi")), now=NOW)

    assert result == {"status": "unmatched"}
    assert client.tables["donations"][0]["email"] is None
    assert "premium_users" not in client.tables or not client.tables["premium_users"]


def test_small_donation_and_lifetime_supporters_keep_their_status():
    client = FakeSupabase({"premium_users": [{"email": "fan@example.com", "source": "manual", "expires_at": None}]})

    small = donations.record_donation(client, donations.parse_trakteer(_trakteer_tip(tx="s", price=5_000)), now=NOW)
    big = donations.record_donation(client, donations.parse_trakteer(_trakteer_tip(tx="b")), now=NOW)

    assert small["status"] == "below_tier"
    assert big["status"] == "lifetime"
    assert client.tables["premium_users"][0]["expires_at"] is None


def test_active_supporters(monkeypatch):
    client = FakeSupabase({"premium_users": [
        {"email": "life@example.com", "expires_at": None},
        {"email": "fan@example.com", "expires_at": (NOW + datetime.timedelta(days=5)).isoformat()},
        {"email": "lapsed@example.com", "expires_at": (NOW - datetime.timedelta(days=5)).isoformat()},
    ]})
    monkeypatch.delenv("PREMIUM_ENFORCED", raising=False)
    assert donations.active_supporters(client, NOW) is None
    monkeypatch.setenv("PREMIUM_ENFORCED", "true")
    assert donations.active_supporters(client, NOW) == {"life@example.com", "fan@example.com"}


# ── Webhook routes ──────────────────────────────────────────────────────── #

@pytest.fixture
def http(monkeypatch):
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    client = FakeSupabase()
    app = Starlette(routes=[
        Route("/webhooks/bmc", webhooks.bmc_webhook, methods=["POST"]),
        Route("/webhooks/trakteer", webhooks.trakteer_webhook, methods=["POST"]),
        Route("/webhooks/telegram", webhooks.telegram_webhook, methods=["POST"]),
    ])
    sent = []
    with patch.object(webhooks, "service_client", return_value=client), \
            patch.object(alerts, "send_message", side_effect=lambda chat, text, token=None: sent.append((chat, text))):
        yield TestClient(app), client, sent


def _post_bmc(test_client, event, secret="bmc-secret"):
    body = json.dumps(event).encode()
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return test_client.post("/webhooks/bmc", content=body, headers={"x-signature-sha256": signature})


def test_bmc_webhook_unlocks_access(http):
    test_client, client, _ = http

    response = _post_bmc(test_client, _bmc_event())

    assert response.status_code == 200 and response.json() == {"status": "granted"}
    assert client.tables["premium_users"][0]["email"] == "fan@example.com"


def test_bmc_webhook_rejects_a_bad_signature(http):
    test_client, client, _ = http

    response = _post_bmc(test_client, _bmc_event(), secret="wrong")

    assert response.status_code == 401
    assert not client.tables.get("donations")


def test_bmc_webhook_retries_are_idempotent(http):
    test_client, client, _ = http
    _post_bmc(test_client, _bmc_event())

    assert _post_bmc(test_client, _bmc_event()).json() == {"status": "duplicate"}
    assert len(client.tables["donations"]) == 1


def test_trakteer_webhook_checks_the_token(http):
    test_client, client, _ = http
    tip = _trakteer_tip()

    bad = test_client.post("/webhooks/trakteer", json=tip, headers={"X-Webhook-Token": "nope"})
    good = test_client.post("/webhooks/trakteer", json=tip, headers={"X-Webhook-Token": "trakteer-token"})

    assert bad.status_code == 401
    assert good.status_code == 200 and good.json()["status"] == "granted"


def test_webhook_store_failure_asks_for_a_retry(http):
    test_client, _, _ = http
    with patch.object(donations, "record_donation", side_effect=RuntimeError("db down")):
        response = test_client.post("/webhooks/trakteer", json=_trakteer_tip(), headers={"X-Webhook-Token": "trakteer-token"})

    assert response.status_code == 500


def test_webhook_logs_leave_out_personal_data(http, caplog):
    test_client, _, _ = http
    with caplog.at_level("INFO", logger="webhooks"):
        test_client.post("/webhooks/trakteer", json=_trakteer_tip(), headers={"X-Webhook-Token": "trakteer-token"})

    assert "trx-1" in caplog.text
    assert "fan@example.com" not in caplog.text and "semangat" not in caplog.text


def _telegram(test_client, text, chat_id=555, secret="telegram-secret"):
    update = {"update_id": 1, "message": {"chat": {"id": chat_id}, "text": text}}
    return test_client.post("/webhooks/telegram", json=update, headers={"X-Telegram-Bot-Api-Secret-Token": secret})


def test_telegram_start_links_the_chat_and_stop_unlinks(http):
    test_client, client, sent = http
    store.save_settings(client, "fan@example.com", {"telegram_link_token": "tok123"})

    assert _telegram(test_client, "/start tok123").status_code == 200
    settings = store.get_settings(client, "fan@example.com")
    assert settings["telegram_chat_id"] == "555"
    assert settings["telegram_link_token"] is None
    assert sent[-1][0] == 555 and "Linked" in sent[-1][1] and "fa*@example.com" in sent[-1][1]

    _telegram(test_client, "/stop")
    assert store.get_settings(client, "fan@example.com")["telegram_chat_id"] is None
    assert "Alerts stopped" in sent[-1][1]


def test_telegram_unknown_token_and_bad_secret(http):
    test_client, client, sent = http

    assert _telegram(test_client, "/start nope").status_code == 200
    assert "expired" in sent[-1][1]
    assert _telegram(test_client, "/start nope", secret="wrong").status_code == 401


# ── Alerts ──────────────────────────────────────────────────────────────── #

AS_OF = datetime.date(2026, 10, 5)  # Monday; next session Tuesday 6 Oct


def _signal(symbol="BBRI.JK", status="actionable"):
    return {
        "symbol": symbol, "status": status, "entry_price": 3175.0,
        "exit_by": datetime.date(2027, 1, 11), "key_date": datetime.date(2026, 12, 29),
        "key_date_estimated": True,
        "detail": {"window_start": "2026-10-01", "holdout": {"trades": 6, "wins": 4}},
    }


def _alert_client():
    client = FakeSupabase()
    store.save_settings(client, "all@example.com", {"telegram_chat_id": "1", "alert_prefs": {"new_signals": True}})
    store.save_settings(client, "watch@example.com", {"telegram_chat_id": "2", "alert_prefs": {"watch": ["BBRI.JK"]}})
    store.save_settings(client, "quiet@example.com", {"telegram_chat_id": "3"})
    store.save_settings(client, "nochat@example.com", {"alert_prefs": {"new_signals": True}})
    return client


def test_signal_alerts_follow_preferences_and_send_once(monkeypatch):
    monkeypatch.delenv("PREMIUM_ENFORCED", raising=False)
    client, sent = _alert_client(), []
    send = lambda chat, text, token=None: sent.append((chat, text))
    signals = [_signal(), _signal("TLKM.JK"), _signal("ASII.JK", status="opening_soon")]

    stats = alerts.send_dividend_play_alerts(client, AS_OF, signals, token="t", send=send)

    assert sorted(chat for chat, _ in sent) == ["1", "1", "2"]
    assert "entry window for BBRI is open" in sent[0][1] and "Rp 3,175" in sent[0][1]
    assert stats["signal"] == 3
    again = alerts.send_dividend_play_alerts(client, AS_OF, signals, token="t", send=send)
    assert again["signal"] == 0 and again["already_sent"] == 3
    assert len(sent) == 3


def test_signal_alerts_need_access_when_enforced(monkeypatch):
    monkeypatch.setenv("PREMIUM_ENFORCED", "true")
    client, sent = _alert_client(), []
    client.tables["premium_users"] = [{"email": "watch@example.com", "expires_at": None}]

    alerts.send_dividend_play_alerts(client, AS_OF, [_signal()], token="t", send=lambda c, t, k=None: sent.append(c))

    assert sent == ["2"]


def test_exit_reminder_one_session_before_exit_by(monkeypatch):
    monkeypatch.delenv("PREMIUM_ENFORCED", raising=False)
    client, sent = _alert_client(), []
    send = lambda chat, text, token=None: sent.append((chat, text))
    position = {"tool": "dividend_play", "symbol": "BBRI.JK", "entry_date": "2026-10-01",
                "entry_price": 3175, "lots": 10, "exit_rule": {}}
    store.create_position(client, "quiet@example.com", {**position, "exit_by": "2026-10-06"})
    store.create_position(client, "quiet@example.com", {**position, "symbol": "TLKM.JK", "exit_by": "2026-10-20"})

    stats = alerts.send_dividend_play_alerts(client, AS_OF, [], token="t", send=send)

    assert stats["exit"] == 1
    assert sent == [("3", sent[0][1])]
    assert "sell BBRI by Tue 6 Oct (the next session)" in sent[0][1] and "10 lots" in sent[0][1]


def test_failed_send_is_retried_next_run(monkeypatch):
    monkeypatch.delenv("PREMIUM_ENFORCED", raising=False)
    client = _alert_client()

    def broken(chat, text, token=None):
        raise RuntimeError("telegram down")

    failed = alerts.send_dividend_play_alerts(client, AS_OF, [_signal()], token="t", send=broken)
    assert failed["failed"] == 2
    sent = []
    retried = alerts.send_dividend_play_alerts(client, AS_OF, [_signal()], token="t", send=lambda c, t, k=None: sent.append(c))
    assert retried["signal"] == 2 and sorted(sent) == ["1", "2"]


def test_alerts_skip_without_a_bot_token(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    assert "skipped" in alerts.send_dividend_play_alerts(FakeSupabase(), AS_OF, [_signal()])


def _long_term_rows(**overrides):
    row = {"Current PE": 9.0, "PE Mean": 12.0, "Statement Date": pd.Timestamp("2026-06-30"),
           "Earnings Growth TTM": 6.0, "Revenue Growth TTM": 4.0}
    return {"ASII.JK": {**row, **overrides}}


def _long_term_positions(client, entry_date="2026-05-04"):
    store.create_position(client, "quiet@example.com", {
        "tool": "long_term", "symbol": "ASII.JK", "entry_date": entry_date,
        "entry_price": 4700, "lots": 20, "exit_rule": {"pe_mean": 12.0},
    })
    return store.open_positions_all(client, "long_term")


def test_long_term_exit_alert_when_p_e_reaches_its_mean():
    client, sent = _alert_client(), []
    positions = _long_term_positions(client)
    send = lambda chat, text, token=None: sent.append((chat, text))

    quiet = alerts.send_long_term_exit_alerts(client, AS_OF, _long_term_rows(), positions, token="t", send=send)
    assert quiet["exit"] == 0 and sent == []

    stats = alerts.send_long_term_exit_alerts(
        client, AS_OF, _long_term_rows(**{"Current PE": 12.4}), positions, token="t", send=send,
    )
    assert stats["exit"] == 1
    assert sent[0][0] == "3"
    assert "ASII's P/E is back at its historical mean (12.4 vs a mean of 12.0)" in sent[0][1]
    assert "20 lots bought at Rp 4,700" in sent[0][1]
    again = alerts.send_long_term_exit_alerts(
        client, AS_OF, _long_term_rows(**{"Current PE": 13.0}), positions, token="t", send=send,
    )
    assert again["already_sent"] == 1 and len(sent) == 1


def test_long_term_exit_alert_once_per_quarter_that_turns_growth_negative():
    client, sent = _alert_client(), []
    positions = _long_term_positions(client)
    send = lambda chat, text, token=None: sent.append(text)
    falling = _long_term_rows(**{"Earnings Growth TTM": -3.5})

    alerts.send_long_term_exit_alerts(client, AS_OF, falling, positions, token="t", send=send)
    alerts.send_long_term_exit_alerts(client, AS_OF, falling, positions, token="t", send=send)
    assert len(sent) == 1
    assert "latest quarter (Tue 30 Jun 2026) turned trailing twelve-month growth negative (earnings -3.5%)" in sent[0]

    next_quarter = _long_term_rows(**{"Earnings Growth TTM": -1.0, "Statement Date": pd.Timestamp("2026-09-30")})
    alerts.send_long_term_exit_alerts(client, AS_OF, next_quarter, positions, token="t", send=send)
    assert len(sent) == 2


def test_long_term_growth_alert_ignores_quarters_reported_before_entry():
    client, sent = _alert_client(), []
    positions = _long_term_positions(client, entry_date="2026-08-03")

    stats = alerts.send_long_term_exit_alerts(
        client, AS_OF, _long_term_rows(**{"Revenue Growth TTM": -2.0}), positions, token="t",
        send=lambda chat, text, token=None: sent.append(text),
    )

    assert stats["exit"] == 0 and sent == []
