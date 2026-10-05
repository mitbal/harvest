"""Point the Telegram bot at the app's webhook. Run once after deploying (or after changing the secret).

    python -m pipeline.telegram_setup https://panendividen.com/webhooks/telegram

Needs ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_WEBHOOK_SECRET`` (the same secret the app checks).
"""
import argparse
import os
import sys

from harvest.alerts import call_telegram


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", help="Public HTTPS URL of /webhooks/telegram")
    args = parser.parse_args(argv)
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET")
    if not secret:
        parser.error("TELEGRAM_WEBHOOK_SECRET is not set")
    call_telegram("setWebhook", {"url": args.url, "secret_token": secret, "allowed_updates": ["message"]})
    info = call_telegram("getWebhookInfo", {})["result"]
    print(f"Webhook set to {info.get('url')}; pending updates: {info.get('pending_update_count', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
