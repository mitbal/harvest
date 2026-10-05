"""Starlette routes for donation and Telegram webhooks (mounted in ``server.py``).

Each route checks its signature or token first, does one small database write
with the service-role client and answers right away. Logs carry the provider,
transaction id and outcome only, never the payload or the supporter's email.
"""
import json
import logging
import os
from functools import lru_cache

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response

import harvest.alerts as alerts
import harvest.donations as donations


logger = logging.getLogger('webhooks')


@lru_cache(maxsize=1)
def service_client():
    """Supabase client with the service-role key; bypasses row level security, server side only."""
    from supabase import create_client

    url, key = os.environ.get('SUPABASE_URL'), os.environ.get('SUPABASE_SERVICE_KEY')
    if not url or not key:
        raise RuntimeError('SUPABASE_URL and SUPABASE_SERVICE_KEY are required for webhooks')
    return create_client(url, key)


async def _donation(request, provider: str, verified: bool, parse) -> Response:
    if not verified:
        logger.warning('Donation webhook rejected | provider=%s | bad signature or token', provider)
        return Response(status_code=401)
    try:
        payload = json.loads(await request.body())
    except ValueError:
        return Response(status_code=400)
    donation = parse(payload)
    if donation is None:
        return JSONResponse({'status': 'ignored'})
    try:
        result = await run_in_threadpool(donations.record_donation, service_client(), donation)
    except Exception:
        logger.exception('Donation webhook failed | provider=%s | tx=%s', provider, donation.provider_tx_id)
        # A 5xx makes the provider retry; the insert is idempotent.
        return Response(status_code=500)
    logger.info('Donation webhook | provider=%s | tx=%s | status=%s', provider, donation.provider_tx_id, result['status'])
    return JSONResponse({'status': result['status']})


async def bmc_webhook(request):
    body = await request.body()
    verified = donations.verify_bmc_signature(
        body, request.headers.get('x-signature-sha256'), os.environ.get('BMC_WEBHOOK_SECRET', ''),
    )
    return await _donation(request, 'bmc', verified, donations.parse_bmc)


async def trakteer_webhook(request):
    verified = donations.verify_token(
        request.headers.get('x-webhook-token'), os.environ.get('TRAKTEER_WEBHOOK_TOKEN', ''),
    )
    return await _donation(request, 'trakteer', verified, donations.parse_trakteer)


async def telegram_webhook(request):
    if not donations.verify_token(
        request.headers.get('x-telegram-bot-api-secret-token'), os.environ.get('TELEGRAM_WEBHOOK_SECRET', ''),
    ):
        return Response(status_code=401)
    try:
        update = json.loads(await request.body())
    except ValueError:
        return Response(status_code=400)
    message = update.get('message') or {}
    chat_id = (message.get('chat') or {}).get('id')
    text = (message.get('text') or '').strip()
    if chat_id is None or not text:
        return Response(status_code=200)

    command, _, argument = text.partition(' ')
    try:
        if command == '/start' and argument.strip():
            reply = await run_in_threadpool(alerts.link_chat, service_client(), argument.strip(), chat_id)
        elif command == '/stop':
            reply = await run_in_threadpool(alerts.unlink_chat, service_client(), chat_id)
        else:
            reply = 'Link this chat from My Trades on panendividen.com to get Dividend Play alerts.'
        await run_in_threadpool(alerts.send_message, chat_id, reply)
    except Exception:
        logger.exception('Telegram webhook failed')
    # Always 200: Telegram retries failed updates, and a retry would not help here.
    return Response(status_code=200)
