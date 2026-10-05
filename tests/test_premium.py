import datetime
from unittest.mock import MagicMock, patch

import pytest

import harvest.premium as premium


NOW = datetime.datetime(2026, 10, 5, tzinfo=datetime.timezone.utc)


class Stopped(Exception):
    pass


def _client(rows):
    client = MagicMock()
    client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = rows
    return client


def _st(logged_in=False, email=None):
    """A Streamlit mock whose st.stop() ends the script like the real one."""
    st = MagicMock(name='st')
    st.user.is_logged_in = logged_in
    st.user.email = email
    st.stop.side_effect = Stopped
    st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
    return st


@pytest.fixture
def enforced(monkeypatch):
    monkeypatch.setenv('PREMIUM_ENFORCED', 'true')


@pytest.mark.parametrize('value, expected', [
    ('true', True), ('1', True), ('ON', True), (' yes ', True),
    ('false', False), ('0', False), ('', False),
])
def test_is_enforced_reads_the_env_flag(monkeypatch, value, expected):
    monkeypatch.setenv('PREMIUM_ENFORCED', value)
    assert premium.is_enforced() is expected


def test_is_enforced_defaults_off(monkeypatch):
    monkeypatch.delenv('PREMIUM_ENFORCED', raising=False)
    assert premium.is_enforced() is False


def test_is_active_handles_lifetime_future_and_past():
    assert premium.is_active(None, now=NOW)
    assert premium.is_active('2026-11-01T00:00:00+00:00', now=NOW)
    assert not premium.is_active('2026-10-01T00:00:00+00:00', now=NOW)
    # Naive timestamps are read as UTC.
    assert not premium.is_active('2026-10-04 23:59:59', now=NOW)


def test_is_premium_checks_premium_users():
    client = _client([{'expires_at': None}])

    assert premium.is_premium('Supporter@Example.com ', client=client)
    client.table.assert_called_once_with('premium_users')
    client.table.return_value.select.return_value.eq.assert_called_once_with(
        'email', 'supporter@example.com'
    )


def test_is_premium_rejects_unknown_and_expired_emails():
    assert not premium.is_premium('nobody@example.com', client=_client([]))
    assert not premium.is_premium(
        'lapsed@example.com', client=_client([{'expires_at': '2020-01-01T00:00:00+00:00'}])
    )
    assert not premium.is_premium(None, client=_client([{'expires_at': None}]))


def test_is_premium_treats_lookup_errors_as_not_premium():
    client = MagicMock()
    client.table.side_effect = RuntimeError('database unavailable')

    assert not premium.is_premium('supporter@example.com', client=client)


def test_require_premium_does_nothing_when_not_enforced(monkeypatch):
    monkeypatch.delenv('PREMIUM_ENFORCED', raising=False)
    st = _st()
    with patch.object(premium, 'st', st):
        premium.require_premium()

    st.stop.assert_not_called()
    st.title.assert_not_called()


def test_require_premium_blocks_logged_out_visitors_with_login(enforced):
    st = _st(logged_in=False)
    teaser = MagicMock()
    with patch.object(premium, 'st', st), patch.object(premium, 'is_premium') as is_premium:
        with pytest.raises(Stopped):
            premium.require_premium(teaser=teaser)

    is_premium.assert_not_called()
    teaser.assert_called_once_with()
    st.title.assert_called_once_with('Unlock the trading lab')
    assert st.button.call_args.args[0] == 'Log in with Google'
    link_urls = [call.args[1] for call in st.link_button.call_args_list]
    assert link_urls == [premium.TRAKTEER_URL, premium.BMC_URL]
    st.info.assert_not_called()


def test_require_premium_blocks_free_users_and_shows_their_email(enforced):
    st = _st(logged_in=True, email='free@example.com')
    with patch.object(premium, 'st', st), patch.object(premium, 'is_premium', return_value=False):
        with pytest.raises(Stopped):
            premium.require_premium()

    st.title.assert_called_once_with('Unlock the trading lab')
    st.button.assert_not_called()
    assert 'free@example.com' in st.info.call_args.args[0]


def test_require_premium_lets_supporters_through(enforced):
    st = _st(logged_in=True, email='supporter@example.com')
    with patch.object(premium, 'st', st), patch.object(
        premium, '_premium_rows', return_value=[{'expires_at': '2099-01-01T00:00:00+00:00'}]
    ):
        premium.require_premium(teaser=MagicMock(side_effect=AssertionError('teaser shown')))

    st.stop.assert_not_called()
    st.title.assert_not_called()


def test_require_premium_blocks_expired_supporters(enforced):
    st = _st(logged_in=True, email='lapsed@example.com')
    with patch.object(premium, 'st', st), patch.object(
        premium, '_premium_rows', return_value=[{'expires_at': '2026-01-01T00:00:00+00:00'}]
    ):
        with pytest.raises(Stopped):
            premium.require_premium()

    st.title.assert_called_once_with('Unlock the trading lab')


def test_has_access_without_enforcement(monkeypatch):
    monkeypatch.delenv('PREMIUM_ENFORCED', raising=False)
    with patch.object(premium, 'st', _st()), patch.object(premium, 'is_premium') as is_premium:
        assert premium.has_access()
    is_premium.assert_not_called()


def test_has_access_follows_premium_status_when_enforced(enforced):
    with patch.object(premium, 'st', _st(logged_in=True, email='free@example.com')), \
            patch.object(premium, '_premium_rows', return_value=[]):
        assert not premium.has_access()
    with patch.object(premium, 'st', _st(logged_in=True, email='fan@example.com')), \
            patch.object(premium, '_premium_rows', return_value=[{'expires_at': None}]):
        assert premium.has_access()
    with patch.object(premium, 'st', _st(logged_in=False)):
        assert not premium.has_access()
