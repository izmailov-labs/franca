"""Executable specification for `franca.transports.scripted`.

The interesting assertions are about failure, not success: a cassette that replays is worth
little unless the message you get when it does not replay tells you which key changed. So
most of this file drives a mismatch on purpose and reads the `AssertionError`.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from franca.transports.scripted import Cassette, Exchange, ScriptedTransport

if TYPE_CHECKING:
    from pathlib import Path

    from franca.core.transport import RawResponse, Transport

CHAT_URL = "https://api.anthropic.test/v1/messages"
POLL_URL = "https://api.xai.test/v1/video/generations/vid_1"
UPLOAD_URL = "https://api.openai.test/v1/images/edits"

# Canonical form of {"model": "claude", "max_tokens": 16}: sorted keys, no whitespace.
CHAT_REQUEST = '{"max_tokens":16,"model":"claude"}'
PNG = b"\x89PNG\r\n\x1a\nnot json at all"


def _cassette() -> Cassette:
    """A two-exchange script: one chat POST, then one job-poll GET."""
    return Cassette(
        exchanges=(
            Exchange(
                method="POST",
                url=CHAT_URL,
                request=CHAT_REQUEST,
                status=200,
                headers={"content-type": "application/json"},
                body='{"id":"msg_1"}',
            ),
            Exchange(method="GET", url=POLL_URL, status=200, body='{"status":"done"}'),
        )
    )


async def _drive(transport: Transport, url: str) -> RawResponse:
    """Call through the `Transport` protocol type, so mypy checks the structural fit."""
    return await transport.get(url, {}, timeout_s=1.0)


async def test_replays_exchanges_in_order() -> None:
    transport = ScriptedTransport(_cassette())

    first = await transport.post(CHAT_URL, b'{"model":"claude","max_tokens":16}', {}, timeout_s=1.0)
    second = await transport.get(POLL_URL, {}, timeout_s=1.0)

    assert first.status == 200
    assert first.headers == {"content-type": "application/json"}
    assert first.body == b'{"id":"msg_1"}'
    assert second.body == b'{"status":"done"}'


async def test_satisfies_the_transport_protocol() -> None:
    transport = ScriptedTransport(
        Cassette(exchanges=(Exchange(method="GET", url=POLL_URL, body='{"status":"queued"}'),))
    )

    response = await _drive(transport, POLL_URL)

    assert response.body == b'{"status":"queued"}'


async def test_out_of_order_replay_is_a_method_mismatch() -> None:
    transport = ScriptedTransport(_cassette())

    with pytest.raises(AssertionError, match="expected POST"):
        await transport.get(CHAT_URL, {}, timeout_s=1.0)


async def test_url_mismatch_names_both_urls() -> None:
    transport = ScriptedTransport(_cassette())

    with pytest.raises(AssertionError) as excinfo:
        await transport.post("https://proxy.test/v1/messages", b"{}", {}, timeout_s=1.0)

    message = str(excinfo.value)
    assert CHAT_URL in message
    assert "https://proxy.test/v1/messages" in message


async def test_request_mismatch_raises_a_unified_diff() -> None:
    transport = ScriptedTransport(_cassette())

    with pytest.raises(AssertionError) as excinfo:
        await transport.post(CHAT_URL, b'{"model":"claude","max_tokens":32}', {}, timeout_s=1.0)

    message = str(excinfo.value)
    assert "--- expected" in message
    assert "+++ actual" in message
    assert '-  "max_tokens": 16' in message
    assert '+  "max_tokens": 32' in message
    # The key that did not change must not show up as a difference.
    assert '-  "model": "claude"' not in message


async def test_running_past_the_end_names_the_index() -> None:
    transport = ScriptedTransport(
        Cassette(exchanges=(Exchange(method="GET", url=POLL_URL, body="{}"),))
    )
    await transport.get(POLL_URL, {}, timeout_s=1.0)

    with pytest.raises(AssertionError, match="index 1"):
        await transport.get(POLL_URL, {}, timeout_s=1.0)


async def test_calls_capture_content_and_headers_verbatim() -> None:
    transport = ScriptedTransport(_cassette())
    headers = {"x-api-key": "sk-ant-not-a-real-key", "Content-Type": "application/json"}
    content = b'{"max_tokens":16,"model":"claude"}'

    await transport.post(CHAT_URL, content, headers, timeout_s=1.0)
    await transport.get(POLL_URL, {"accept": "application/json"}, timeout_s=1.0)

    assert transport.calls == [
        (
            "POST",
            CHAT_URL,
            content,
            {"x-api-key": "sk-ant-not-a-real-key", "Content-Type": "application/json"},
        ),
        ("GET", POLL_URL, None, {"accept": "application/json"}),
    ]
    # Recorded headers are a snapshot, not an alias of the caller's mapping.
    assert transport.calls[0][3] is not headers


async def test_a_failed_call_is_still_recorded() -> None:
    transport = ScriptedTransport(_cassette())

    with pytest.raises(AssertionError):
        await transport.post(CHAT_URL, b"{}", {"accept": "*/*"}, timeout_s=1.0)

    assert transport.calls == [("POST", CHAT_URL, b"{}", {"accept": "*/*"})]


async def test_key_order_inside_the_request_does_not_matter() -> None:
    transport = ScriptedTransport(
        Cassette(exchanges=(Exchange(method="POST", url=CHAT_URL, request='{"b":1,"a":2}'),))
    )

    response = await transport.post(CHAT_URL, b'{"a":2,"b":1}', {}, timeout_s=1.0)

    assert response.status == 200


async def test_a_non_json_body_matches_by_hash() -> None:
    digest = hashlib.sha256(PNG).hexdigest()
    transport = ScriptedTransport(
        Cassette(exchanges=(Exchange(method="POST", url=UPLOAD_URL, request=digest),))
    )

    response = await transport.post(UPLOAD_URL, PNG, {}, timeout_s=1.0)

    assert response.status == 200


async def test_a_different_non_json_body_fails_the_hash() -> None:
    digest = hashlib.sha256(PNG).hexdigest()
    transport = ScriptedTransport(
        Cassette(exchanges=(Exchange(method="POST", url=UPLOAD_URL, request=digest),))
    )

    with pytest.raises(AssertionError) as excinfo:
        await transport.post(UPLOAD_URL, PNG + b"!", {}, timeout_s=1.0)

    message = str(excinfo.value)
    assert "--- expected" in message
    assert digest in message


def test_post_stream_is_deferred_to_m1() -> None:
    transport = ScriptedTransport(_cassette())

    with pytest.raises(NotImplementedError, match="M1"):
        transport.post_stream(CHAT_URL, b"{}", {}, timeout_s=1.0)

    assert transport.exits == 0


async def test_aclose_is_a_noop() -> None:
    transport = ScriptedTransport(_cassette())

    await transport.aclose()

    assert transport.calls == []


def test_repr_shows_the_position_and_no_payload() -> None:
    assert repr(ScriptedTransport(_cassette())) == "ScriptedTransport(at=0/2)"


def test_cassette_round_trips_through_a_file(tmp_path: Path) -> None:
    cassette = _cassette()
    path = tmp_path / "chat.json"

    cassette.to_path(path)

    assert Cassette.from_path(path) == cassette
    assert path.read_text(encoding="utf-8").startswith('{\n  "exchanges"')


def test_an_empty_cassette_is_the_default() -> None:
    assert Cassette().exchanges == ()
