"""Executable specification for `franca.core.connector.Connector` (M0, plus two M1 stream cases).

Written before the implementation. Every double lives in this file, so the module runs on
its own with nothing from `franca.testing` (which is also unwritten):

    uv run pytest tests/core/test_connector.py
    uv run python tests/core/test_connector.py

The module skips cleanly until `core/connector.py` exists; `-ra` in addopts keeps the skip
reason on screen so it cannot be quietly forgotten.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from email.utils import formatdate
from typing import TYPE_CHECKING, Any

import pytest

if __name__ == "__main__":  # `python test_connector.py` re-enters under pytest
    raise SystemExit(pytest.main([__file__, "-ra"]))

pytest.importorskip("franca.core.connector", reason="M0: Connector is not implemented yet")

from pydantic import SecretStr

from franca.core.adapter import WireRequest
from franca.core.connector import Connector
from franca.core.endpoint import Endpoint
from franca.core.enums import AuthScheme
from franca.core.errors import ModelError
from franca.core.ids import ANTHROPIC_MESSAGES, CHAT, Provider
from franca.core.transport import RawResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping
    from contextlib import AbstractAsyncContextManager

ANTHROPIC = Provider("anthropic")
WALL_T0 = 1_000_000.0

# --------------------------------------------------------------------------------------
# Doubles. Deliberately not ScriptedTransport/FakeClock: this file must not wait on M1.
# --------------------------------------------------------------------------------------


class _StubTransport:
    """Transport returning a canned response and recording what it was handed."""

    def __init__(
        self,
        response: RawResponse | None = None,
        *,
        raises: Exception | None = None,
        sse_lines: Iterable[bytes] = (),
    ) -> None:
        """Store the canned outcome; `raises` wins over `response`."""
        self._response = response or RawResponse(status=200, headers={}, body=b"{}")
        self._raises = raises
        self._sse_lines = list(sse_lines)
        self.calls: list[tuple[str, str, bytes | None, Mapping[str, str], float]] = []
        self.exits = 0

    async def post(
        self, url: str, content: bytes, headers: Mapping[str, str], *, timeout_s: float
    ) -> RawResponse:
        """Record the POST and return the canned response."""
        self.calls.append(("POST", url, content, dict(headers), timeout_s))
        if self._raises is not None:
            raise self._raises
        return self._response

    async def get(self, url: str, headers: Mapping[str, str], *, timeout_s: float) -> RawResponse:
        """Record the GET and return the canned response."""
        self.calls.append(("GET", url, None, dict(headers), timeout_s))
        if self._raises is not None:
            raise self._raises
        return self._response

    def post_stream(
        self, url: str, content: bytes, headers: Mapping[str, str], *, timeout_s: float
    ) -> AbstractAsyncContextManager[_StubStream]:
        """Record the POST and hand back a one-shot stream context."""
        self.calls.append(("POST", url, content, dict(headers), timeout_s))

        @asynccontextmanager
        async def _cm() -> AsyncGenerator[_StubStream]:
            try:
                yield _StubStream(self._response, self._sse_lines)
            finally:
                self.exits += 1

        return _cm()

    async def aclose(self) -> None:
        """No pooled resources to release."""


class _StubStream:
    """RawStream over a fixed list of already-framed SSE lines."""

    def __init__(self, response: RawResponse, lines: list[bytes]) -> None:
        """Take status and headers from `response`; replay `lines` verbatim."""
        self._response = response
        self._lines = lines

    @property
    def status(self) -> int:
        """HTTP status of the streaming response."""
        return self._response.status

    @property
    def headers(self) -> Mapping[str, str]:
        """Response headers."""
        return self._response.headers

    async def lines(self) -> AsyncIterator[bytes]:
        """Yield the canned lines."""
        for line in self._lines:
            yield line

    async def read(self) -> bytes:
        """Body of a non-2xx streaming response, read before any event is yielded."""
        return self._response.body


class _FakeClock:
    """Clock with a fixed wall time, so Retry-After HTTP-date arithmetic is exact."""

    def __init__(self, wall: float = WALL_T0) -> None:
        """Freeze `wall()` at the given epoch seconds."""
        self._wall = wall
        self.slept: list[float] = []

    def monotonic(self) -> float:
        """Monotonic reading; unused by the Connector but part of the protocol."""
        return 0.0

    def wall(self) -> float:
        """Frozen wall-clock reading."""
        return self._wall

    async def sleep(self, seconds: float) -> None:
        """Record instead of sleeping."""
        self.slept.append(seconds)


class _Keys:
    """KeyProvider over a mutable mapping, to prove keys are read at call time."""

    def __init__(self, mapping: dict[Provider, SecretStr] | None = None) -> None:
        """Start from `mapping`; callers may mutate `.mapping` between calls."""
        self.mapping = mapping if mapping is not None else {ANTHROPIC: SecretStr("sk-secret")}

    def key_for(self, provider: Provider) -> SecretStr | None:
        """Return the current key for `provider`, or None."""
        return self.mapping.get(provider)


def _endpoint(
    *,
    auth: AuthScheme = AuthScheme.x_api_key,
    extra_headers: dict[str, str] | None = None,
) -> Endpoint:
    """Build the Anthropic messages endpoint row with an overridable auth scheme."""
    return Endpoint(
        id="anthropic/chat/messages",
        provider=ANTHROPIC,
        capability=CHAT,
        dialect=ANTHROPIC_MESSAGES,
        base_url="https://api.anthropic.com",
        path="/v1/messages",
        auth=auth,
        extra_headers=extra_headers
        if extra_headers is not None
        else {"anthropic-version": "2023-06-01"},
    )


def _connector(
    transport: _StubTransport,
    *,
    endpoint: Endpoint | None = None,
    keys: _Keys | None = None,
    clock: _FakeClock | None = None,
    timeout_s: float = 60.0,
    extra_headers: Mapping[str, str] | None = None,
) -> Connector:
    """Assemble a Connector over the doubles above."""
    return Connector(
        endpoint or _endpoint(),
        keys=keys or _Keys(),
        transport=transport,
        clock=clock or _FakeClock(),
        timeout_s=timeout_s,
        extra_headers=extra_headers,
    )


async def _drain(cm: AbstractAsyncContextManager[AsyncIterator[Any]], sink: list[Any]) -> None:
    """Consume a stream into `sink`, so `pytest.raises` can wrap a single statement."""
    async with cm as events:
        sink.extend([event async for event in events])


def _ok(payload: dict[str, Any]) -> RawResponse:
    """A 200 carrying `payload` as JSON."""
    return RawResponse(status=200, headers={}, body=json.dumps(payload).encode())


# --------------------------------------------------------------------------------------
# headers()
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("auth", "header"),
    [
        (AuthScheme.x_api_key, "x-api-key"),
        (AuthScheme.bearer, "Authorization"),
        (AuthScheme.x_goog_api_key, "x-goog-api-key"),
    ],
)
def test_auth_scheme_decides_which_header_carries_the_key(auth: AuthScheme, header: str) -> None:
    conn = _connector(_StubTransport(), endpoint=_endpoint(auth=auth))
    headers = conn.headers(WireRequest(body={}))
    expected = "Bearer sk-secret" if auth is AuthScheme.bearer else "sk-secret"
    assert headers[header] == expected


def test_endpoint_extra_headers_are_sent() -> None:
    conn = _connector(_StubTransport())
    assert conn.headers(WireRequest(body={}))["anthropic-version"] == "2023-06-01"


def test_header_precedence_is_endpoint_then_connector_then_wire() -> None:
    """Most specific source wins: the adapter's own header beats configuration."""
    conn = _connector(
        _StubTransport(),
        endpoint=_endpoint(extra_headers={"x-tier": "endpoint", "x-ep": "1"}),
        extra_headers={"x-tier": "connector", "x-conn": "1"},
    )
    headers = conn.headers(WireRequest(body={}, headers={"x-tier": "wire"}))
    assert headers["x-tier"] == "wire"
    assert headers["x-ep"] == "1"
    assert headers["x-conn"] == "1"


def test_missing_key_raises_auth_error_naming_the_env_var() -> None:
    conn = _connector(_StubTransport(), keys=_Keys({}))
    with pytest.raises(ModelError, match=r"FRANCA_ANTHROPIC_API_KEY") as exc:
        conn.headers(WireRequest(body={}))
    assert exc.value.failure_class == "auth"
    assert exc.value.retryable is False
    assert exc.value.status is None


def test_key_is_read_at_call_time_not_at_construction() -> None:
    """A KeyProvider backed by env vars must see a key exported after the registry is built."""
    keys = _Keys({})
    conn = _connector(_StubTransport(), keys=keys)
    keys.mapping[ANTHROPIC] = SecretStr("sk-late")
    assert conn.headers(WireRequest(body={}))["x-api-key"] == "sk-late"


def test_the_key_never_appears_in_repr_or_in_the_auth_error() -> None:
    conn = _connector(_StubTransport())
    assert "sk-secret" not in repr(conn)
    missing = _connector(_StubTransport(), keys=_Keys({}))
    with pytest.raises(ModelError) as exc:
        missing.headers(WireRequest(body={}))
    assert "sk-secret" not in str(exc.value)


# --------------------------------------------------------------------------------------
# send(): encoding, URL, method
# --------------------------------------------------------------------------------------


async def test_body_is_encoded_with_sorted_keys_and_compact_separators() -> None:
    tp = _StubTransport(_ok({}))
    await _connector(tp).send(WireRequest(body={"b": 1, "a": {"d": 2, "c": 3}}))
    assert tp.calls[0][2] == b'{"a":{"c":3,"d":2},"b":1}'


async def test_two_identical_builds_are_byte_equal() -> None:
    """The mechanical basis of cassette matching: same wire in, same bytes out."""
    tp = _StubTransport(_ok({}))
    conn = _connector(tp)
    wire = WireRequest(body={"z": 1, "a": 2})
    await conn.send(wire)
    await conn.send(wire)
    assert tp.calls[0][2] == tp.calls[1][2]


async def test_non_ascii_is_sent_as_utf8_not_escaped() -> None:
    tp = _StubTransport(_ok({}))
    await _connector(tp).send(WireRequest(body={"t": "café"}))
    assert tp.calls[0][2] == '{"t":"café"}'.encode()


async def test_pre_encoded_content_wins_over_body() -> None:
    """Multipart image edits arrive already encoded; the Connector must not re-serialise."""
    tp = _StubTransport(_ok({}))
    await _connector(tp).send(WireRequest(body={"ignored": True}, content=b"--boundary--"))
    assert tp.calls[0][2] == b"--boundary--"


async def test_wire_path_overrides_the_endpoint_path() -> None:
    """Adapters pick Responses vs Chat, generations vs edits, by setting `path`."""
    tp = _StubTransport(_ok({}))
    await _connector(tp).send(WireRequest(body={}, path="/v1/responses"))
    assert tp.calls[0][1] == "https://api.anthropic.com/v1/responses"


async def test_none_path_falls_back_to_the_endpoint_path() -> None:
    tp = _StubTransport(_ok({}))
    await _connector(tp).send(WireRequest(body={}))
    assert tp.calls[0][1] == "https://api.anthropic.com/v1/messages"


async def test_get_uses_transport_get_and_sends_no_body() -> None:
    """The job-poll path is a GET; it must not smuggle a body through post()."""
    tp = _StubTransport(_ok({}))
    await _connector(tp).send(WireRequest(method="GET", path="/v1/jobs/abc"))
    assert tp.calls[0][0] == "GET"
    assert tp.calls[0][2] is None


async def test_timeout_is_passed_through_to_the_transport() -> None:
    tp = _StubTransport(_ok({}))
    await _connector(tp, timeout_s=12.5).send(WireRequest(body={}))
    assert tp.calls[0][4] == 12.5


async def test_success_returns_the_parsed_json_body() -> None:
    tp = _StubTransport(_ok({"id": "msg_1", "content": []}))
    assert await _connector(tp).send(WireRequest(body={})) == {"id": "msg_1", "content": []}


# --------------------------------------------------------------------------------------
# send(): the status map. M0 acceptance is "Connector maps every status class".
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "failure_class", "retryable"),
    [
        (401, "auth", False),
        (403, "auth", False),
        (429, "rate_limit", True),
        (408, "provider", True),
        (409, "provider", True),
        (425, "provider", True),
        (500, "provider", True),
        (502, "provider", True),
        (503, "provider", True),
        (400, "provider", False),
        (404, "provider", False),
        (422, "provider", False),
    ],
)
async def test_status_maps_to_failure_class_and_retryability(
    status: int, failure_class: str, retryable: bool
) -> None:
    tp = _StubTransport(RawResponse(status=status, headers={}, body=b'{"error":{"message":"x"}}'))
    with pytest.raises(ModelError) as exc:
        await _connector(tp).send(WireRequest(body={}))
    assert exc.value.status == status
    assert exc.value.failure_class == failure_class
    assert exc.value.retryable is retryable
    assert exc.value.provider == ANTHROPIC


async def test_retry_after_in_seconds_is_parsed() -> None:
    tp = _StubTransport(RawResponse(status=429, headers={"retry-after": "30"}, body=b"{}"))
    with pytest.raises(ModelError) as exc:
        await _connector(tp).send(WireRequest(body={}))
    assert exc.value.retry_after_s == 30.0


async def test_retry_after_as_http_date_is_parsed_against_the_wall_clock() -> None:
    """The only reason Connector needs `clock.wall()`: dates need real calendar time."""
    when = formatdate(WALL_T0 + 30.0, usegmt=True)
    tp = _StubTransport(RawResponse(status=429, headers={"retry-after": when}, body=b"{}"))
    with pytest.raises(ModelError) as exc:
        await _connector(tp, clock=_FakeClock(WALL_T0)).send(WireRequest(body={}))
    assert exc.value.retry_after_s == pytest.approx(30.0, abs=1.0)


async def test_retry_after_in_the_past_clamps_to_zero() -> None:
    when = formatdate(WALL_T0 - 300.0, usegmt=True)
    tp = _StubTransport(RawResponse(status=429, headers={"retry-after": when}, body=b"{}"))
    with pytest.raises(ModelError) as exc:
        await _connector(tp, clock=_FakeClock(WALL_T0)).send(WireRequest(body={}))
    assert exc.value.retry_after_s == 0.0


async def test_unparseable_retry_after_is_ignored_not_fatal() -> None:
    """A malformed provider header must not turn a retryable 429 into a crash."""
    tp = _StubTransport(RawResponse(status=429, headers={"retry-after": "soon"}, body=b"{}"))
    with pytest.raises(ModelError) as exc:
        await _connector(tp).send(WireRequest(body={}))
    assert exc.value.retry_after_s is None
    assert exc.value.retryable is True


async def test_retry_after_header_is_matched_case_insensitively() -> None:
    tp = _StubTransport(RawResponse(status=429, headers={"Retry-After": "5"}, body=b"{}"))
    with pytest.raises(ModelError) as exc:
        await _connector(tp).send(WireRequest(body={}))
    assert exc.value.retry_after_s == 5.0


async def test_transport_exception_becomes_a_retryable_transport_error() -> None:
    tp = _StubTransport(raises=OSError("connection reset"))
    with pytest.raises(ModelError) as exc:
        await _connector(tp).send(WireRequest(body={}))
    assert exc.value.failure_class == "transport"
    assert exc.value.retryable is True
    assert exc.value.status is None


async def test_undecodable_json_on_a_2xx_becomes_a_parse_error() -> None:
    tp = _StubTransport(RawResponse(status=200, headers={}, body=b"<html>502 Bad Gateway</html>"))
    with pytest.raises(ModelError) as exc:
        await _connector(tp).send(WireRequest(body={}))
    assert exc.value.failure_class == "parse"


async def test_undecodable_json_on_an_error_status_keeps_the_status_mapping() -> None:
    """An HTML 503 from a proxy is still a retryable provider error, not a parse error."""
    tp = _StubTransport(RawResponse(status=503, headers={}, body=b"<html>upstream</html>"))
    with pytest.raises(ModelError) as exc:
        await _connector(tp).send(WireRequest(body={}))
    assert exc.value.failure_class == "provider"
    assert exc.value.retryable is True


async def test_the_error_body_is_preserved_on_raw() -> None:
    """`adapter.error_from` refines the error, so it needs the provider's own payload."""
    body = b'{"error":{"type":"overloaded_error","message":"Overloaded"}}'
    tp = _StubTransport(RawResponse(status=529, headers={}, body=body))
    with pytest.raises(ModelError) as exc:
        await _connector(tp).send(WireRequest(body={}))
    assert exc.value.raw == {"error": {"type": "overloaded_error", "message": "Overloaded"}}


# --------------------------------------------------------------------------------------
# stream(): M1. Kept here because it is the same status map on a different path.
# --------------------------------------------------------------------------------------


async def test_stream_raises_before_yielding_any_event_on_an_error_status() -> None:
    """A 429 arrives as a normal response with a JSON body, not as an SSE error event."""
    tp = _StubTransport(
        RawResponse(
            status=429, headers={"retry-after": "7"}, body=b'{"error":{"message":"slow down"}}'
        ),
        sse_lines=[b"data: never\n", b"\n"],
    )
    seen: list[object] = []
    with pytest.raises(ModelError) as exc:
        await _drain(_connector(tp).stream(WireRequest(body={}, stream=True)), seen)
    assert seen == []
    assert exc.value.failure_class == "rate_limit"
    assert exc.value.retry_after_s == 7.0


async def test_stream_yields_parsed_sse_events_on_success() -> None:
    tp = _StubTransport(
        RawResponse(status=200, headers={}, body=b""),
        sse_lines=[b"event: message_start\n", b'data: {"type":"message_start"}\n', b"\n"],
    )
    async with _connector(tp).stream(WireRequest(body={}, stream=True)) as events:
        received = [event async for event in events]
    assert [e.event for e in received] == ["message_start"]
    assert tp.exits == 1
