"""Executable specification for `franca.core.transport`: the byte-level HTTP seam.

Half of what this file asserts is checked by mypy rather than pytest. The doubles below
are assigned to `Transport`- and `RawStream`-typed names, and helpers typed against the
protocols receive them. If either protocol drifted so that a plain ``def lines`` or an
``async def lines`` generator stopped satisfying it, `make typecheck` would fail here.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from franca.core.transport import RawResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping
    from contextlib import AbstractAsyncContextManager

    from franca.core.transport import RawStream, Transport

URL = "https://api.example.test/v1/messages"
# Looks like an API key on purpose: the repr/str tests assert it never gets printed.
LEAKY_BODY = b'{"echo":"sk-ant-api03-must-never-be-printed"}'
SSE_LINES = [b"event: message_start\n", b'data: {"type":"message_start"}\n', b"\n"]


# --------------------------------------------------------------------------------------
# Doubles. Two RawStream shapes on purpose -- both must satisfy the protocol.
# --------------------------------------------------------------------------------------


async def _replay(lines: Iterable[bytes]) -> AsyncIterator[bytes]:
    """Turn a plain iterable into an async iterator, one item per step."""
    for line in lines:
        yield line


class _GeneratorStream:
    """RawStream whose `lines` is an async generator function, as the Connector spec's stub."""

    def __init__(self, response: RawResponse, lines: Iterable[bytes]) -> None:
        """Serve status and headers from `response`; replay `lines` verbatim."""
        self._response = response
        self._lines = list(lines)

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


class _IteratorStream:
    """RawStream whose `lines` is a plain method, with status/headers as bare attributes."""

    def __init__(self, status: int, headers: Mapping[str, str], lines: Iterable[bytes]) -> None:
        """Store the parts; a read-only protocol property is satisfied by an attribute."""
        self.status = status
        self.headers = headers
        self._lines = list(lines)

    def lines(self) -> AsyncIterator[bytes]:
        """Hand back an async iterator over the canned lines."""
        return _replay(self._lines)


class _FakeTransport:
    """Transport that records every call and serves one canned response."""

    def __init__(self, response: RawResponse | None = None, *, lines: Iterable[bytes] = ()) -> None:
        """Serve `response` (a 200 with an empty JSON object by default) and `lines`."""
        self.response = (
            response if response is not None else RawResponse(status=200, headers={}, body=b"{}")
        )
        self._lines = list(lines)
        self.calls: list[tuple[str, str, bytes | None, dict[str, str], float]] = []
        self.stream_exits = 0
        self.closed = False

    async def post(
        self, url: str, content: bytes, headers: Mapping[str, str], *, timeout_s: float
    ) -> RawResponse:
        """Record the POST and return the canned response."""
        self.calls.append(("POST", url, content, dict(headers), timeout_s))
        return self.response

    async def get(self, url: str, headers: Mapping[str, str], *, timeout_s: float) -> RawResponse:
        """Record the GET and return the canned response."""
        self.calls.append(("GET", url, None, dict(headers), timeout_s))
        return self.response

    def post_stream(
        self, url: str, content: bytes, headers: Mapping[str, str], *, timeout_s: float
    ) -> AbstractAsyncContextManager[_GeneratorStream]:
        """Record the POST and hand back a one-shot stream context.

        Declared to yield the concrete `_GeneratorStream`, narrower than the protocol's
        `RawStream`: the context manager type is covariant, so this must still type-check.
        """
        self.calls.append(("POST", url, content, dict(headers), timeout_s))

        @asynccontextmanager
        async def _open() -> AsyncGenerator[_GeneratorStream]:
            try:
                yield _GeneratorStream(self.response, self._lines)
            finally:
                self.stream_exits += 1

        return _open()

    async def aclose(self) -> None:
        """Mark the pool as released."""
        self.closed = True


# --------------------------------------------------------------------------------------
# Helpers typed against the protocols, never against the fakes.
# --------------------------------------------------------------------------------------


async def _post_through_protocol(transport: Transport) -> RawResponse:
    """POST through a `Transport`-typed name; handing it a fake is the mypy assertion."""
    return await transport.post(URL, b"{}", {"x-api-key": "k"}, timeout_s=1.5)


async def _drain_through_protocol(
    transport: Transport,
) -> tuple[int, Mapping[str, str], list[bytes]]:
    """Open a stream through a `Transport`-typed name and consume it inside the context."""
    async with transport.post_stream(URL, b"{}", {}, timeout_s=1.5) as stream:
        return stream.status, stream.headers, [line async for line in stream.lines()]


async def _abandon_midway(transport: Transport) -> None:
    """Leave the stream context by raising from inside `async for`."""
    async with transport.post_stream(URL, b"{}", {}, timeout_s=1.5) as stream:
        async for _ in stream.lines():
            raise RuntimeError("caller gave up")


# --------------------------------------------------------------------------------------
# RawResponse
# --------------------------------------------------------------------------------------


def test_raw_response_round_trips_status_headers_and_body() -> None:
    resp = RawResponse(status=429, headers={"Retry-After": "7"}, body=b'{"error":"slow"}')
    assert resp.status == 429
    assert resp.headers == {"Retry-After": "7"}
    assert resp.body == b'{"error":"slow"}'


def test_headers_are_copied_not_aliased() -> None:
    """A transport that reuses its header dict must not be able to mutate a past response."""
    headers = {"content-type": "application/json"}
    resp = RawResponse(status=200, headers=headers, body=b"")
    headers["x-injected"] = "later"
    assert "x-injected" not in resp.headers


def test_body_never_appears_in_repr_or_str() -> None:
    resp = RawResponse(status=200, headers={"content-type": "application/json"}, body=LEAKY_BODY)
    assert b"sk-ant-api03" in LEAKY_BODY  # the fixture really does look like a key
    assert "sk-ant-api03" not in repr(resp)
    assert "sk-ant-api03" not in str(resp)
    assert "body" not in repr(resp)
    assert "status=200" in repr(resp)


def test_body_is_still_part_of_the_serialised_model() -> None:
    """Hiding the body from repr is a logging guard, not a serialisation exclusion."""
    resp = RawResponse(status=200, headers={}, body=LEAKY_BODY)
    assert resp.model_dump()["body"] == LEAKY_BODY


@pytest.mark.parametrize(
    ("field", "value"),
    [("status", 500), ("headers", {"x": "y"}), ("body", b"tampered")],
)
def test_raw_response_is_frozen(field: str, value: object) -> None:
    resp = RawResponse(status=200, headers={}, body=b"")
    with pytest.raises(ValidationError, match="frozen"):
        setattr(resp, field, value)


def test_raw_response_is_validated_not_a_bare_record() -> None:
    with pytest.raises(ValidationError, match="status"):
        RawResponse.model_validate({"status": "two hundred", "headers": {}, "body": b""})


def test_raw_responses_compare_by_value() -> None:
    """Cassette matching relies on two identical responses being equal."""
    left = RawResponse(status=200, headers={"a": "b"}, body=b"x")
    right = RawResponse(status=200, headers={"a": "b"}, body=b"x")
    assert left == right
    assert left != RawResponse(status=200, headers={"a": "b"}, body=b"y")


# --------------------------------------------------------------------------------------
# RawStream: both `lines` shapes satisfy the protocol
# --------------------------------------------------------------------------------------


async def test_async_generator_lines_is_a_raw_stream() -> None:
    """The typed assignment is the assertion; the loop shows the shape is consumable."""
    stream: RawStream = _GeneratorStream(
        RawResponse(status=200, headers={"content-type": "text/event-stream"}, body=b""),
        SSE_LINES,
    )
    assert stream.status == 200
    assert stream.headers["content-type"] == "text/event-stream"
    assert [line async for line in stream.lines()] == SSE_LINES


async def test_plain_method_lines_with_bare_attributes_is_a_raw_stream() -> None:
    stream: RawStream = _IteratorStream(200, {"x": "y"}, SSE_LINES)
    assert stream.status == 200
    assert stream.headers == {"x": "y"}
    assert [line async for line in stream.lines()] == SSE_LINES


# --------------------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------------------


async def test_fake_is_accepted_where_transport_is_annotated() -> None:
    tp = _FakeTransport(RawResponse(status=201, headers={"x": "y"}, body=b"{}"))
    resp = await _post_through_protocol(tp)
    assert resp.status == 201
    assert tp.calls == [("POST", URL, b"{}", {"x-api-key": "k"}, 1.5)]


async def test_get_is_awaitable_and_carries_no_body() -> None:
    tp = _FakeTransport()
    transport: Transport = tp
    resp = await transport.get(URL, {"accept": "application/json"}, timeout_s=2.0)
    assert resp.status == 200
    assert tp.calls == [("GET", URL, None, {"accept": "application/json"}, 2.0)]


async def test_post_stream_is_an_async_context_manager_over_a_raw_stream() -> None:
    tp = _FakeTransport(
        RawResponse(status=200, headers={"content-type": "text/event-stream"}, body=b""),
        lines=SSE_LINES,
    )
    status, headers, lines = await _drain_through_protocol(tp)
    assert status == 200
    assert headers == {"content-type": "text/event-stream"}
    assert lines == SSE_LINES
    assert tp.calls == [("POST", URL, b"{}", {}, 1.5)]
    assert tp.stream_exits == 1


async def test_stream_context_is_left_even_when_the_consumer_raises() -> None:
    """The contract that lets a caller's deadline close the socket: exit runs on any path."""
    tp = _FakeTransport(lines=SSE_LINES)
    with pytest.raises(RuntimeError, match="gave up"):
        await _abandon_midway(tp)
    assert tp.stream_exits == 1


async def test_aclose_is_awaitable() -> None:
    tp = _FakeTransport()
    transport: Transport = tp
    await transport.aclose()
    assert tp.closed is True
