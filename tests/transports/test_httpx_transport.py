"""Executable specification for `franca.transports.httpx`, the pooled httpx `Transport`.

No socket is ever opened here. `httpx.AsyncClient` is patched with `_MockBackend`, which
builds the *real* client around an `httpx.MockTransport`, so every request travels the
genuine httpx request path -- header merging, content encoding, timeout extensions,
streaming line decoding -- and is answered in process. Patching the class rather than
injecting a transport is what lets these tests also watch the client's lifecycle: the
backend records each client it builds, which is how laziness, reuse and closing are
asserted without reading a private attribute.

Two claims here are about imports rather than behaviour. `import httpx` inside a module
*named* httpx must resolve to the library, and `import franca` must not drag this module
(or httpx) in at all -- the second is checked in a fresh interpreter, since this one has
imported both by the time any test runs.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import warnings
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from franca.core.errors import ConfigError
from franca.core.sse import parse_sse
from franca.transports.httpx import HttpxTransport, _import_httpx

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from franca.core.transport import Transport

URL = "https://api.example.test/v1/messages"
BODY = b'{"model":"x"}'
# Captured before any fixture patches the name, so the backend can still build a real one.
_REAL_ASYNC_CLIENT = httpx.AsyncClient

_ISOLATION_PROBE = (
    "import sys\n"
    "import franca\n"
    "print(','.join(sorted(m for m in sys.modules "
    "if m == 'httpx' or m.startswith('franca.transports'))))\n"
)


def _echo(request: httpx.Request) -> httpx.Response:
    """Answer 200 with a JSON body, echoing the request method back in a header."""
    return httpx.Response(200, headers={"x-echo": request.method}, content=b'{"ok":true}')


class _MockBackend:
    """Stand-in for `httpx.AsyncClient` that answers in process and records lifecycles.

    Called wherever the transport would call the class, it builds the real client with an
    `httpx.MockTransport` wired in and keeps the client, the constructor options and every
    served request for inspection.
    """

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response] = _echo) -> None:
        """Serve every request through `handler`, which tests may replace at any time."""
        self.handler = handler
        self.requests: list[httpx.Request] = []
        self.clients: list[httpx.AsyncClient] = []
        self.options: list[dict[str, Any]] = []

    def __call__(self, **options: Any) -> httpx.AsyncClient:
        """Build a socket-free client with the options the transport chose."""
        self.options.append(options)
        client = _REAL_ASYNC_CLIENT(**options, transport=httpx.MockTransport(self._serve))
        self.clients.append(client)
        return client

    def _serve(self, request: httpx.Request) -> httpx.Response:
        """Record the request and hand it to the current handler."""
        self.requests.append(request)
        return self.handler(request)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> _MockBackend:
    """Patch `httpx.AsyncClient` so anything built during the test is socket-free."""
    mock = _MockBackend()
    monkeypatch.setattr(httpx, "AsyncClient", mock)
    return mock


@pytest.fixture
async def transport(backend: _MockBackend) -> AsyncIterator[HttpxTransport]:
    """A transport wired to the mock backend, closed however the test ends."""
    tx = HttpxTransport()
    try:
        yield tx
    finally:
        await tx.aclose()


# --------------------------------------------------------------------------------------
# The optional extra: a lazy import, and a ConfigError when it is missing.
# --------------------------------------------------------------------------------------


def test_import_resolves_to_the_library_not_this_module() -> None:
    """`import httpx` inside `franca.transports.httpx` must not import itself."""
    resolved = _import_httpx()

    assert resolved is httpx
    assert resolved.__name__ == "httpx"
    assert hasattr(resolved, "AsyncClient")


def test_construction_uses_the_real_httpx(backend: _MockBackend) -> None:
    """Building `Limits` proves the resolved module is the library, not this file."""
    HttpxTransport(max_connections=4)

    assert backend.options == []  # still nothing built: only the import ran


def test_missing_httpx_is_a_config_error_naming_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the extra installed, the failure explains the fix instead of an ImportError."""
    monkeypatch.setitem(sys.modules, "httpx", None)

    with pytest.raises(ConfigError) as excinfo:
        _import_httpx()

    assert "franca[http]" in str(excinfo.value)


def test_importing_franca_does_not_import_this_module_or_httpx() -> None:
    """Checked in a fresh interpreter: this one has imported both already."""
    probe = subprocess.run(  # noqa: S603 - fixed argv, this interpreter, no shell
        [sys.executable, "-c", _ISOLATION_PROBE],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=120,
    )

    assert probe.stdout.strip() == ""


# --------------------------------------------------------------------------------------
# Requests: bytes in, RawResponse out.
# --------------------------------------------------------------------------------------


async def test_post_returns_the_response_verbatim(
    transport: HttpxTransport, backend: _MockBackend
) -> None:
    result = await transport.post(URL, BODY, {"x-api-key": "secret"}, timeout_s=2.5)

    assert result.status == 200
    assert result.body == b'{"ok":true}'
    assert result.headers["x-echo"] == "POST"
    sent = backend.requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == URL
    assert sent.content == BODY


async def test_get_returns_the_response_verbatim(
    transport: HttpxTransport, backend: _MockBackend
) -> None:
    result = await transport.get(URL, {"x-api-key": "secret"}, timeout_s=2.5)

    assert result.status == 200
    assert result.body == b'{"ok":true}'
    assert result.headers["x-echo"] == "GET"
    assert backend.requests[0].method == "GET"


@pytest.mark.parametrize("method", ["post", "get"])
async def test_headers_round_trip(
    transport: HttpxTransport, backend: _MockBackend, method: str
) -> None:
    headers = {"x-api-key": "secret", "anthropic-version": "2023-06-01"}

    if method == "post":
        await transport.post(URL, BODY, headers, timeout_s=1.0)
    else:
        await transport.get(URL, headers, timeout_s=1.0)

    sent = backend.requests[0]
    assert sent.headers["x-api-key"] == "secret"
    assert sent.headers["anthropic-version"] == "2023-06-01"


async def test_response_headers_reach_the_raw_response(
    transport: HttpxTransport, backend: _MockBackend
) -> None:
    backend.handler = lambda _: httpx.Response(
        429, headers={"retry-after": "3", "x-request-id": "abc"}, content=b"{}"
    )

    result = await transport.post(URL, BODY, {}, timeout_s=1.0)

    assert result.headers["retry-after"] == "3"
    assert result.headers["x-request-id"] == "abc"


async def test_error_status_is_data_not_an_exception(
    transport: HttpxTransport, backend: _MockBackend
) -> None:
    backend.handler = lambda _: httpx.Response(500, content=b"<html>gateway</html>")

    result = await transport.post(URL, BODY, {}, timeout_s=1.0)

    assert result.status == 500
    assert result.body == b"<html>gateway</html>"


@pytest.mark.parametrize("method", ["post", "get"])
async def test_per_request_timeout_reaches_httpx(
    transport: HttpxTransport, backend: _MockBackend, method: str
) -> None:
    if method == "post":
        await transport.post(URL, BODY, {}, timeout_s=7.5)
    else:
        await transport.get(URL, {}, timeout_s=7.5)

    assert backend.requests[0].extensions["timeout"] == {
        "connect": 7.5,
        "read": 7.5,
        "write": 7.5,
        "pool": 7.5,
    }


async def test_httpx_exceptions_propagate_unchanged(
    transport: HttpxTransport, backend: _MockBackend
) -> None:
    """Classifying a transport failure is the Connector's job, not this layer's."""

    def _refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    backend.handler = _refuse

    with pytest.raises(httpx.ConnectError, match="connection refused"):
        await transport.post(URL, BODY, {}, timeout_s=1.0)


# --------------------------------------------------------------------------------------
# Streaming.
# --------------------------------------------------------------------------------------


async def test_post_stream_exposes_status_headers_and_lines(
    transport: HttpxTransport, backend: _MockBackend
) -> None:
    backend.handler = lambda _: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=b"event: ping\ndata: hi\n\n"
    )

    async with transport.post_stream(URL, BODY, {}, timeout_s=1.0) as stream:
        assert stream.status == 200
        assert stream.headers["content-type"] == "text/event-stream"
        lines = [line async for line in stream.lines()]

    assert lines == [b"event: ping", b"data: hi", b""]


async def test_streamed_lines_frame_as_sse(
    transport: HttpxTransport, backend: _MockBackend
) -> None:
    """The bytes this transport yields are what `parse_sse` expects, blank line included."""
    backend.handler = lambda _: httpx.Response(
        200, content=b'data: {"n":1}\n\ndata: {"n":2}\n\ndata: [DONE]\n\n'
    )

    async with transport.post_stream(URL, BODY, {}, timeout_s=1.0) as stream:
        events = [event async for event in parse_sse(stream.lines())]

    assert [event.data for event in events] == ['{"n":1}', '{"n":2}']


async def test_stream_may_be_abandoned_part_way(
    transport: HttpxTransport, backend: _MockBackend
) -> None:
    backend.handler = lambda _: httpx.Response(200, content=b"one\ntwo\nthree\n")
    first: list[bytes] = []

    async with transport.post_stream(URL, BODY, {}, timeout_s=1.0) as stream:
        async for line in stream.lines():
            first.append(line)
            break

    assert first == [b"one"]
    assert backend.requests[0].method == "POST"


# --------------------------------------------------------------------------------------
# The pooled client: lazy, single, reusable, closable.
# --------------------------------------------------------------------------------------


def test_no_client_exists_until_the_first_request(backend: _MockBackend) -> None:
    tx = HttpxTransport()

    assert tx.is_open is False
    assert backend.clients == []


async def test_the_first_request_creates_the_one_client(
    transport: HttpxTransport, backend: _MockBackend
) -> None:
    await transport.post(URL, BODY, {}, timeout_s=1.0)
    await transport.get(URL, {}, timeout_s=1.0)
    async with transport.post_stream(URL, BODY, {}, timeout_s=1.0) as stream:
        assert stream.status == 200

    assert transport.is_open is True
    assert len(backend.clients) == 1
    assert len(backend.requests) == 3


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, {}),
        ({"max_connections": 7}, {"limits": httpx.Limits(max_connections=7)}),
        ({"timeout": 12.0}, {"timeout": 12.0}),
        (
            {"limits": httpx.Limits(max_connections=3), "max_connections": 99},
            {"limits": httpx.Limits(max_connections=3)},
        ),
    ],
)
async def test_pool_options_reach_the_client(
    backend: _MockBackend, kwargs: dict[str, Any], expected: dict[str, Any]
) -> None:
    """`max_connections` is shorthand for `Limits`, and an explicit `limits` wins."""
    tx = HttpxTransport(**kwargs)

    await tx.post(URL, BODY, {}, timeout_s=1.0)
    await tx.aclose()

    assert backend.options == [expected]


async def test_aclose_closes_the_client_and_is_idempotent(backend: _MockBackend) -> None:
    tx = HttpxTransport()
    await tx.post(URL, BODY, {}, timeout_s=1.0)

    await tx.aclose()
    await tx.aclose()

    assert tx.is_open is False
    assert len(backend.clients) == 1
    assert backend.clients[0].is_closed is True


async def test_aclose_before_any_request_is_a_no_op(backend: _MockBackend) -> None:
    tx = HttpxTransport()

    await tx.aclose()

    assert backend.clients == []
    assert tx.is_open is False


async def test_a_request_after_aclose_opens_a_fresh_client(backend: _MockBackend) -> None:
    """Closing releases the pool; it does not retire the transport."""
    tx = HttpxTransport()
    await tx.post(URL, BODY, {}, timeout_s=1.0)
    await tx.aclose()

    result = await tx.post(URL, BODY, {}, timeout_s=1.0)
    await tx.aclose()

    assert result.status == 200
    assert len(backend.clients) == 2
    assert backend.clients[0] is not backend.clients[1]
    assert [client.is_closed for client in backend.clients] == [True, True]


async def test_closing_leaves_nothing_to_warn_about(backend: _MockBackend) -> None:
    """An unclosed client surfaces here: collection under warnings-as-errors is silent."""
    tx = HttpxTransport()
    await tx.post(URL, BODY, {}, timeout_s=1.0)
    await tx.aclose()
    client = backend.clients.pop()
    assert client.is_closed is True

    del client, tx
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gc.collect()

    assert [w for w in caught if issubclass(w.category, ResourceWarning)] == []


def test_repr_names_the_options_and_never_a_header(backend: _MockBackend) -> None:
    tx = HttpxTransport(max_connections=5, timeout=3.0)

    assert repr(tx) == "HttpxTransport(options=[limits, timeout], open=False)"
    assert backend.clients == []


def test_satisfies_the_transport_protocol(backend: _MockBackend) -> None:
    """Asserted by mypy: the annotation is the test, the body only keeps pytest happy."""
    tx: Transport = HttpxTransport()

    assert callable(tx.post_stream)
    assert backend.clients == []
