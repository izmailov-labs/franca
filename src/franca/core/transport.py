"""The byte-level HTTP seam: the `Transport` protocol and the raw types it hands back.

Byte-level on purpose. A transport moves bytes and headers and nothing else: JSON
encoding and decoding live in the `Connector`, SSE framing lives in `core.sse`, and
status-to-error mapping happens above both. Keeping the seam this thin means every
implementation -- `HttpxTransport` in production, `ScriptedTransport` under test -- stays
small, and a cassette can replay exactly what crossed the wire without re-encoding it.

`RawStream` and `Transport` are structural protocols. A test double satisfies them by
shape alone; nothing here is meant to be subclassed.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from pydantic import BaseModel, Field


class RawResponse(BaseModel, frozen=True):
    """A complete, non-streaming HTTP response exactly as the transport received it.

    `body` is excluded from `repr` and `str`. Response bodies can echo request content,
    and request content can hold secrets, so the bytes must never reach a log line by
    accident. They are still present in `model_dump()`, deliberately: cassette recording
    needs them.

    Frozen is shallow. The model rejects attribute assignment, but `headers` is an
    ordinary `dict` and is not made read-only.

    Attributes:
        status: HTTP status code. Non-2xx values are data here, not exceptions; the
            `Connector` decides what they mean.
        headers: Response headers as delivered. Header-name case is whatever the
            transport produced; nothing is normalised at this layer, so lookups above it
            must be case-insensitive.
        body: The response body, verbatim and undecoded.
    """

    status: int
    headers: dict[str, str]
    body: bytes = Field(repr=False)


class RawStream(Protocol):
    """A streaming HTTP response: status and headers up front, the body as a line feed.

    Produced by `Transport.post_stream` and only valid inside that context, so the
    connection is released when the caller leaves the `async with` block -- including on
    cancellation, which is what lets a caller's deadline close the socket.
    """

    @property
    def status(self) -> int:
        """HTTP status code, readable before the first line is consumed."""

    @property
    def headers(self) -> Mapping[str, str]:
        """Response headers, readable before the first line is consumed."""

    def lines(self) -> AsyncIterator[bytes]:
        """Iterate the body one raw line at a time.

        Lines are split at newline and delivered as bytes, undecoded. A trailing LF or
        CRLF may still be attached; the SSE parser normalises it, so a transport need not.
        Implementations may write this as a plain method returning an async iterator or
        as an async generator function -- both shapes satisfy the protocol.
        """


class Transport(Protocol):
    """The HTTP seam: the only place franca touches a socket.

    An implementation owns connection pooling, per-request timeouts and TLS, and nothing
    more. Every method receives the fully joined URL, an already-encoded body and the
    complete header mapping -- authentication included -- and answers in bytes. No JSON,
    no retries, no interpretation of status codes: a 500 comes back as a `RawResponse`
    with `status=500`, and only a transport-level failure (connection refused, TLS error,
    timeout) surfaces as an exception, which the `Connector` classifies as retryable.

    `HttpxTransport` is the production implementation; `ScriptedTransport` replays
    cassettes in tests. Both are injected into the code that calls them, never imported.
    """

    async def post(
        self, url: str, content: bytes, headers: Mapping[str, str], *, timeout_s: float
    ) -> RawResponse:
        """Send a POST and read the whole response.

        Args:
            url: Absolute URL, already joined from the endpoint's base URL and path.
            content: Request body, already encoded.
            headers: Complete request headers, authentication included.
            timeout_s: Per-request timeout in seconds. A deadline spanning several
                requests is the caller's business, not the transport's.

        Returns:
            The full response, whatever its status.
        """

    async def get(self, url: str, headers: Mapping[str, str], *, timeout_s: float) -> RawResponse:
        """Send a GET -- the job-poll path -- and read the whole response.

        Args:
            url: Absolute URL, already joined from the endpoint's base URL and path.
            headers: Complete request headers, authentication included.
            timeout_s: Per-request timeout in seconds.

        Returns:
            The full response, whatever its status.
        """

    def post_stream(
        self, url: str, content: bytes, headers: Mapping[str, str], *, timeout_s: float
    ) -> AbstractAsyncContextManager[RawStream]:
        """Send a POST and open the response as a stream.

        A plain method, not a coroutine: the request goes out when the returned context
        is entered, and the connection is released when it is left, however that happens.
        Status and headers are readable as soon as the context is entered, so a caller
        can reject a non-2xx before consuming a single line.

        Args:
            url: Absolute URL, already joined from the endpoint's base URL and path.
            content: Request body, already encoded.
            headers: Complete request headers, authentication included.
            timeout_s: Per-request timeout in seconds, applied by the transport. How it
                is split between connecting and reading is the implementation's choice.

        Returns:
            An async context manager that yields the open `RawStream`.
        """

    async def aclose(self) -> None:
        """Release pooled resources.

        `HttpxTransport` closes the `AsyncClient` it created on first use; a transport
        that holds no pooled resources implements this as a no-op.
        """
