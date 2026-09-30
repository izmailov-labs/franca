"""`HttpxTransport`: the production `Transport`, one pooled `AsyncClient` behind the seam.

httpx is an optional extra (`franca[http]`), so this is the only module in franca that
touches it, and the import runs inside `HttpxTransport.__init__` rather than at module
scope. `pip install franca` therefore stays usable end to end -- `franca`, `franca.core`
and every offline path import cleanly without httpx; only constructing this transport
needs the extra, and a missing httpx surfaces as a `ConfigError` naming the command that
fixes it, not as an `ImportError` from a stack frame the caller never asked about.

This module's own name shadows the third-party package. That is safe: Python 3 imports
are absolute, so `import httpx` here resolves through `sys.path` to the installed library
and never to this file. `test_httpx_transport.py` asserts that rather than trusting it.

One transport owns exactly one `httpx.AsyncClient`, created on first use and closed by
`aclose()`. The pool is franca's fan-out control -- an eval runner opens many trials at
once and `limits` / `max_connections` are what bound the concurrent sockets per process --
and it is reachable from configuration as well as from code, because
`Settings.transport_options` is passed here as keyword arguments. Per-provider pacing is
the rate-limit layer's job, not the pool's.

Nothing here interprets a response. A 500 comes back as a `RawResponse` with `status=500`,
and any httpx exception propagates unchanged: classifying either one is the `Connector`'s
job, and duplicating that judgement at the socket would put it in two places.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from franca.core.errors import ConfigError
from franca.core.transport import RawResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Mapping
    from types import ModuleType

    import httpx

_INSTALL_HINT = "HttpxTransport requires httpx; install it with: pip install 'franca[http]'"


def _import_httpx() -> ModuleType:
    """Import the installed httpx, translating its absence into a `ConfigError`.

    Kept as a function so the import happens per construction rather than at module
    import, and so the shadowing test can assert what the name actually resolves to.

    Returns:
        The third-party `httpx` module -- resolved absolutely, never this module.

    Raises:
        ConfigError: When httpx is not installed. The message names the extra to install.
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - only reachable without the extra
        raise ConfigError(_INSTALL_HINT) from exc
    return httpx


def _raw(response: httpx.Response) -> RawResponse:
    """Convert a fully read httpx response into the transport-level `RawResponse`."""
    return RawResponse(
        status=response.status_code,
        headers=dict(response.headers),
        body=response.content,
    )


class _HttpxStream:
    """`RawStream` view of an httpx streaming response, valid inside its context only.

    Holds nothing of its own: status, headers and lines are read straight from the open
    response, so leaving the `post_stream` context -- normally, by exception or by
    cancellation -- releases the connection and ends this object's usefulness with it.
    """

    __slots__ = ("_response",)

    def __init__(self, response: httpx.Response) -> None:
        """Wrap an already-open streaming response.

        Args:
            response: The response yielded by `httpx.AsyncClient.stream`.
        """
        self._response = response

    @property
    def status(self) -> int:
        """HTTP status code, readable before the first line is consumed."""
        return self._response.status_code

    @property
    def headers(self) -> Mapping[str, str]:
        """Response headers: httpx's own case-insensitive mapping, not a copy of it."""
        return self._response.headers

    async def lines(self) -> AsyncIterator[bytes]:
        """Iterate the body one line at a time, re-encoded as UTF-8 bytes.

        httpx decodes the stream to text and splits it with `str.splitlines` semantics, so
        the line endings are already gone and an SSE separator arrives as `b""`.
        `parse_sse` accepts lines with or without their endings, so both shapes frame
        identically.

        Yields:
            One line of the body per item, without its line ending.
        """
        async for line in self._response.aiter_lines():
            yield line.encode()


class HttpxTransport:
    """The production `Transport`: one pooled `httpx.AsyncClient`, created on first use.

    The client is deliberately not built in `__init__`. Constructing a transport is a
    configuration step that may happen far from an event loop -- at import time, in a
    settings factory, inside a registry that is never used -- while an `AsyncClient` binds
    itself to the loop that first drives it. Deferring creation to the first request keeps
    construction free of that commitment; `aclose()` undoes it again.

    Every method receives the fully joined URL, an already-encoded body and the complete
    headers, and each request passes its own `timeout_s` to httpx, overriding the client
    default. Statuses are data, exceptions are httpx's own: neither is interpreted here.
    """

    def __init__(
        self,
        *,
        limits: object | None = None,
        timeout: object | None = None,
        max_connections: int | None = None,
    ) -> None:
        """Resolve httpx and record the pool options; the client is built on first use.

        Args:
            limits: An `httpx.Limits`, handed to the client verbatim. Typed as `object`
                so this signature carries no import of an optional dependency.
            timeout: The client's default timeout, an `httpx.Timeout` or a float. Every
                request made through this transport passes its own `timeout_s`, which
                overrides this value; it therefore only matters to a caller who reaches
                past the `Transport` protocol.
            max_connections: Shorthand for `httpx.Limits(max_connections=...)`, applied
                only when `limits` was not given -- an explicit `limits` always wins.

        Raises:
            ConfigError: When httpx is not installed.
        """
        httpx_module = _import_httpx()
        if limits is None and max_connections is not None:
            limits = httpx_module.Limits(max_connections=max_connections)
        options: dict[str, Any] = {}
        if limits is not None:
            options["limits"] = limits
        if timeout is not None:
            options["timeout"] = timeout
        self._httpx: ModuleType = httpx_module
        self._options = options
        self._client: httpx.AsyncClient | None = None

    def __repr__(self) -> str:
        """Name the class, the configured option keys and whether a client is open."""
        options = ", ".join(sorted(self._options))
        return f"{type(self).__name__}(options=[{options}], open={self.is_open})"

    @property
    def is_open(self) -> bool:
        """Whether a pooled client exists right now.

        `False` after construction and again after `aclose()`; `True` from the first
        request until the next `aclose()`.
        """
        return self._client is not None

    def _pooled_client(self) -> httpx.AsyncClient:
        """Return the one pooled client, creating it if this is the first use."""
        client = self._client
        if client is None:
            created: httpx.AsyncClient = self._httpx.AsyncClient(**self._options)
            self._client = created
            return created
        return client

    async def post(
        self, url: str, content: bytes, headers: Mapping[str, str], *, timeout_s: float
    ) -> RawResponse:
        """Send a POST and read the whole response.

        Args:
            url: Absolute URL, already joined from the endpoint's base URL and path.
            content: Request body, already encoded.
            headers: Complete request headers, authentication included.
            timeout_s: Per-request timeout in seconds, overriding the client default.

        Returns:
            The full response, whatever its status.
        """
        response = await self._pooled_client().post(
            url, content=content, headers=headers, timeout=timeout_s
        )
        return _raw(response)

    async def get(self, url: str, headers: Mapping[str, str], *, timeout_s: float) -> RawResponse:
        """Send a GET -- the job-poll path -- and read the whole response.

        Args:
            url: Absolute URL, already joined from the endpoint's base URL and path.
            headers: Complete request headers, authentication included.
            timeout_s: Per-request timeout in seconds, overriding the client default.

        Returns:
            The full response, whatever its status.
        """
        response = await self._pooled_client().get(url, headers=headers, timeout=timeout_s)
        return _raw(response)

    @asynccontextmanager
    async def post_stream(
        self, url: str, content: bytes, headers: Mapping[str, str], *, timeout_s: float
    ) -> AsyncGenerator[_HttpxStream]:
        """Send a POST and open the response as a stream.

        The request goes out when the context is entered and the connection is released
        when it is left, however that happens -- return, exception or cancellation. Status
        and headers are readable immediately, so a caller can reject a non-2xx before
        consuming a line.

        Args:
            url: Absolute URL, already joined from the endpoint's base URL and path.
            content: Request body, already encoded.
            headers: Complete request headers, authentication included.
            timeout_s: Per-request timeout in seconds, applied to every phase.

        Yields:
            The open stream, for as long as the context is held.
        """
        async with self._pooled_client().stream(
            "POST", url, content=content, headers=headers, timeout=timeout_s
        ) as response:
            yield _HttpxStream(response)

    async def aclose(self) -> None:
        """Close the pooled client, if one was ever created.

        Releasing the pool is not a terminal state: the transport stays usable, and a
        later request simply builds a fresh client. That choice is deliberate. It makes
        `aclose()` idempotent -- a second call, or a call on a transport that never sent
        anything, does nothing -- and it makes a transport safe to carry across
        independent event loops, since an `AsyncClient` belongs to the loop that drove it
        and dropping it at the end of one run is exactly what the next run needs.
        """
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()
