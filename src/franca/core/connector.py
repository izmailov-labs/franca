"""The `Connector`: one endpoint's HTTP face, and the only layer that knows HTTP is HTTP.

Everything above it -- adapters, the model leaf, retry and trace middleware -- reasons in
`failure_class` and `retryable` on a `ModelError`; everything below it -- the `Transport` --
moves bytes and headers and nothing else. Between the two, the `Connector` joins the URL,
encodes the JSON body, sets the authentication header, and turns an HTTP status into a
classified error, which is why the status map lives here and nowhere else.
"""

import json
import math
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import Any, assert_never

from franca.core import _json
from franca.core.adapter import WireRequest
from franca.core.clock import Clock
from franca.core.endpoint import Endpoint
from franca.core.enums import AuthScheme
from franca.core.errors import FailureClass, ModelError
from franca.core.keys import KeyProvider
from franca.core.sse import SseEvent, parse_sse
from franca.core.transport import Transport

_JSON_CONTENT_TYPE = "application/json"

# Provider error text reaches logs, spans and exception reprs. Cap it, and replace the
# credential that was just sent if the provider echoed it back.
_MAX_MESSAGE_CHARS = 500
_SCRUBBED = "<scrubbed>"


class Connector:
    """Sends an adapter's `WireRequest` to one `Endpoint` and classifies what comes back.

    The connector holds the endpoint row, a `KeyProvider`, a `Transport` and a `Clock`. It
    never stores a key: `headers()` asks the provider for one on every call, so a key
    exported after the registry was built is picked up, and nothing here can leak one
    through `repr()`. It is the one place at runtime where a `SecretStr` is unwrapped.

    The status map is the contract that lets everything above this layer stay HTTP-free:
    401 and 403 are `auth`; 429 is `rate_limit`, retryable, with `Retry-After` parsed;
    408, 409, 425 and any 5xx are `provider` and retryable; every other non-2xx is
    `provider` and not retryable; an exception from the transport is `transport` and
    retryable; and a 2xx whose body is not a JSON object is `parse`. An error status always
    wins over an undecodable body, so an HTML 503 from a proxy stays a retryable provider
    error. `adapter.error_from` may refine the result; it never has to widen it.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        *,
        keys: KeyProvider,
        transport: Transport,
        clock: Clock,
        timeout_s: float = 60.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Bind an endpoint row to the seams it needs.

        Args:
            endpoint: The row to send to: base URL, default path, provider and auth scheme.
            keys: Where the provider's API key is read from, at call time.
            transport: The byte-level HTTP implementation.
            clock: Source of wall-clock time, used only to turn an HTTP-date `Retry-After`
                into a delay in seconds.
            timeout_s: Per-request timeout handed to the transport on every call.
            extra_headers: Headers added to every request, after the endpoint's own and
                before the adapter's. Copied; the caller's mapping is not aliased.
        """
        self._endpoint = endpoint
        self._keys = keys
        self._transport = transport
        self._clock = clock
        self._timeout_s = timeout_s
        self._extra_headers: dict[str, str] = dict(extra_headers or {})

    @property
    def endpoint(self) -> Endpoint:
        """The endpoint row this connector sends to."""
        return self._endpoint

    def __repr__(self) -> str:
        """Name the endpoint and timeout only; no header value and no key ever appears."""
        return (
            f"{type(self).__name__}(endpoint={self.endpoint.id!r}, timeout_s={self._timeout_s!r})"
        )

    def headers(self, wire: WireRequest) -> dict[str, str]:
        """Assemble the complete request headers, authentication included.

        Sources merge from least to most specific: the endpoint's `extra_headers`, then
        the connector's, then the adapter's `wire.headers`. When the connector is about
        to JSON-encode `wire.body` itself and no `Content-Type` was supplied, it adds
        `application/json`. The authentication header is written last and replaces any
        same-named header, compared case-insensitively, so nothing upstream can override
        it. The key is read from the `KeyProvider` on every call and unwrapped here only.

        Args:
            wire: The request whose adapter-supplied headers take precedence.

        Returns:
            A fresh header dict ready for the transport.

        Raises:
            ModelError: With `failure_class="auth"` and `retryable=False` when the
                provider has no configured key. The message names the environment
                variable to set and never contains key material.
        """
        provider = self._endpoint.provider
        secret = self._keys.key_for(provider)
        if secret is None:
            raise ModelError(
                f"no API key for {provider}; set FRANCA_{provider.upper()}_API_KEY",
                status=None,
                provider=provider,
                retryable=False,
                failure_class="auth",
            )

        merged: dict[str, str] = {
            **self._endpoint.extra_headers,
            **self._extra_headers,
            **wire.headers,
        }
        if (
            wire.method == "POST"
            and wire.content is None
            and wire.body is not None
            and _header(merged, "content-type") is None
        ):
            merged["Content-Type"] = _JSON_CONTENT_TYPE

        key = secret.get_secret_value()
        auth = self._endpoint.auth
        match auth:
            case AuthScheme.x_api_key:
                name, value = "x-api-key", key
            case AuthScheme.bearer:
                name, value = "Authorization", f"Bearer {key}"
            case AuthScheme.x_goog_api_key:
                name, value = "x-goog-api-key", key
            case _:
                assert_never(auth)

        headers = {k: v for k, v in merged.items() if k.lower() != name.lower()}
        headers[name] = value
        return headers

    async def send(self, wire: WireRequest) -> dict[str, Any]:
        """Send one non-streaming request and return the decoded JSON object.

        `GET` goes through `Transport.get` with no body; `POST` goes through
        `Transport.post` with `wire.content` when the adapter pre-encoded it, otherwise
        `wire.body` encoded canonically by `franca.core._json.dumps`, or an empty body
        when both are `None`.

        Args:
            wire: The request as the adapter described it.

        Returns:
            The response body decoded as a JSON object.

        Raises:
            ModelError: From `headers()` when there is no key; with
                `failure_class="transport"` and `retryable=True` when the transport
                itself raised; via the status map for any non-2xx status; with
                `failure_class="parse"` when a 2xx body is not a JSON object.
        """
        url = self._endpoint.url(wire.path)
        content = _content(wire)
        headers = self.headers(wire)
        try:
            if wire.method == "GET":
                response = await self._transport.get(url, headers, timeout_s=self._timeout_s)
            else:
                response = await self._transport.post(
                    url, content, headers, timeout_s=self._timeout_s
                )
        except Exception as exc:
            raise self._transport_error(exc) from exc

        if not _is_success(response.status):
            raise self._status_error(response.status, response.headers, response.body)
        raw = _decode(response.body)
        if raw is None:
            raise ModelError(
                f"HTTP {response.status}: response body is not a JSON object",
                status=response.status,
                provider=self._endpoint.provider,
                retryable=False,
                failure_class="parse",
            )
        return raw

    def stream(self, wire: WireRequest) -> AbstractAsyncContextManager[AsyncIterator[SseEvent]]:
        """Open a streaming request and expose its server-sent events.

        The transport context is entered on `async with` and left exactly once, however
        the block exits. A non-2xx status is detected before any event is yielded: the
        body is read in full and raised through the same status map `send()` uses, so a
        429 on the streaming path carries the same `failure_class` and `retry_after_s`
        as on the non-streaming one.

        Args:
            wire: The request as the adapter described it, normally with `stream=True`.

        Returns:
            An async context manager yielding the parsed `SseEvent` iterator.
        """
        return self._stream(wire)

    @asynccontextmanager
    async def _stream(self, wire: WireRequest) -> AsyncGenerator[AsyncIterator[SseEvent]]:
        url = self._endpoint.url(wire.path)
        content = _content(wire)
        headers = self.headers(wire)
        async with AsyncExitStack() as stack:
            try:
                raw = await stack.enter_async_context(
                    self._transport.post_stream(url, content, headers, timeout_s=self._timeout_s)
                )
            except Exception as exc:
                raise self._transport_error(exc) from exc
            if not _is_success(raw.status):
                # Draining the error body is itself I/O and can fail mid-read. If it
                # does, the status is still the authoritative signal, so classify on
                # the status with no body rather than letting a raw transport
                # exception escape a path whose contract is to raise ModelError.
                try:
                    body = b"".join([line async for line in raw.lines()])
                except Exception:
                    body = b""
                raise self._status_error(raw.status, raw.headers, body)
            yield parse_sse(raw.lines())

    def _sanitize(self, text: str) -> str:
        """Make third-party text safe to put in an exception, a log line and a span.

        Three separate hazards, all observed rather than hypothetical. A provider may
        echo the credential it just rejected -- OpenAI's 401 body reads "Incorrect API
        key provided: sk-..." -- so the key that was actually sent is redacted by exact
        match, the same way cassette recording scrubs it. A message may carry newlines
        or control characters, which forge line structure in a log. And it may be
        arbitrarily long, so it is truncated.

        Args:
            text: Provider- or transport-supplied text.

        Returns:
            The text with the live key redacted, control characters collapsed to
            spaces, and the whole thing capped at `_MAX_MESSAGE_CHARS`.
        """
        secret = self._keys.key_for(self._endpoint.provider)
        if secret is not None:
            key = secret.get_secret_value()
            if key:
                text = text.replace(key, _SCRUBBED)
        text = "".join(" " if c < " " or c == "\x7f" else c for c in text)
        if len(text) > _MAX_MESSAGE_CHARS:
            text = text[:_MAX_MESSAGE_CHARS] + "..."
        return text.strip()

    def _transport_error(self, exc: Exception) -> ModelError:
        return ModelError(
            self._sanitize(f"{type(exc).__name__}: {exc}"),
            status=None,
            provider=self._endpoint.provider,
            retryable=True,
            failure_class="transport",
        )

    def _status_error(self, status: int, headers: Mapping[str, str], body: bytes) -> ModelError:
        raw = _decode(body)
        failure_class, retryable = _classify(status)
        return ModelError(
            self._sanitize(_message(raw, status)),
            status=status,
            provider=self._endpoint.provider,
            retryable=retryable,
            failure_class=failure_class,
            raw=raw,
            retry_after_s=_retry_after(headers, self._clock),
        )


type ConnectorFactory = Callable[
    [Endpoint, KeyProvider, Transport, Clock, float, Mapping[str, str]], Connector
]
"""How a registry builds connectors: positional endpoint, keys, transport, clock, timeout, headers."""


def _content(wire: WireRequest) -> bytes:
    """Return the request body bytes: pre-encoded content, canonical JSON, or empty."""
    if wire.content is not None:
        return wire.content
    if wire.body is not None:
        return _json.dumps(wire.body)
    return b""


def _is_success(status: int) -> bool:
    """Return whether `status` is in the 2xx range."""
    return 200 <= status < 300


def _classify(status: int) -> tuple[FailureClass, bool]:
    """Map a non-2xx status to its failure class and whether a retry may help."""
    if status in {401, 403}:
        return "auth", False
    if status == 429:
        return "rate_limit", True
    if status in {408, 409, 425} or status >= 500:
        return "provider", True
    return "provider", False


def _decode(body: bytes) -> dict[str, Any] | None:
    """Decode `body` as a JSON object; anything else, including a JSON array, is `None`."""
    try:
        parsed: object = json.loads(body)
    except ValueError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _message(raw: Mapping[str, Any] | None, status: int) -> str:
    """Pick the provider's error message out of its payload, or fall back to the status."""
    if raw is not None:
        error: object = raw.get("error")
        if isinstance(error, str):
            return error
        if isinstance(error, Mapping):
            message: object = error.get("message")
            if isinstance(message, str):
                return message
    return f"HTTP {status}"


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Look a header up by name, case-insensitively."""
    wanted = name.lower()
    return next((value for key, value in headers.items() if key.lower() == wanted), None)


def _retry_after(headers: Mapping[str, str], clock: Clock) -> float | None:
    """Parse `Retry-After` into a non-negative delay in seconds, or `None`.

    Accepts the delay-seconds form and the HTTP-date form. A date is turned into a delay
    against `clock.wall()`, with a naive date read as UTC, as HTTP requires; a date in the
    past clamps to zero. A value that is neither is ignored rather than raised on, because
    a malformed provider header must not turn a retryable response into a crash.
    """
    value = _header(headers, "retry-after")
    if value is None:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        return max(seconds, 0.0) if math.isfinite(seconds) else None
    try:
        when = parsedate_to_datetime(value)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        delay = when.timestamp() - clock.wall()
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    return max(delay, 0.0)
