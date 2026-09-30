"""The adapter seam: what an adapter hands the `Connector` to put on the wire.

An adapter turns a capability request into a `WireRequest` and turns the raw
provider reply back into the capability's response type. This module holds the
wire-facing half of that contract: `WireRequest`, the `Adapter` base class and its
streaming refinement. `JobAdapter` lands with the job path, which needs `Job` polling
plumbing that does not exist yet.

`WireRequest` is deliberately dumb. It carries no endpoint, no key and no encoded
bytes unless the adapter built them itself; the `Connector` resolves a `None` path
to the endpoint's default, JSON-encodes `body`, prefers `content` when both are
set, and merges `headers` over its own. Keeping the model inert is what lets it be
constructed and inspected in tests without a transport.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from franca.core.errors import ModelError
from franca.core.ids import Capability, Dialect, Provider
from franca.core.profile import BaseProfile
from franca.core.sse import SseEvent


class WireRequest(BaseModel, frozen=True):
    """One HTTP request as an adapter describes it, before the `Connector` sends it.

    Attributes:
        method: HTTP verb. `POST` for every generation call; `GET` for job polls.
        path: Request path relative to the endpoint's base URL. `None` means "use
            the endpoint's own path"; adapters override it to pick Responses over
            Chat, or `edits` over `generations`.
        body: JSON payload. The `Connector` encodes it; the model never does.
        content: Pre-encoded payload such as a multipart form. Wins over `body` when
            both are set, and is excluded from `repr` because it may embed uploads.
        headers: Extra request headers, e.g. the multipart `Content-Type` with its
            boundary. Merged over the `Connector`'s own headers.
        stream: Whether the adapter shaped the request for server-sent events.
        unverified: Names of profile flags the adapter applied while their value was
            `None` (neither confirmed nor denied). The leaf unions them into
            `CallTrace.unverified` so policies and spans can see what was assumed.
    """

    method: Literal["POST", "GET"] = "POST"
    path: str | None = None
    body: dict[str, Any] | None = None
    content: bytes | None = Field(default=None, repr=False)
    headers: dict[str, str] = Field(default_factory=dict)
    stream: bool = False
    unverified: frozenset[str] = frozenset()


class Adapter[Req, Res](ABC):
    """Translates one capability's IR into one wire dialect, and the reply back.

    An adapter is the only place in franca that knows what a provider's JSON looks
    like. It is stateless and cheap to construct: the registry keeps one instance per
    dialect and shares it across calls, so an implementation must not stash per-call
    state on `self`.

    The class variables are metadata the registry indexes on, not behaviour:
    `capability` and `dialect` say which slot this adapter fills, and `status`
    distinguishes an adapter that has been exercised against the live provider from
    one written from documentation alone. Selection rejects a `"stub"` unless the
    caller passed `Requirements.allow_stubs`, and the conformance report lists every
    check for a stub as `todo`.

    Type parameters `Req` and `Res` are the capability's request and response types --
    for chat, `PromptPackage` and `ModelResponse`. They are generic rather than fixed
    so a capability package can be added without touching core.
    """

    capability: ClassVar[Capability]
    dialect: ClassVar[Dialect]
    status: ClassVar[Literal["stable", "stub"]] = "stable"

    @abstractmethod
    def to_request(
        self,
        req: Req,
        model: str,
        profile: BaseProfile,
        *,
        stream: bool = False,
    ) -> WireRequest:
        """Render a capability request as one HTTP request in this dialect.

        The adapter reads `profile` instead of branching on `model`; when it acts on a
        flag whose value is `None` it must name that flag in `WireRequest.unverified`,
        so the assumption reaches the trace.

        Args:
            req: The capability's request object.
            model: The model identifier to put on the wire, as the caller spelled it.
            profile: What franca believes about that model.
            stream: Whether to shape the request for server-sent events.

        Returns:
            The request for the `Connector` to send.
        """

    @abstractmethod
    def from_response(
        self,
        raw: Mapping[str, Any],
        req: Req,
        model: str,
        provider: Provider,
    ) -> Res:
        """Parse a successful, decoded provider reply into the capability's response.

        `req` is passed back in because several dialects answer with deltas against
        the request rather than a self-contained document. The adapter never sets
        `trace`: the leaf stamps that after this returns.

        Args:
            raw: The decoded JSON body of a 2xx response.
            req: The request that produced it.
            model: The model identifier that was sent.
            provider: The provider that answered.

        Returns:
            The capability's response object, with `trace` still `None`.
        """

    def error_from(
        self,
        status: int,
        raw: Mapping[str, Any] | None,
        provider: Provider,
    ) -> ModelError | None:
        """Refine the `Connector`'s classification of a failed response, if it can.

        The `Connector` already maps HTTP status to a `failure_class` and a
        `retryable` flag. A dialect that puts a more specific verdict in its error
        envelope -- an overloaded marker inside a 400, say -- overrides this to say
        so. Returning `None`, the default, keeps the `Connector`'s own classification;
        an override is expected to narrow it, never to widen it.

        Args:
            status: The HTTP status the provider answered with.
            raw: The decoded error body, when there was a decodable one.
            provider: The provider that answered.

        Returns:
            A more precise error, or `None` to accept the `Connector`'s.
        """
        return None


class StreamingAdapter[Req, Res, D](Adapter[Req, Res], ABC):
    """An `Adapter` that can also turn a framed SSE stream into incremental deltas.

    The extra type parameter `D` is the capability's delta type -- what one token or
    one tool-argument fragment looks like once the dialect's event JSON is decoded.

    Streaming is a separate base class because not every dialect streams, and a
    capability leaf that never streams should not be able to call `from_stream`.
    """

    @abstractmethod
    def from_stream(self, events: AsyncIterator[SseEvent], req: Req) -> AsyncIterator[D]:
        """Turn framed server-sent events into this capability's deltas.

        Declared as an ordinary method returning an async iterator, so an
        implementation can simply be an `async def` generator. Framing is already
        done: `events` yields parsed `SseEvent`s, and everything left is the JSON
        inside each `data:` payload.

        Args:
            events: The framed events of one streamed response, in order.
            req: The request that produced the stream.

        Returns:
            An async iterator of deltas.
        """
