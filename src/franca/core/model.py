"""The pipeline leaf: the one object that turns a capability request into a provider call.

`Model` sits at the bottom of every chain `franca.core.client.wrap` builds. Above it
are layers that reason about failure classes and traces; below it are an `Adapter`
(what the wire looks like) and a `Connector` (where the wire goes and what a status
code means). The leaf owns only the four steps no layer above it should have to
repeat: run the capability's `prepare` hook, ask the adapter for a `WireRequest`,
send it through the connector, and stamp a `CallTrace` on whatever comes back.

It satisfies `Client[Req, Res, D]` structurally rather than by inheritance --
`complete` and `stream` are the whole protocol -- so any layer sharing its three type
parameters can wrap it.

Two decisions are worth stating out loud. Selection is never redone here: the
`Selection` is either handed in by whoever chose the endpoint, or derived from the
connector's own endpoint row as `via="native"`, so a leaf cannot silently re-open a
decision that was already made. And a `ValidationError` raised while building a
request is translated in exactly one place -- `build` -- into
`ModelError(failure_class="request")` via `format_validation`, which renders the
field path and never the rejected value (FR-2); inputs routinely hold key material.

`stream` is declared but deliberately unimplemented. Streaming needs
`Connector.stream`, a `StreamingAdapter` and the last-item trace stamp with
`ttft_ms`, which land together in M1; until then it raises an `unsupported`
`ModelError` rather than half-working.
"""

from collections.abc import AsyncIterator
from typing import Any, ClassVar

from pydantic import ValidationError

from franca.core.adapter import Adapter, WireRequest
from franca.core.clock import Clock
from franca.core.connector import Connector
from franca.core.endpoint import Endpoint, Selection
from franca.core.errors import ModelError, format_validation
from franca.core.ids import Capability, Provider
from franca.core.profile import BaseProfile
from franca.core.types import CallTrace, HasTrace


class Model[Req, Res: HasTrace, D]:
    """One model, bound to one endpoint, one adapter and one profile.

    Adapters never construct a `CallTrace`: `complete()` always returns
    `res.with_trace(...)`, so `trace` is non-None on anything leaving a leaf.

    The three type parameters are the capability's request, its response and what its
    stream yields; a capability that does not stream instantiates `D` as `Never`.
    `Res` is bound to `HasTrace` because stamping the trace is the leaf's whole
    remaining job once the adapter has parsed the reply.

    Subclasses specialise the leaf by overriding `prepare` (and, for a capability with
    endpoint requirements to re-check, `build`); `ChatModel` is the first of them.
    Everything else -- the trace, the `ValidationError` translation, the latency
    measurement -- is the same for every capability and lives here.

    Attributes:
        model: The model identifier to put on the wire, as the caller spelled it.
        connector: The endpoint's HTTP face; the leaf never touches a transport.
        adapter: The dialect translation, in both directions.
        profile: What franca believes about `model`, for the adapter to read.
        clock: Where `latency_ms` comes from; injected so tests are exact.
        selection: How this endpoint and dialect were arrived at, for the trace.
    """

    capability: ClassVar[Capability]

    def __init__(
        self,
        *,
        model: str,
        connector: Connector,
        adapter: Adapter[Req, Res],
        profile: BaseProfile,
        clock: Clock,
        selection: Selection | None = None,
    ) -> None:
        """Bind a model identifier to the seams one call needs.

        Args:
            model: The model identifier to send, unnormalised.
            connector: The connector for the endpoint this leaf calls.
            adapter: The adapter for the dialect that endpoint speaks.
            profile: The resolved profile row for `model`.
            clock: Source of monotonic readings for the latency measurement.
            selection: The outcome of dialect selection. `None` means nobody selected
                anything -- the connector's endpoint was given directly -- and a
                `via="native"` selection is derived from that endpoint row, so the
                trace is populated either way.
        """
        self.model = model
        self.connector = connector
        self.adapter = adapter
        self.profile = profile
        self.clock = clock
        self.selection = (
            selection if selection is not None else _native(connector.endpoint, adapter)
        )

    @property
    def provider(self) -> Provider:
        """The provider whose key signs this leaf's requests."""
        return self.connector.endpoint.provider

    def prepare(self, req: Req) -> Req:
        """Return the request the adapter should actually see.

        The base leaf changes nothing. Subclasses use this hook for the work that is
        capability-specific but dialect-independent: filling in ids, refusing a retired
        model, or clamping a limit the profile declares.

        Args:
            req: The request as the caller passed it.

        Returns:
            The request to translate; `req` itself, unchanged, here.
        """
        return req

    def build(self, req: Req, *, stream: bool = False) -> WireRequest:
        """Translate `req` into one HTTP request, via `prepare` and the adapter.

        This is the only place a request-shaping `ValidationError` is turned into a
        `ModelError`, and it goes through `format_validation`, so the message carries
        the field path and never the rejected input.

        Args:
            req: The request as the caller passed it.
            stream: Whether to shape the request for server-sent events.

        Returns:
            The `WireRequest` for the connector to send.

        Raises:
            ModelError: With `failure_class="request"` and `retryable=False` when the
                adapter rejected the request. Repeating an invalid request cannot help.
        """
        try:
            return self.adapter.to_request(
                self.prepare(req), self.model, self.profile, stream=stream
            )
        except ValidationError as exc:
            raise ModelError(
                format_validation(exc),
                status=None,
                provider=self.provider,
                retryable=False,
                failure_class="request",
            ) from exc

    async def complete(self, req: Req, /) -> Res:
        """Build, send, parse and stamp: one call, one response, one trace.

        The clock is read either side of the connector call alone, so `latency_ms`
        measures the request and not the translation on either end of it.

        Args:
            req: The capability request.

        Returns:
            The capability response, carrying a non-None `trace`.

        Raises:
            ModelError: From `build` when the request is invalid, or from the
                connector for anything that went wrong on or before the wire --
                after `adapter.error_from` has had the chance to reclassify it.
        """
        wire = self.build(req)
        t0 = self.clock.monotonic()
        try:
            raw = await self.connector.send(wire)
        except ModelError as exc:
            raise self._refine(exc) from exc
        res = self.adapter.from_response(raw, req, self.model, self.provider)
        return res.with_trace(self._trace(wire, latency_ms=(self.clock.monotonic() - t0) * 1000))

    def _refine(self, exc: ModelError) -> ModelError:
        """Give the adapter a chance to reclassify a connector error, then raise it.

        The `Connector` classifies by HTTP status because that is all it can see; it is
        deliberately adapter-agnostic, so it cannot read a dialect's error envelope.
        Some statuses are genuinely ambiguous at that level -- OpenAI answers both "you
        are going too fast" and "your balance is empty" with a 429, and only the body
        separates a retryable failure from a permanent one. `error_from` is the seam
        where the dialect that understands the envelope gets to correct the guess.

        This is where that seam is actually exercised: the leaf is the first place that
        holds both the adapter and the connector.

        Args:
            exc: The error the connector raised.

        Returns:
            The adapter's replacement when it offers one, otherwise `exc` unchanged.
        """
        if exc.status is None:
            return exc
        refined = self.adapter.error_from(exc.status, exc.raw, self.provider)
        return refined if refined is not None else exc

    def stream(self, req: Req, /) -> AsyncIterator[D]:
        """Answer `req` as a stream of deltas -- not yet implemented.

        Streaming needs `Connector.stream`, a `StreamingAdapter` and a trace stamped on
        the last item with `ttft_ms` from the first, which land together in M1. Until
        then this raises for every adapter, streaming-capable or not: a leaf that
        streamed without a trace would be worse than one that refuses.

        Args:
            req: The capability request.

        Returns:
            An async iterator over the capability's deltas, once M1 lands.

        Raises:
            ModelError: Always, with `failure_class="unsupported"` and
                `retryable=False`.
        """
        raise ModelError(
            "streaming lands in M1; use complete() for now",
            status=None,
            provider=self.provider,
            retryable=False,
            failure_class="unsupported",
        )

    def _trace(self, wire: WireRequest, **timing: float | None) -> CallTrace:
        """Record what the call did: where it went, how it was chosen, what was assumed.

        `unverified` is the union of two different assumptions: the features the
        selected endpoint leaves unverified, and the profile flags the adapter acted on
        while their value was `None`. Both are things nobody has confirmed, so the
        trace carries them as one set.

        Args:
            wire: The request that was sent, for the flags the adapter recorded on it.
            timing: Timing fields to set on the trace, e.g. `latency_ms`.

        Returns:
            The trace for `complete` to stamp on the response.
        """
        # Widened to `Any` deliberately: pydantic's `dataclass_transform` gives mypy a
        # synthesised `__init__` whose timing parameters are `float`, `float | None`,
        # `int` and `bool`, so a uniformly typed mapping cannot be splatted into it.
        # Every caller of `_trace` is inside this class, so the field names stay
        # checked by review rather than by the type checker.
        fields: dict[str, Any] = dict(timing)
        return CallTrace(
            endpoint_id=self.selection.endpoint.id,
            dialect=self.selection.dialect,
            selection_via=self.selection.via,
            unverified=self.selection.unverified | wire.unverified,
            **fields,
        )


def _native(endpoint: Endpoint, adapter: object) -> Selection:
    """Describe an endpoint nobody selected as the selection that would have picked it.

    Nothing was rejected and nothing was left unverified, because no candidate was ever
    considered: the caller named this endpoint and this adapter directly.

    Args:
        endpoint: The connector's endpoint row.
        adapter: The adapter bound to the leaf; only its class name is recorded.

    Returns:
        A `via="native"` selection over `endpoint`.
    """
    return Selection(
        endpoint=endpoint,
        dialect=endpoint.dialect,
        adapter=type(adapter).__name__,
        via="native",
        unverified=frozenset(),
        rejected=(),
    )
