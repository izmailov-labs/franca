"""The middleware seam: one call shape, one delegating base, one way to compose them.

franca borrows the Microsoft.Extensions.AI pipeline model. A `Client` is anything that
can answer a request in one shot *and* as a stream; middleware is a `DelegatingClient`
subclass that overrides one or both; a layer is a plain callable that wraps a client in
another client. There is deliberately no `next` parameter anywhere -- the inner client is
constructor state, not an argument -- which keeps the two call paths symmetric: a layer
that only cares about `complete` inherits a correct `stream`, and a layer that only cares
about `stream` inherits a correct `complete`. Nothing has to be registered twice, and a
one-off layer is a lambda.

Three type parameters, because the pipeline is capability-agnostic: `Req` and `Res` are
the capability's request and response models, and `D` is what a stream yields -- a chat
delta, a `Job` for the polling capabilities, `Never` where streaming is not offered at
all. Every layer in one chain shares all three, so `wrap` cannot accidentally splice a
chat layer into a video route.

`Client` is a structural protocol: the leaf that actually talks to a provider satisfies
it by shape, and so does every middleware, so `wrap` composes them without a common base
class. `DelegatingClient` is the base only because forwarding is tedious, never because
the protocol demands it.
"""

from collections.abc import AsyncIterator, Callable
from typing import Protocol


class Client[Req, Res, D](Protocol):
    """One capability call, in both shapes: awaited whole, or consumed as deltas.

    The leaf implementation is the model bound to a connector; every middleware is a
    `Client` wrapping another `Client`; the registry hands one back. Because this is a
    protocol, none of them share a base class -- conformance is by shape.
    """

    async def complete(self, req: Req, /) -> Res:
        """Answer `req` in one shot.

        The request parameter is positional-only so a middleware can name it whatever
        reads best without breaking a caller.

        Args:
            req: The capability request, already validated by its own model.

        Returns:
            The capability response.

        Raises:
            ModelError: When the call fails. Callers above this seam reason about
                `failure_class` and `retryable`, never about HTTP.
        """

    def stream(self, req: Req, /) -> AsyncIterator[D]:
        """Answer `req` as a stream of deltas.

        A plain method, not a coroutine: nothing goes on the wire until the returned
        iterator is first advanced, which is what lets a middleware decide to open a
        different inner stream instead. Consumers that stop early must close the
        iterator (`contextlib.aclosing` or `ChatStream`); a bare `break` leaves the
        underlying connection to the event loop's async-generator finalizer.

        Args:
            req: The capability request, already validated by its own model.

        Returns:
            An async iterator over the capability's stream item type.
        """


class DelegatingClient[Req, Res, D]:
    """The base every middleware extends: forwards both calls to `inner` unchanged.

    Subclass it and override only what you need. `RetryClient` overrides both;
    a metrics layer might override only `complete`, and its `stream` still reaches
    `inner` correctly. `inner` is a plain attribute rather than a private slot because
    tests and diagnostics walk the chain.
    """

    def __init__(self, inner: Client[Req, Res, D]) -> None:
        """Wrap `inner`.

        Args:
            inner: The next client in the chain -- another middleware, or the leaf.
        """
        self.inner = inner

    async def complete(self, req: Req, /) -> Res:
        """Forward `req` to `inner.complete`.

        Args:
            req: The capability request.

        Returns:
            Whatever `inner` answered.
        """
        return await self.inner.complete(req)

    def stream(self, req: Req, /) -> AsyncIterator[D]:
        """Forward `req` to `inner.stream`.

        The inner iterator is returned as-is rather than re-yielded, so a subclass that
        does not care about streaming adds no frame and no extra finalizer.

        Args:
            req: The capability request.

        Returns:
            The iterator `inner` produced.
        """
        return self.inner.stream(req)


type Layer[Req, Res, D] = Callable[[Client[Req, Res, D]], Client[Req, Res, D]]
"""A pipeline step: takes the client it wraps and returns the client that replaces it."""


def wrap[Req, Res, D](
    leaf: Client[Req, Res, D], *layers: Layer[Req, Res, D]
) -> Client[Req, Res, D]:
    """Compose `layers` around `leaf`, outermost first.

    Outermost first means `wrap(m, trace, retry) == trace(retry(m))`; calls run trace then
    retry then m.

    Layers are written in the order a call travels through them, which is the order they
    are read in configuration and in a trace. Construction therefore runs in reverse: the
    innermost layer is built first, around the leaf. With no layers the leaf is returned
    unchanged -- the same object, not a wrapper -- so an empty pipeline costs nothing.

    Args:
        leaf: The client at the bottom of the chain, usually the model bound to a
            connector.
        layers: Layers to apply, outermost first.

    Returns:
        The outermost client, or `leaf` itself when no layers were given.
    """
    client = leaf
    for layer in reversed(layers):
        client = layer(client)
    return client


def identity[Req, Res, D](client: Client[Req, Res, D]) -> Client[Req, Res, D]:
    """Return `client` unchanged: the layer that adds nothing.

    The registry binds this into a slot that nothing has filled -- the `"tools"` slot on a
    capability with no tool support, for instance -- so a default route stays valid
    everywhere instead of needing a per-capability layer list.

    Args:
        client: The client to leave alone.

    Returns:
        `client`, the same object.
    """
    return client
