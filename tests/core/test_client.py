"""Executable specification for `franca.core.client`.

Two things are worth pinning down here and nothing else is. The first is composition
order: `wrap` takes layers outermost-first but has to build them innermost-first, and
getting that backwards would silently invert every pipeline in the library -- so the
layers below append their name to a shared list and the tests assert the exact sequence
rather than a property of it. The second is that `DelegatingClient` forwards *both*
calls, because the whole point of the no-`next` design is that a middleware overriding
one method inherits a correct implementation of the other.

Conformance to `Client` is structural, so a handful of annotated assignments are
themselves the test: they only type-check if the shapes match, and mypy is part of the
gate.
"""

from collections.abc import AsyncIterator

from franca.core.client import Client, DelegatingClient, Layer, identity, wrap


class RecordingLeaf:
    """A concrete `Client[str, str, str]` that records every call it is handed.

    It names no base class on purpose: conformance to the `Client` protocol is
    structural, and the annotated assignments in this module are what prove it.
    """

    def __init__(self, calls: list[str], *, name: str = "leaf") -> None:
        """Record each call under `name` into the shared `calls` list.

        Args:
            calls: Shared log, appended to on every `complete` and `stream`.
            name: The name this leaf logs under.
        """
        self.name = name
        self.calls = calls
        self.completed: list[str] = []
        self.streamed: list[str] = []

    async def complete(self, req: str, /) -> str:
        """Record the call and answer with a tagged echo of `req`."""
        self.calls.append(self.name)
        self.completed.append(req)
        return f"{self.name}:{req}"

    def stream(self, req: str, /) -> AsyncIterator[str]:
        """Record the call and hand back one delta per character of `req`."""
        self.calls.append(self.name)
        self.streamed.append(req)
        return self._deltas(req)

    async def _deltas(self, req: str) -> AsyncIterator[str]:
        """Yield `req` one character at a time."""
        for char in req:
            yield char


class NamedLayer(DelegatingClient[str, str, str]):
    """Middleware that appends its own name to a shared log, then delegates.

    It overrides both methods, so the log records the order a call travels down the
    chain for `complete` and for `stream` alike.
    """

    def __init__(self, inner: Client[str, str, str], name: str, calls: list[str]) -> None:
        """Wrap `inner`, recording every call under `name` in `calls`.

        Args:
            inner: The next client in the chain.
            name: The name this layer logs under.
            calls: Shared log, appended to on every `complete` and `stream`.
        """
        super().__init__(inner)
        self.name = name
        self.calls = calls

    async def complete(self, req: str, /) -> str:
        """Record the call, then delegate to `inner`."""
        self.calls.append(self.name)
        return await super().complete(req)

    def stream(self, req: str, /) -> AsyncIterator[str]:
        """Record the call, then delegate to `inner`."""
        self.calls.append(self.name)
        return super().stream(req)


class UppercasingClient(DelegatingClient[str, str, str]):
    """Middleware that overrides only `complete`, to prove `stream` still forwards."""

    async def complete(self, req: str, /) -> str:
        """Uppercase whatever `inner` answered."""
        return (await super().complete(req)).upper()


def named(name: str, calls: list[str], built: list[str]) -> Layer[str, str, str]:
    """Build a layer that logs construction into `built` and calls into `calls`."""

    def layer(client: Client[str, str, str]) -> Client[str, str, str]:
        built.append(name)
        return NamedLayer(client, name, calls)

    return layer


async def collect(deltas: AsyncIterator[str]) -> list[str]:
    """Drain an async iterator into a list."""
    return [delta async for delta in deltas]


def test_wrap_with_no_layers_returns_the_leaf_object() -> None:
    leaf = RecordingLeaf([])

    # Identity of object, not merely equality: an empty pipeline allocates nothing.
    assert wrap(leaf) is leaf


def test_wrap_with_one_layer_returns_that_layers_product() -> None:
    calls: list[str] = []
    built: list[str] = []
    leaf = RecordingLeaf(calls)

    client = wrap(leaf, named("retry", calls, built))

    assert isinstance(client, NamedLayer)
    assert client.name == "retry"
    assert client.inner is leaf
    assert built == ["retry"]


def test_wrap_nests_outermost_first() -> None:
    calls: list[str] = []
    built: list[str] = []
    leaf = RecordingLeaf(calls)

    # wrap(m, trace, retry) == trace(retry(m)).
    outer = wrap(leaf, named("trace", calls, built), named("retry", calls, built))

    assert isinstance(outer, NamedLayer)
    assert outer.name == "trace"
    middle = outer.inner
    assert isinstance(middle, NamedLayer)
    assert middle.name == "retry"
    assert middle.inner is leaf


def test_wrap_applies_layers_in_reverse_so_the_innermost_is_built_first() -> None:
    calls: list[str] = []
    built: list[str] = []
    leaf = RecordingLeaf(calls)

    wrap(
        leaf,
        named("trace", calls, built),
        named("tools", calls, built),
        named("retry", calls, built),
    )

    assert built == ["retry", "tools", "trace"]


async def test_complete_runs_three_layers_outermost_first() -> None:
    calls: list[str] = []
    built: list[str] = []
    leaf = RecordingLeaf(calls)
    client = wrap(
        leaf,
        named("trace", calls, built),
        named("tools", calls, built),
        named("retry", calls, built),
    )

    assert await client.complete("hi") == "leaf:hi"
    assert calls == ["trace", "tools", "retry", "leaf"]


async def test_stream_runs_three_layers_outermost_first() -> None:
    calls: list[str] = []
    built: list[str] = []
    leaf = RecordingLeaf(calls)
    client = wrap(
        leaf,
        named("trace", calls, built),
        named("tools", calls, built),
        named("retry", calls, built),
    )

    assert await collect(client.stream("hi")) == ["h", "i"]
    assert calls == ["trace", "tools", "retry", "leaf"]


async def test_wrap_result_matches_manual_nesting() -> None:
    wrapped_calls: list[str] = []
    manual_calls: list[str] = []
    built: list[str] = []
    wrapped = wrap(
        RecordingLeaf(wrapped_calls),
        named("trace", wrapped_calls, built),
        named("retry", wrapped_calls, built),
    )
    manual = NamedLayer(
        NamedLayer(RecordingLeaf(manual_calls), "retry", manual_calls), "trace", manual_calls
    )

    assert await wrapped.complete("hi") == await manual.complete("hi")
    assert wrapped_calls == manual_calls


async def test_delegating_client_forwards_complete() -> None:
    calls: list[str] = []
    leaf = RecordingLeaf(calls)
    client = DelegatingClient(leaf)

    assert await client.complete("hi") == "leaf:hi"
    assert leaf.completed == ["hi"]
    assert calls == ["leaf"]


async def test_delegating_client_forwards_stream() -> None:
    calls: list[str] = []
    leaf = RecordingLeaf(calls)
    client = DelegatingClient(leaf)

    assert await collect(client.stream("hi")) == ["h", "i"]
    assert leaf.streamed == ["hi"]
    assert calls == ["leaf"]


def test_delegating_client_exposes_inner() -> None:
    leaf = RecordingLeaf([])

    assert DelegatingClient(leaf).inner is leaf


async def test_subclass_overriding_only_complete_still_forwards_stream() -> None:
    calls: list[str] = []
    leaf = RecordingLeaf(calls)
    client = UppercasingClient(leaf)

    assert await client.complete("hi") == "LEAF:HI"
    # stream was never overridden, yet it reaches the leaf untouched.
    assert await collect(client.stream("hi")) == ["h", "i"]
    assert leaf.streamed == ["hi"]


def test_identity_returns_the_same_client() -> None:
    leaf = RecordingLeaf([])

    assert identity(leaf) is leaf


def test_identity_is_a_valid_layer() -> None:
    # The annotation is the test: `identity` has to satisfy the `Layer` alias.
    layer: Layer[str, str, str] = identity
    leaf = RecordingLeaf([])

    assert layer(leaf) is leaf


def test_wrapping_with_identity_layers_leaves_the_leaf_reachable() -> None:
    leaf = RecordingLeaf([])

    # An unfilled slot binds `identity`, so a default route stays free of wrappers.
    assert wrap(leaf, identity, identity) is leaf


def test_a_lambda_is_a_valid_layer() -> None:
    calls: list[str] = []
    leaf = RecordingLeaf(calls)

    client = wrap(leaf, lambda inner: NamedLayer(inner, "lint", calls))

    assert isinstance(client, NamedLayer)
    assert client.name == "lint"


def test_concrete_leaf_satisfies_the_client_protocol() -> None:
    # Structural conformance, checked by mypy: no base class, no registration.
    client: Client[str, str, str] = RecordingLeaf([])

    assert isinstance(client, RecordingLeaf)


def test_delegating_client_satisfies_the_client_protocol() -> None:
    client: Client[str, str, str] = DelegatingClient(RecordingLeaf([]))

    assert isinstance(client, DelegatingClient)


def test_wrap_returns_something_that_satisfies_the_client_protocol() -> None:
    calls: list[str] = []
    built: list[str] = []
    client: Client[str, str, str] = wrap(RecordingLeaf(calls), named("trace", calls, built))

    assert isinstance(client, NamedLayer)
