"""Executable specification for `franca.core.model.Model`, the pipeline leaf.

Four things are worth pinning here and everything else follows from them.

The `prepare` hook must actually run: it is the only place a capability subclass gets
to touch a request, and a leaf that called the adapter with the caller's object would
silently disable `ChatModel.prepare` and every check that lives in it.

A request-shaping `ValidationError` must arrive as `ModelError(failure_class="request")`
whose message names the *path* and not the value (FR-2). That is a security property,
not a formatting preference -- request bodies hold key material -- so the rejecting
adapter below is fed a secret-looking string and the test asserts both that the path
survives and that the secret does not, including that `str()` on the original
`ValidationError` would have printed it.

The trace must be stamped, and stamped from the `Selection`. So the explicit-selection
test deliberately points at a *different* endpoint and dialect than the connector's:
if `_trace` read the connector instead, the trace would still be non-None and only that
asymmetry catches it.

And latency must come from the injected clock, not the wall: the fake below walks a
fixed script so `latency_ms` is an exact number rather than a bound.

Every double lives in this file. `franca.testing` and `ScriptedTransport` are other
people's milestones, and the leaf is worth being able to test without either; the
`Connector` underneath is real, so the seam the leaf actually calls is exercised.
"""

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel, SecretStr, ValidationError

from franca.core.adapter import Adapter, WireRequest
from franca.core.connector import Connector
from franca.core.endpoint import Endpoint, Selection
from franca.core.enums import AuthScheme
from franca.core.errors import ModelError
from franca.core.ids import (
    ANTHROPIC,
    ANTHROPIC_MESSAGES,
    CHAT,
    OPENAI,
    OPENAI_RESPONSES,
    Provider,
)
from franca.core.keys import StaticKeyProvider
from franca.core.model import Model
from franca.core.profile import BaseProfile
from franca.core.transport import RawResponse, RawStream
from franca.core.types import Traced

if TYPE_CHECKING:
    from franca.core.client import Client

SECRET = "sk-ant-api03-do-not-print-me"  # noqa: S105  -- a decoy, not a credential
MODEL = "claude-opus-4-5"

ENDPOINT = Endpoint(
    id="anthropic/chat/messages",
    provider=ANTHROPIC,
    capability=CHAT,
    dialect=ANTHROPIC_MESSAGES,
    base_url="https://api.anthropic.com",
    path="/v1/messages",
    auth=AuthScheme.x_api_key,
)
OTHER_ENDPOINT = Endpoint(
    id="openai/chat/responses",
    provider=OPENAI,
    capability=CHAT,
    dialect=OPENAI_RESPONSES,
    base_url="https://api.openai.com",
    path="/v1/responses",
    auth=AuthScheme.bearer,
)
PROFILE = BaseProfile(provider=ANTHROPIC, model_prefix="claude-opus")
KEYS = StaticKeyProvider({ANTHROPIC: SecretStr("not-a-real-key")})


class Reply(Traced, frozen=True):
    """The response type of the toy capability these tests exercise."""

    text: str


class Sampling(BaseModel, frozen=True):
    """A nested body fragment, so a rejected field has a real dotted path."""

    temperature: float


class ToyBody(BaseModel, frozen=True):
    """The outbound body the rejecting adapter validates before sending it."""

    sampling: Sampling


class ToyAdapter(Adapter[str, Reply]):
    """Renders a string request as a one-field body and reads the reply back.

    It records every `to_request` call, which is how the tests see what `prepare`
    handed the adapter rather than what the caller passed the leaf.
    """

    capability = CHAT
    dialect = ANTHROPIC_MESSAGES

    def __init__(self, *, unverified: frozenset[str] = frozenset()) -> None:
        """Record calls; `unverified` is stamped on every `WireRequest` produced.

        Args:
            unverified: Profile flags this adapter claims to have assumed, as a real
                adapter would name a flag it acted on while its value was `None`.
        """
        self.seen: list[tuple[str, str, bool]] = []
        self._unverified = unverified

    def to_request(
        self,
        req: str,
        model: str,
        profile: BaseProfile,
        *,
        stream: bool = False,
    ) -> WireRequest:
        """Record what the leaf passed, then render it as a one-field JSON body."""
        self.seen.append((req, model, stream))
        return WireRequest(
            body={"model": model, "text": req},
            stream=stream,
            unverified=self._unverified,
        )

    def from_response(
        self,
        raw: Mapping[str, Any],
        req: str,
        model: str,
        provider: Provider,
    ) -> Reply:
        """Read the echoed text back out, tagged with the provider that answered."""
        return Reply(text=f"{provider}:{raw['text']}")


class RejectingAdapter(ToyAdapter):
    """An adapter that validates its outbound body and is handed a bad value.

    This is the realistic shape of the FR-2 failure: the adapter builds a typed request
    model, pydantic rejects a field, and the leaf has to render that without echoing
    what was rejected.
    """

    def to_request(
        self,
        req: str,
        model: str,
        profile: BaseProfile,
        *,
        stream: bool = False,
    ) -> WireRequest:
        """Validate a body whose `sampling.temperature` is `req`, and fail on a string."""
        body = ToyBody.model_validate({"sampling": {"temperature": req}})
        return WireRequest(body=body.model_dump())


class StubTransport:
    """A `Transport` that answers every POST with one canned JSON body."""

    def __init__(self, body: bytes = b'{"text": "hi"}') -> None:
        """Answer every POST with `body`, recording the URL and content it was given.

        Args:
            body: The response bytes to hand back, always with status 200.
        """
        self.posted: list[tuple[str, bytes]] = []
        self._body = body

    async def post(
        self, url: str, content: bytes, headers: Mapping[str, str], *, timeout_s: float
    ) -> RawResponse:
        """Record the POST and hand back the canned 200."""
        self.posted.append((url, content))
        return RawResponse(status=200, headers={}, body=self._body)

    async def get(self, url: str, headers: Mapping[str, str], *, timeout_s: float) -> RawResponse:
        """Unreachable: the leaf's non-streaming path never issues a GET."""
        raise NotImplementedError

    def post_stream(
        self, url: str, content: bytes, headers: Mapping[str, str], *, timeout_s: float
    ) -> AbstractAsyncContextManager[RawStream]:
        """Unreachable: `Model.stream` refuses before any transport is reached."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Hold no pooled resources, so releasing them is a no-op."""


class StepClock:
    """A `Clock` whose monotonic reading walks a fixed script, one step per read.

    Once the script runs out the last reading repeats, so a test that only cares about
    latency scripts two values and nothing downstream has to count reads.
    """

    def __init__(self, *readings: float) -> None:
        """Return `readings` in order, then hold at the last one.

        Args:
            readings: The monotonic values to hand out, in order. At least one.
        """
        self._readings = list(readings)
        self._index = 0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        """Return the next scripted reading, holding at the last."""
        reading = self._readings[min(self._index, len(self._readings) - 1)]
        self._index += 1
        return reading

    def wall(self) -> float:
        """Return a fixed wall reading; nothing under test converts an HTTP-date."""
        return 1_000_000.0

    async def sleep(self, seconds: float) -> None:
        """Record the request instead of waiting."""
        self.slept.append(seconds)


class ShoutingModel(Model[str, Reply, str]):
    """A leaf whose `prepare` hook uppercases the request, to prove the hook runs."""

    capability = CHAT

    def prepare(self, req: str) -> str:
        """Uppercase the request before the adapter sees it."""
        return req.upper()


def build_leaf(
    *,
    adapter: Adapter[str, Reply] | None = None,
    transport: StubTransport | None = None,
    clock: StepClock | None = None,
    selection: Selection | None = None,
    endpoint: Endpoint = ENDPOINT,
) -> Model[str, Reply, str]:
    """Assemble a leaf over a real `Connector` and a stub transport.

    Anything a test does not care about gets a default: an echoing adapter, a transport
    that answers `{"text": "hi"}`, and a clock pinned at zero.
    """
    the_clock = clock if clock is not None else StepClock(0.0)
    the_transport = transport if transport is not None else StubTransport()
    return Model(
        model=MODEL,
        connector=Connector(endpoint, keys=KEYS, transport=the_transport, clock=the_clock),
        adapter=adapter if adapter is not None else ToyAdapter(),
        profile=PROFILE,
        clock=the_clock,
        selection=selection,
    )


def pinned(endpoint: Endpoint, *, unverified: frozenset[str] = frozenset()) -> Selection:
    """Build a `Selection` that says a caller pinned `endpoint`'s dialect."""
    return Selection(
        endpoint=endpoint,
        dialect=endpoint.dialect,
        adapter="PinnedAdapter",
        via="pin",
        unverified=unverified,
        rejected=("anthropic/chat/messages: dialect not pinned",),
    )


def test_prepare_output_is_what_the_adapter_receives() -> None:
    adapter = ToyAdapter()
    leaf = ShoutingModel(
        model=MODEL,
        connector=Connector(ENDPOINT, keys=KEYS, transport=StubTransport(), clock=StepClock(0.0)),
        adapter=adapter,
        profile=PROFILE,
        clock=StepClock(0.0),
    )

    wire = leaf.build("hello")

    assert adapter.seen == [("HELLO", MODEL, False)]
    assert wire.body == {"model": MODEL, "text": "HELLO"}


def test_the_base_prepare_hook_changes_nothing() -> None:
    leaf = build_leaf()

    assert leaf.prepare("hello") == "hello"


def test_build_passes_the_stream_flag_to_the_adapter() -> None:
    adapter = ToyAdapter()

    wire = build_leaf(adapter=adapter).build("hello", stream=True)

    assert wire.stream is True
    assert adapter.seen == [("hello", MODEL, True)]


def test_a_validation_error_in_to_request_becomes_a_request_model_error() -> None:
    leaf = build_leaf(adapter=RejectingAdapter())

    with pytest.raises(ModelError) as caught:
        leaf.build(SECRET)

    error = caught.value
    assert error.failure_class == "request"
    assert error.retryable is False
    assert error.status is None
    assert error.provider == ANTHROPIC


def test_the_request_error_names_the_field_path_and_never_the_value() -> None:
    leaf = build_leaf(adapter=RejectingAdapter())

    with pytest.raises(ModelError) as caught:
        leaf.build(SECRET)

    error = caught.value
    assert "sampling.temperature" in error.message
    assert SECRET not in error.message
    assert SECRET not in str(error)
    assert SECRET not in repr(error)


def test_the_unwrapped_validation_error_would_have_printed_the_value() -> None:
    """Prove the guard is load-bearing: pydantic's own rendering leaks the input."""
    leaf = build_leaf(adapter=RejectingAdapter())

    with pytest.raises(ModelError) as caught:
        leaf.build(SECRET)

    cause = caught.value.__cause__
    assert isinstance(cause, ValidationError)
    assert SECRET in str(cause)


async def test_complete_returns_the_adapters_response_stamped_with_a_trace() -> None:
    transport = StubTransport(b'{"text": "pong"}')

    reply = await build_leaf(transport=transport).complete("ping")

    assert reply.text == "anthropic:pong"
    assert reply.trace is not None
    assert reply.trace.endpoint_id == "anthropic/chat/messages"
    assert reply.trace.dialect == ANTHROPIC_MESSAGES
    assert reply.trace.selection_via == "native"
    assert transport.posted == [
        ("https://api.anthropic.com/v1/messages", b'{"model":"claude-opus-4-5","text":"ping"}')
    ]


async def test_latency_ms_comes_from_the_injected_clock() -> None:
    reply = await build_leaf(clock=StepClock(0.0, 0.25)).complete("ping")

    assert reply.trace is not None
    assert reply.trace.latency_ms == 250.0
    assert reply.trace.ttft_ms is None


async def test_the_default_selection_is_native_over_the_connectors_endpoint() -> None:
    leaf = build_leaf()

    assert leaf.selection.endpoint == ENDPOINT
    assert leaf.selection.dialect == ANTHROPIC_MESSAGES
    assert leaf.selection.adapter == "ToyAdapter"
    assert leaf.selection.via == "native"
    assert leaf.selection.unverified == frozenset()
    assert leaf.selection.rejected == ()
    assert (await leaf.complete("ping")).trace is not None


async def test_an_explicit_selection_is_used_unchanged_and_wins_over_the_endpoint() -> None:
    selection = pinned(OTHER_ENDPOINT)

    leaf = build_leaf(selection=selection)
    reply = await leaf.complete("ping")

    assert leaf.selection is selection
    assert reply.trace is not None
    # The connector still points at Anthropic; the trace reports what was selected.
    assert leaf.connector.endpoint.id == "anthropic/chat/messages"
    assert reply.trace.endpoint_id == "openai/chat/responses"
    assert reply.trace.dialect == OPENAI_RESPONSES
    assert reply.trace.selection_via == "pin"


async def test_wire_unverified_is_unioned_into_the_traces_unverified() -> None:
    leaf = build_leaf(
        adapter=ToyAdapter(unverified=frozenset({"caching"})),
        selection=pinned(OTHER_ENDPOINT, unverified=frozenset({"strict"})),
    )

    reply = await leaf.complete("ping")

    assert reply.trace is not None
    assert reply.trace.unverified == frozenset({"caching", "strict"})


def test_stream_raises_an_unsupported_model_error() -> None:
    leaf = build_leaf()

    with pytest.raises(ModelError) as caught:
        leaf.stream("ping")

    error = caught.value
    assert error.failure_class == "unsupported"
    assert error.retryable is False
    assert error.status is None
    assert error.provider == ANTHROPIC
    assert "M1" in error.message


def test_provider_comes_from_the_connectors_endpoint() -> None:
    assert build_leaf().provider == ANTHROPIC
    assert build_leaf(endpoint=OTHER_ENDPOINT).provider == OPENAI


async def test_the_leaf_satisfies_the_client_protocol_structurally() -> None:
    # mypy --strict checks this assignment; nothing here names a base class.
    client: Client[str, Reply, str] = build_leaf()

    assert (await client.complete("ping")).text == "anthropic:hi"


def test_capability_is_readable_off_the_subclass() -> None:
    assert ShoutingModel.capability == CHAT
