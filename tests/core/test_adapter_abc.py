"""Executable specification for the `Adapter` / `StreamingAdapter` base classes.

The ABCs carry almost no behaviour, which is the point: what they enforce is that a
dialect implementation cannot exist without both halves of the translation, and that
the registry can read `capability`, `dialect` and `status` off the class before
anything is instantiated. `WireRequest` itself is specified in `test_adapter.py`.
"""

from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest

from franca.core.adapter import Adapter, StreamingAdapter, WireRequest
from franca.core.errors import ModelError
from franca.core.ids import ANTHROPIC, ANTHROPIC_MESSAGES, CHAT, Provider
from franca.core.profile import BaseProfile
from franca.core.sse import SseEvent

PROFILE = BaseProfile(provider=ANTHROPIC, model_prefix="claude-opus")


class EchoAdapter(Adapter[str, str]):
    """A minimal concrete adapter whose IR, in both directions, is a plain string."""

    capability = CHAT
    dialect = ANTHROPIC_MESSAGES

    def to_request(
        self,
        req: str,
        model: str,
        profile: BaseProfile,
        *,
        stream: bool = False,
    ) -> WireRequest:
        """Render the string as a one-field JSON body, naming the profile it used."""
        return WireRequest(
            body={"model": model, "text": req, "profile": profile.name}, stream=stream
        )

    def from_response(
        self,
        raw: Mapping[str, Any],
        req: str,
        model: str,
        provider: Provider,
    ) -> str:
        """Read the echoed text back out of the reply, tagged with the provider."""
        return f"{provider}:{raw['text']}"


class RefiningAdapter(EchoAdapter):
    """An adapter that overrides `error_from` to narrow the connector's verdict."""

    def error_from(
        self,
        status: int,
        raw: Mapping[str, Any] | None,
        provider: Provider,
    ) -> ModelError | None:
        """Treat an `overloaded` envelope inside a 400 as a retryable provider error."""
        if raw is not None and raw.get("type") == "overloaded":
            return ModelError(
                "overloaded",
                status=status,
                provider=provider,
                retryable=True,
                failure_class="provider",
            )
        return None


class EchoStreamingAdapter(StreamingAdapter[str, str, str]):
    """The same echo adapter, refined with a stream; deliberately marked a stub."""

    capability = CHAT
    dialect = ANTHROPIC_MESSAGES
    status = "stub"

    def to_request(
        self,
        req: str,
        model: str,
        profile: BaseProfile,
        *,
        stream: bool = False,
    ) -> WireRequest:
        """Render the string as a one-field JSON body."""
        return WireRequest(body={"text": req}, stream=stream)

    def from_response(
        self,
        raw: Mapping[str, Any],
        req: str,
        model: str,
        provider: Provider,
    ) -> str:
        """Read the echoed text back out of the reply."""
        return str(raw["text"])

    async def from_stream(self, events: AsyncIterator[SseEvent], req: str) -> AsyncIterator[str]:
        """Yield each event's payload verbatim; framing has already been done."""
        async for event in events:
            yield event.data


class HalfAdapter(Adapter[str, str]):
    """An adapter that never implemented `from_response`, so it must stay abstract."""

    capability = CHAT
    dialect = ANTHROPIC_MESSAGES

    def to_request(
        self,
        req: str,
        model: str,
        profile: BaseProfile,
        *,
        stream: bool = False,
    ) -> WireRequest:
        """Render the string as a one-field JSON body."""
        return WireRequest(body={"text": req})


async def events(*frames: SseEvent) -> AsyncIterator[SseEvent]:
    """Feed a fixed sequence of framed events to an adapter under test."""
    for frame in frames:
        yield frame


def test_a_concrete_subclass_instantiates_and_round_trips() -> None:
    adapter = EchoAdapter()

    wire = adapter.to_request("hello", "claude-opus-4-5", PROFILE)

    assert isinstance(wire, WireRequest)
    assert wire.method == "POST"
    assert wire.body == {
        "model": "claude-opus-4-5",
        "text": "hello",
        "profile": "anthropic:claude-opus",
    }
    assert wire.stream is False
    assert adapter.from_response({"text": "hello"}, "hello", "claude-opus-4-5", ANTHROPIC) == (
        "anthropic:hello"
    )


def test_stream_flag_reaches_the_wire_request() -> None:
    wire = EchoAdapter().to_request("hello", "claude-opus-4-5", PROFILE, stream=True)

    assert wire.stream is True


def test_missing_abstractmethod_blocks_instantiation() -> None:
    with pytest.raises(TypeError):
        HalfAdapter()  # type: ignore[abstract]


def test_missing_from_stream_blocks_instantiation_of_a_streaming_adapter() -> None:
    class NoStream(EchoAdapter, StreamingAdapter[str, str, str]):
        """A streaming adapter missing the one method streaming adds."""

    with pytest.raises(TypeError):
        NoStream()  # type: ignore[abstract]


def test_error_from_defaults_to_none() -> None:
    assert EchoAdapter().error_from(429, {"error": {"type": "rate_limit"}}, ANTHROPIC) is None
    assert EchoAdapter().error_from(500, None, ANTHROPIC) is None


def test_error_from_may_be_overridden_to_narrow_the_verdict() -> None:
    adapter = RefiningAdapter()

    refined = adapter.error_from(400, {"type": "overloaded"}, ANTHROPIC)

    assert isinstance(refined, ModelError)
    assert refined.retryable is True
    assert refined.failure_class == "provider"
    assert adapter.error_from(400, {"type": "invalid_request"}, ANTHROPIC) is None


def test_class_vars_are_readable_off_the_class() -> None:
    assert EchoAdapter.capability == CHAT
    assert EchoAdapter.dialect == ANTHROPIC_MESSAGES
    assert EchoAdapter.status == "stable"
    assert Adapter.status == "stable"
    assert EchoStreamingAdapter.status == "stub"


def test_a_streaming_adapter_is_also_an_adapter() -> None:
    adapter = EchoStreamingAdapter()

    assert issubclass(EchoStreamingAdapter, Adapter)
    assert isinstance(adapter, Adapter)
    assert adapter.to_request("hi", "claude-opus-4-5", PROFILE).body == {"text": "hi"}


async def test_from_stream_yields_one_delta_per_event() -> None:
    adapter = EchoStreamingAdapter()
    frames = events(SseEvent(event="delta", data="a"), SseEvent(event="delta", data="b"))

    deltas = [delta async for delta in adapter.from_stream(frames, "hi")]

    assert deltas == ["a", "b"]
