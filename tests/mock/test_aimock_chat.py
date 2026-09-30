"""The mock suite: the live call path, end to end, against aimock instead of a vendor.

Between the cassettes and the contract suite there is a gap. `ScriptedTransport`
never opens a socket, so it cannot show that `HttpxTransport` sends what the connector
built, that a `Retry-After` survives a real header parse, or that SSE framing holds up
when the bytes arrive chunked. The contract suite can, but it costs money, needs keys,
and answers with prose that no assertion can pin. aimock closes the gap: real HTTP,
real server-sent events, no key and no bill, and a reply that is exactly what
`fixtures/chat.json` says it is.

So the assertions here are precise where the contract suite's are thin: the text *is*
"Blue.", the meters *are* 17 and 5, the served model *is* what the fixture named. Every
case runs against both wires, because the point of the IR is that one package and one
set of assertions hold whichever dialect is underneath; the row per dialect carries
only what genuinely differs -- how it spells a stop reason, where a tool call sits on
the raw body, how its deltas reassemble.

Two things this suite deliberately does not do. It does not replace the cassettes --
those are the specification of the wire, and a mock that renders its own idea of a
provider's shape is evidence rather than specification. And it does not run by
default: `mock` is deselected in `addopts` next to `contract`, because it needs node
on the machine; `make test-mock` selects it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from franca.chat.dialects.anthropic_messages import AnthropicMessagesAdapter
from franca.chat.dialects.google_generate_content import GoogleGenerateContentAdapter
from franca.chat.dialects.openai_chat import OpenAIChatAdapter
from franca.chat.endpoints import (
    ANTHROPIC_MESSAGES_ENDPOINT,
    DEEPSEEK_CHAT_ENDPOINT,
    GOOGLE_GENERATE_CONTENT_ENDPOINT,
    OPENAI_CHAT_ENDPOINT,
    XAI_CHAT_ENDPOINT,
)
from franca.chat.ir import Item, ModelResponse, PromptPackage, SystemBlock
from franca.chat.model import ChatModel
from franca.chat.profile import ChatProfile
from franca.core.clock import AsyncioClock
from franca.core.connector import Connector
from franca.core.errors import ModelError
from franca.core.ids import ANTHROPIC, DEEPSEEK, GOOGLE, OPENAI, XAI
from franca.core.keys import StaticKeyProvider
from franca.transports.httpx import HttpxTransport

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping, Sequence

    from franca.core.adapter import Adapter
    from franca.core.endpoint import Endpoint
    from franca.core.ids import Provider
    from franca.core.sse import SseEvent

pytestmark = pytest.mark.mock

MODEL = "mock-model"
"""What the caller asks for. The fixture decides what the wire says served it."""

SENTENCE = "one two three four five six"
"""The `Count to six.` fixture's text, streamed four characters at a time."""


def _openai_stream_text(events: Sequence[SseEvent]) -> str:
    """Reassemble `delta.content` across Chat Completions chunks."""
    parts: list[str] = []
    for event in events:
        chunk = json.loads(event.data)
        parts.extend(choice.get("delta", {}).get("content") or "" for choice in chunk["choices"])
    return "".join(parts)


def _anthropic_stream_text(events: Sequence[SseEvent]) -> str:
    """Reassemble `text_delta` payloads across Messages `content_block_delta` events."""
    return "".join(
        json.loads(event.data)["delta"]["text"]
        for event in events
        if event.event == "content_block_delta"
    )


def _openai_tool_name(raw: Mapping[str, Any]) -> str:
    name: str = raw["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
    return name


def _anthropic_tool_name(raw: Mapping[str, Any]) -> str:
    name: str = raw["content"][0]["name"]
    return name


def _google_stream_text(events: Sequence[SseEvent]) -> str:
    """Reassemble the text parts of each candidate across `generateContent` chunks."""
    parts: list[str] = []
    for event in events:
        chunk = json.loads(event.data)
        for candidate in chunk.get("candidates") or ():
            parts.extend(part.get("text") or "" for part in candidate["content"]["parts"])
    return "".join(parts)


def _google_tool_name(raw: Mapping[str, Any]) -> str:
    name: str = raw["candidates"][0]["content"]["parts"][0]["functionCall"]["name"]
    return name


@dataclass(frozen=True)
class MockCase:
    """One wire the mock suite drives, and the few facts that differ per wire.

    Attributes:
        name: The pytest id.
        provider: The provider slug the endpoint row carries.
        endpoint: The built-in row; the suite rebases its `base_url` onto the mock.
        adapter: Builds the dialect adapter; a factory so each test gets a fresh one.
        stop_reason: How this wire spells a fixture's `finishReason: "stop"`.
        tool_stop_reason: How it spells a reply that is a tool call.
        tool_name: Where the tool call's name sits on the raw body.
        stream_text: How the wire's deltas reassemble into the streamed text.
        events_named: Whether the wire sets `event:` on its SSE blocks.
        served_model: What the wire reports as the model that answered, or `None` where
            it reports nothing. The fixture names `mock-served-model` and aimock echoes
            it on every wire that has a field for it; Gemini's `modelVersion` is not one
            it fills, and `None` is the honest reading of an absent field.
    """

    name: str
    provider: Provider
    endpoint: Endpoint
    adapter: Callable[[], Adapter[PromptPackage, ModelResponse]]
    stop_reason: str
    tool_stop_reason: str
    tool_name: Callable[[Mapping[str, Any]], str]
    stream_text: Callable[[Sequence[SseEvent]], str]
    events_named: bool
    served_model: str | None = "mock-served-model"


CASES: tuple[MockCase, ...] = (
    MockCase(
        name="openai_chat",
        provider=OPENAI,
        endpoint=OPENAI_CHAT_ENDPOINT,
        adapter=OpenAIChatAdapter,
        stop_reason="stop",
        tool_stop_reason="tool_calls",
        tool_name=_openai_tool_name,
        stream_text=_openai_stream_text,
        events_named=False,
    ),
    MockCase(
        name="anthropic_messages",
        provider=ANTHROPIC,
        endpoint=ANTHROPIC_MESSAGES_ENDPOINT,
        adapter=AnthropicMessagesAdapter,
        stop_reason="end_turn",
        tool_stop_reason="tool_use",
        tool_name=_anthropic_tool_name,
        stream_text=_anthropic_stream_text,
        events_named=True,
    ),
    MockCase(
        name="google_generate_content",
        provider=GOOGLE,
        endpoint=GOOGLE_GENERATE_CONTENT_ENDPOINT,
        adapter=GoogleGenerateContentAdapter,
        stop_reason="STOP",
        tool_stop_reason="FUNCTION_CALL",
        tool_name=_google_tool_name,
        stream_text=_google_stream_text,
        events_named=False,
        served_model=None,
    ),
    MockCase(
        name="xai_chat",
        provider=XAI,
        endpoint=XAI_CHAT_ENDPOINT,
        adapter=OpenAIChatAdapter,
        stop_reason="stop",
        tool_stop_reason="tool_calls",
        tool_name=_openai_tool_name,
        stream_text=_openai_stream_text,
        events_named=False,
    ),
    MockCase(
        name="deepseek_chat",
        provider=DEEPSEEK,
        endpoint=DEEPSEEK_CHAT_ENDPOINT,
        adapter=OpenAIChatAdapter,
        stop_reason="stop",
        tool_stop_reason="tool_calls",
        tool_name=_openai_tool_name,
        stream_text=_openai_stream_text,
        events_named=False,
    ),
)
"""One row per wire franca can drive, not one per vendor.

The last two rows carry no new adapter and no new assertions, and that is exactly what
they are for. xAI and DeepSeek reach `OpenAIChatAdapter` through their own endpoint
rows, so what these cases actually exercise is the rows themselves: that the path is
right, that the auth scheme is the one the provider wants, and that the provider slug
travels through to the response and the trace. A typo in a row is otherwise invisible
until somebody with a key finds it.

Google is the opposite case -- a third dialect, with the model in the path, its own
role vocabulary, its own meters and no `event:` names on its stream -- and it shares
every assertion below with the other four. That sharing is the IR's whole claim.
"""

IDS = [case.name for case in CASES]


def _ask(text: str) -> PromptPackage:
    """A package with a system block, so both wires render one on every call."""
    return PromptPackage(
        system=(SystemBlock(text="You are terse."),),
        items=(Item(role="user", kind="text", text=text),),
        max_output_tokens=32,
    )


@pytest.fixture
async def transport() -> AsyncIterator[HttpxTransport]:
    transport = HttpxTransport()
    try:
        yield transport
    finally:
        await transport.aclose()


def _model(case: MockCase, url: str, key: str, transport: HttpxTransport) -> ChatModel:
    """Assemble the live leaf, with the endpoint row rebased onto the mock.

    Only `base_url` changes. The row's `id`, path, auth scheme and version header are
    exactly the production ones, which is what makes a pass here say something about
    the production path.
    """
    endpoint = case.endpoint.model_copy(update={"base_url": url})
    return ChatModel(
        model=MODEL,
        connector=Connector(
            endpoint,
            keys=StaticKeyProvider({case.provider: SecretStr(key)}),
            transport=transport,
            clock=AsyncioClock(),
        ),
        adapter=case.adapter(),
        profile=ChatProfile(provider=case.provider, model_prefix=""),
        clock=AsyncioClock(),
    )


# --------------------------------------------------------------------- the round trip


@pytest.mark.parametrize("case", CASES, ids=IDS)
async def test_a_completion_round_trips_over_real_http(
    case: MockCase, aimock_url: str, aimock_key: str, transport: HttpxTransport
) -> None:
    model = _model(case, aimock_url, aimock_key, transport)

    res = await model.complete(_ask("Name one primary colour."))

    assert res.text == "Blue."
    assert res.usage.input_tokens == 17
    assert res.usage.output_tokens == 5
    assert res.usage.cache_read_tokens == 0
    assert res.usage.cache_write_tokens == 0
    assert res.served_model == case.served_model
    assert res.stop_reason == case.stop_reason
    assert res.model == MODEL
    assert res.provider == case.provider
    assert res.dialect == case.endpoint.dialect

    trace = res.trace
    assert trace is not None
    assert trace.endpoint_id == case.endpoint.id, "rebasing the URL must not rename the row"
    assert trace.dialect == case.endpoint.dialect
    assert trace.selection_via == "native"
    assert trace.attempts == 1
    assert trace.latency_ms > 0.0
    assert trace.unverified == frozenset()


@pytest.mark.parametrize("case", CASES, ids=IDS)
async def test_the_key_travels_in_the_header_the_row_names(
    case: MockCase, aimock_url: str, aimock_spawned: bool, transport: HttpxTransport
) -> None:
    """The spawned server enforces `AIMOCK_API_KEYS`, on each wire's own header.

    A wrong key is therefore a real 401 from the server, not a fixture: the proof that
    `Connector.headers()` chose `Authorization: Bearer` for one row and `x-api-key` for
    the other, rather than sending a key somewhere the server does not look.
    """
    if not aimock_spawned:
        pytest.skip("key enforcement is only known for a server this session spawned")
    model = _model(case, aimock_url, "not-the-configured-key", transport)

    with pytest.raises(ModelError) as excinfo:
        await model.complete(_ask("Name one primary colour."))

    error = excinfo.value
    assert error.status == 401
    assert error.failure_class == "auth"
    assert error.retryable is False
    assert error.provider == case.provider
    assert "not-the-configured-key" not in str(error)


# ------------------------------------------------------------- the status map, live


@pytest.mark.parametrize("case", CASES, ids=IDS)
async def test_a_request_no_fixture_matches_is_a_visible_provider_error(
    case: MockCase, aimock_url: str, aimock_key: str, transport: HttpxTransport
) -> None:
    """An unmatched request comes back from aimock as a 404 that names the problem.

    That lands as a non-retryable provider error carrying the mock's message, which is
    the signal a developer sees when a new prompt has no fixture yet. The fix is a
    fixture, never a looser classification.
    """
    model = _model(case, aimock_url, aimock_key, transport)

    with pytest.raises(ModelError) as excinfo:
        await model.complete(_ask("There is no fixture for this sentence."))

    error = excinfo.value
    assert error.status == 404
    assert error.failure_class == "provider"
    assert error.retryable is False
    assert "No fixture matched" in error.message


@pytest.mark.parametrize("case", CASES, ids=IDS)
async def test_a_429_arrives_with_retry_after_parsed_from_a_real_header(
    case: MockCase, aimock_url: str, aimock_key: str, transport: HttpxTransport
) -> None:
    model = _model(case, aimock_url, aimock_key, transport)

    with pytest.raises(ModelError) as excinfo:
        await model.complete(_ask("Rate limit me."))

    error = excinfo.value
    assert error.status == 429
    assert error.failure_class == "rate_limit"
    assert error.retryable is True
    assert error.message == "slow down"
    assert error.retry_after_s is not None, "aimock sends Retry-After on every 429"
    assert error.retry_after_s >= 0.0
    assert error.raw is not None


@pytest.mark.parametrize("case", CASES, ids=IDS)
async def test_a_500_is_a_retryable_provider_failure(
    case: MockCase, aimock_url: str, aimock_key: str, transport: HttpxTransport
) -> None:
    model = _model(case, aimock_url, aimock_key, transport)

    with pytest.raises(ModelError) as excinfo:
        await model.complete(_ask("Break."))

    error = excinfo.value
    assert error.status == 500
    assert error.failure_class == "provider"
    assert error.retryable is True
    assert error.message == "mock upstream failure"


# ------------------------------------------------------------ what M0 leaves on raw


@pytest.mark.parametrize("case", CASES, ids=IDS)
async def test_a_tool_call_reply_is_kept_on_raw_until_the_ir_can_carry_it(
    case: MockCase, aimock_url: str, aimock_key: str, transport: HttpxTransport
) -> None:
    """Tools are not in the M0 IR, so a tool-call reply yields no items and loses nothing.

    Pins the documented behaviour of both adapters: a `tool_calls` message or a
    `tool_use` block is passed over rather than mistranslated, and survives on `raw`.
    When the full IR lands this test changes with it.
    """
    res = await _model(case, aimock_url, aimock_key, transport).complete(_ask("Use a tool."))

    assert res.items == ()
    assert res.stop_reason == case.tool_stop_reason
    assert case.tool_name(res.raw) == "get_weather"


# ----------------------------------------------------------------------- streaming


@pytest.mark.parametrize("case", CASES, ids=IDS)
async def test_a_streaming_reply_frames_as_server_sent_events(
    case: MockCase, aimock_url: str, aimock_key: str, transport: HttpxTransport
) -> None:
    """`Connector.stream` over a real chunked SSE body, framed by `parse_sse`.

    The chat leaf does not stream until M1, so this drives the connector directly with
    the request the adapter builds for `stream=True`. The fixture is sent four
    characters at a time, so the text only comes back whole if every delta was framed
    and none was lost or doubled at a chunk boundary.
    """
    model = _model(case, aimock_url, aimock_key, transport)
    wire = model.build(_ask("Count to six."), stream=True)

    async with model.connector.stream(wire) as events:
        got = [event async for event in events]

    assert len(got) >= 7, "four characters per chunk means at least seven deltas"
    assert all(event.data != "[DONE]" for event in got), "the sentinel ends the stream"
    assert all((event.event is not None) is case.events_named for event in got)
    assert case.stream_text(got) == SENTENCE
