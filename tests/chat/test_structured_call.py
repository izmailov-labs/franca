"""Structured output, end to end through the leaf, from a hand-written cassette.

The adapter tests pin each body and each parse in isolation; this one pins that they
meet. A JSON-schema package goes through `ChatModel.complete` -- prepare, render, the
connector's canonical encoding, the transport, the parse, the trace stamp -- and two
things only a whole call can show come out the other end: the request bytes are exactly
the documented `output_config.format` shape, and the assumption the adapter made
reaches `CallTrace.unverified`, where a policy or a span can see it.

Like every cassette here it is written from the wire's documented shape, not recorded,
so it is a specification of what franca must send rather than a capture of what it
happens to send today.
"""

from pathlib import Path
from typing import Any

from pydantic import SecretStr

from franca.chat.dialects.anthropic_messages import AnthropicMessagesAdapter
from franca.chat.endpoints import ANTHROPIC_MESSAGES_ENDPOINT
from franca.chat.ir import Item, OutputFormat, PromptPackage, SystemBlock
from franca.chat.model import ChatModel
from franca.chat.profile import ChatProfile
from franca.core.connector import Connector
from franca.core.enums import Strategy
from franca.core.ids import ANTHROPIC
from franca.testing import Cassette, FakeClock, ScriptedTransport, StaticKeyProvider

CASSETTE = Path(__file__).parent / "cassettes" / "anthropic_structured.json"

SCHEMA: dict[str, Any] = {
    "title": "Colour",
    "type": "object",
    "properties": {"colour": {"type": "string"}},
    "required": ["colour"],
    "additionalProperties": False,
}

PKG = PromptPackage(
    system=(SystemBlock(text="You are terse."),),
    items=(Item(role="user", kind="text", text="Name one primary colour."),),
    max_output_tokens=64,
    output_format=OutputFormat(kind="json", json_schema=SCHEMA),
)


def leaf(profile: ChatProfile) -> ChatModel:
    return ChatModel(
        model="claude-opus-4-5",
        connector=Connector(
            ANTHROPIC_MESSAGES_ENDPOINT,
            keys=StaticKeyProvider({ANTHROPIC: SecretStr("sk-test")}),
            transport=ScriptedTransport(Cassette.from_path(CASSETTE)),
            clock=FakeClock(),
        ),
        adapter=AnthropicMessagesAdapter(),
        profile=profile,
        clock=FakeClock(step=0.25),
    )


async def test_a_structured_call_sends_the_schema_and_returns_the_decoded_answer() -> None:
    res = await leaf(ChatProfile(provider=ANTHROPIC, model_prefix="")).complete(PKG)

    assert res.structured == {"colour": "blue"}
    assert res.text == '{"colour":"blue"}'
    assert res.stop_reason == "end_turn"


async def test_the_unmeasured_route_reaches_the_trace() -> None:
    res = await leaf(ChatProfile(provider=ANTHROPIC, model_prefix="")).complete(PKG)

    assert res.trace is not None
    assert res.trace.unverified == {"structured_output"}


async def test_a_measured_route_leaves_the_trace_clean() -> None:
    profile = ChatProfile(provider=ANTHROPIC, model_prefix="", structured_output=Strategy.native)
    res = await leaf(profile).complete(PKG)

    assert res.trace is not None
    assert res.trace.unverified == frozenset()
    assert res.structured == {"colour": "blue"}
