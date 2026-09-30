"""The M0 acceptance test: one chat call, end to end, from a hand-written cassette.

This is the milestone gate rather than a unit test. Every seam the walking skeleton
is made of appears in the single `await` below -- `ChatModel.prepare`, the adapter's
`to_request`, the `Connector`'s URL join and canonical JSON encoding and auth header,
the `Transport`, the adapter's `from_response`, and the leaf's `CallTrace` stamp --
and the point is that all of them are exercised together, by the same call a user
would make, with nothing stubbed in between.

The cassettes are hand-written on purpose. M0 has no recorder (that lands in M1 with
the streaming replay), so the JSON files under `cassettes/` were typed from the
Messages API documentation and are the *specification* of the wire, not a capture of
it. That is why the request assertion is byte-exact rather than a subset match: a
cassette that records what franca happens to send today would agree with any future
regression, whereas one that states what Anthropic accepts disagrees loudly.

Determinism comes from three injected doubles and nothing else. `ScriptedTransport`
replaces the socket, `StaticKeyProvider` replaces the environment, and two `FakeClock`s
replace time -- the leaf's is stepped by 0.25s per reading, so `latency_ms` is exactly
250.0 rather than "some small number". Change any of the three for a real one and this
file becomes the `contract` test in `tests/contract/test_live.py`.

The last test wraps the leaf in `wrap(leaf, identity)` and asserts the same facts. It
is the proof that `ChatModel` satisfies `Client` structurally: if it did not, the leaf
could never be composed with middleware, and every layer M2 adds would be unreachable.
"""

from pathlib import Path

import pytest
from pydantic import SecretStr

from franca.chat.dialects.anthropic_messages import AnthropicMessagesAdapter
from franca.chat.endpoints import ANTHROPIC_MESSAGES_ENDPOINT
from franca.chat.ir import Item, PromptPackage, SystemBlock
from franca.chat.model import ChatClient, ChatModel
from franca.chat.profile import ChatProfile
from franca.core.client import identity, wrap
from franca.core.connector import Connector
from franca.core.errors import ModelError
from franca.core.ids import ANTHROPIC, ANTHROPIC_MESSAGES
from franca.testing import Cassette, FakeClock, ScriptedTransport, StaticKeyProvider

CASSETTES = Path(__file__).parent / "cassettes"

MODEL = "claude-opus-4-5"

PKG = PromptPackage(
    system=(SystemBlock(text="You are terse."),),
    items=(Item(role="user", kind="text", text="Say hi in three words."),),
    max_output_tokens=64,
)

EXPECTED_REQUEST = (
    b'{"max_tokens":64,'
    b'"messages":[{"content":[{"text":"Say hi in three words.","type":"text"}],"role":"user"}],'
    b'"model":"claude-opus-4-5",'
    b'"system":"You are terse."}'
)


def leaf(cassette: str) -> tuple[ChatModel, ScriptedTransport]:
    transport = ScriptedTransport(Cassette.from_path(CASSETTES / cassette))
    return (
        ChatModel(
            model=MODEL,
            connector=Connector(
                ANTHROPIC_MESSAGES_ENDPOINT,
                keys=StaticKeyProvider({ANTHROPIC: SecretStr("sk-test")}),
                transport=transport,
                clock=FakeClock(),
            ),
            adapter=AnthropicMessagesAdapter(),
            profile=ChatProfile(provider=ANTHROPIC, model_prefix=""),
            clock=FakeClock(step=0.25),
        ),
        transport,
    )


async def test_the_first_call_answers_with_text_meters_and_a_trace() -> None:
    """M0's acceptance criterion, in one call: the text, the four meters, the trace."""
    model, transport = leaf("anthropic_text.json")

    res = await model.complete(PKG)

    assert res.text == "Hi there, friend."

    assert res.usage.input_tokens == 17
    assert res.usage.output_tokens == 5
    assert res.usage.cache_read_tokens == 1024
    assert res.usage.cache_write_tokens == 256

    trace = res.trace
    assert trace is not None
    assert trace.endpoint_id == "anthropic/chat/messages"
    assert trace.dialect == ANTHROPIC_MESSAGES
    assert trace.selection_via == "native"
    assert trace.latency_ms == 250.0
    assert trace.attempts == 1
    assert trace.unverified == frozenset()

    assert res.served_model == "claude-opus-4-5-20251101"
    assert res.stop_reason == "end_turn"
    assert res.model == MODEL
    assert res.provider == ANTHROPIC
    assert res.dialect == ANTHROPIC_MESSAGES

    method, url, content, headers = transport.calls[0]
    assert len(transport.calls) == 1
    assert method == "POST"
    assert url == "https://api.anthropic.com/v1/messages"
    assert content == EXPECTED_REQUEST
    assert headers["x-api-key"] == "sk-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["Content-Type"] == "application/json"


async def test_a_401_arrives_as_a_non_retryable_auth_failure() -> None:
    model, _ = leaf("anthropic_401.json")

    with pytest.raises(ModelError) as excinfo:
        await model.complete(PKG)

    error = excinfo.value
    assert error.failure_class == "auth"
    assert error.retryable is False
    assert error.status == 401
    assert error.provider == ANTHROPIC
    assert error.message == "invalid x-api-key"
    assert error.retry_after_s is None


async def test_a_429_arrives_as_a_retryable_rate_limit_with_its_backoff() -> None:
    model, _ = leaf("anthropic_429.json")

    with pytest.raises(ModelError) as excinfo:
        await model.complete(PKG)

    error = excinfo.value
    assert error.failure_class == "rate_limit"
    assert error.retryable is True
    assert error.status == 429
    assert error.retry_after_s == 30.0
    assert error.message == "Number of requests has exceeded your rate limit."


async def test_the_same_call_through_an_identity_layer_is_indistinguishable() -> None:
    """`wrap` composing the leaf is the proof that `ChatModel` satisfies `Client`."""
    model, transport = leaf("anthropic_text.json")
    client: ChatClient = wrap(model, identity)

    res = await client.complete(PKG)

    assert res.text == "Hi there, friend."
    assert res.usage.input_tokens == 17
    assert res.trace is not None
    assert res.trace.latency_ms == 250.0
    assert res.trace.selection_via == "native"
    assert transport.calls[0][2] == EXPECTED_REQUEST


def test_wrapping_nothing_returns_the_leaf_itself() -> None:
    """An empty pipeline costs nothing, so the M0 route is the leaf, not a wrapper."""
    model, _ = leaf("anthropic_text.json")
    assert wrap(model) is model
