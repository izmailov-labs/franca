"""Executable specification for `ChatModel` and the built-in Anthropic endpoint row.

`ChatModel` adds exactly one thing to the core leaf, and this file is about that one
thing. `prepare` is the hook every capability gets to shape a request before an
adapter sees it, and the M0 subset of chat's hook does two jobs: it refuses a model
the profile says is retired, and it writes back the identifiers the caller left open.

Three properties are worth pinning rather than assuming.

The refusal happens *before any I/O*. That is the point of doing it in `prepare` at
all -- a retired model must cost nothing, not a round trip and a 404 -- so the retired
test asserts on the transport's call log, not just on the exception.

`retired` is tri-state, so only an explicit `True` refuses. A fresh profile row whose
`retired` is `None` means nobody checked, and reading that as "gone" would make every
unaudited model unusable; the parametrized case pins both `None` and `False`.

And filling is non-destructive. A package that pinned `provider` keeps its pin even
when this leaf is pointed somewhere else, because the package is the caller's record
of what they asked for. The counter-test therefore points the package at OpenAI while
the leaf talks to Anthropic and asserts that nothing was rewritten.

The endpoint row is tested as what it is -- data -- so the assertions are the row's
own fields and the URL they join into.
"""

import pytest
from pydantic import SecretStr, ValidationError

from franca.chat.dialects.anthropic_messages import AnthropicMessagesAdapter
from franca.chat.endpoints import ANTHROPIC_MESSAGES_ENDPOINT
from franca.chat.ir import Item, PromptPackage
from franca.chat.model import ChatClient, ChatModel
from franca.chat.profile import ChatProfile
from franca.core.connector import Connector
from franca.core.endpoint import Endpoint, EndpointFeatures
from franca.core.enums import AuthScheme
from franca.core.errors import ModelError
from franca.core.ids import (
    ANTHROPIC,
    ANTHROPIC_MESSAGES,
    CHAT,
    OPENAI,
    OPENAI_RESPONSES,
)
from franca.core.profile import BaseProfile
from franca.testing import Cassette, FakeClock, ScriptedTransport, StaticKeyProvider

MODEL = "claude-opus-4-5"
PKG = PromptPackage(items=(Item(role="user", kind="text", text="hello"),))


def leaf(*, retired: bool | None = None) -> tuple[ChatModel, ScriptedTransport]:
    transport = ScriptedTransport(Cassette())
    connector = Connector(
        ANTHROPIC_MESSAGES_ENDPOINT,
        keys=StaticKeyProvider({ANTHROPIC: SecretStr("sk-test")}),
        transport=transport,
        clock=FakeClock(),
    )
    model = ChatModel(
        model=MODEL,
        connector=connector,
        adapter=AnthropicMessagesAdapter(),
        profile=ChatProfile(provider=ANTHROPIC, model_prefix="", retired=retired),
        clock=FakeClock(step=0.25),
    )
    return model, transport


def test_the_capability_is_chat() -> None:
    assert ChatModel.capability == CHAT


def test_the_leaf_satisfies_the_chat_client_alias() -> None:
    """Conformance is structural, so mypy on this annotation is the real assertion."""
    model, _ = leaf()
    client: ChatClient = model
    assert client is model


def test_prepare_fills_the_identifiers_the_package_left_open() -> None:
    model, _ = leaf()
    prepared = model.prepare(PKG)
    assert prepared.provider == ANTHROPIC
    assert prepared.dialect == ANTHROPIC_MESSAGES
    assert prepared.model == MODEL


def test_prepare_copies_rather_than_mutating_the_caller_s_package() -> None:
    model, _ = leaf()
    prepared = model.prepare(PKG)
    assert prepared is not PKG
    assert PKG.provider is None
    assert PKG.dialect is None
    assert PKG.model is None
    assert prepared.items == PKG.items


def test_prepare_leaves_every_identifier_the_caller_pinned_alone() -> None:
    """A pin survives even when it disagrees with the leaf; the evidence is the point."""
    model, _ = leaf()
    pinned = PKG.model_copy(
        update={"provider": OPENAI, "dialect": OPENAI_RESPONSES, "model": "gpt-5.2"}
    )
    prepared = model.prepare(pinned)
    assert prepared.provider == OPENAI
    assert prepared.dialect == OPENAI_RESPONSES
    assert prepared.model == "gpt-5.2"


def test_prepare_returns_the_same_object_when_nothing_needs_filling() -> None:
    model, _ = leaf()
    complete = PKG.model_copy(
        update={"provider": ANTHROPIC, "dialect": ANTHROPIC_MESSAGES, "model": MODEL}
    )
    assert model.prepare(complete) is complete


def test_prepare_fills_one_field_without_touching_the_others() -> None:
    model, _ = leaf()
    half = PKG.model_copy(update={"model": "claude-opus-4-1"})
    prepared = model.prepare(half)
    assert prepared.model == "claude-opus-4-1"
    assert prepared.provider == ANTHROPIC
    assert prepared.dialect == ANTHROPIC_MESSAGES


async def test_a_retired_model_is_refused_before_any_request_is_sent() -> None:
    model, transport = leaf(retired=True)
    with pytest.raises(ModelError) as excinfo:
        await model.complete(PKG)
    error = excinfo.value
    assert error.failure_class == "unsupported"
    assert error.retryable is False
    assert error.status is None
    assert error.provider == ANTHROPIC
    assert MODEL in error.message
    assert transport.calls == []


@pytest.mark.parametrize("retired", [None, False])
def test_an_unchecked_or_living_model_is_not_refused(retired: bool | None) -> None:
    """`None` is "nobody checked", not "gone"; only an explicit `True` refuses."""
    model, _ = leaf(retired=retired)
    assert model.prepare(PKG).model == MODEL


def test_a_profile_without_the_retired_field_cannot_refuse() -> None:
    """The check narrows to `ChatProfile`; a base profile simply has nothing to say."""
    transport = ScriptedTransport(Cassette())
    connector = Connector(
        ANTHROPIC_MESSAGES_ENDPOINT,
        keys=StaticKeyProvider({ANTHROPIC: SecretStr("sk-test")}),
        transport=transport,
        clock=FakeClock(),
    )
    model = ChatModel(
        model=MODEL,
        connector=connector,
        adapter=AnthropicMessagesAdapter(),
        profile=BaseProfile(provider=ANTHROPIC, model_prefix=""),
        clock=FakeClock(),
    )
    assert model.prepare(PKG).model == MODEL


def test_the_builtin_row_is_the_documented_anthropic_messages_endpoint() -> None:
    row = ANTHROPIC_MESSAGES_ENDPOINT
    assert row.id == "anthropic/chat/messages"
    assert row.provider == ANTHROPIC
    assert row.capability == CHAT
    assert row.dialect == ANTHROPIC_MESSAGES
    assert row.base_url == "https://api.anthropic.com"
    assert row.path == "/v1/messages"
    assert row.auth is AuthScheme.x_api_key
    assert row.extra_headers == {"anthropic-version": "2023-06-01"}
    assert row.features == EndpointFeatures(streaming=True, caching=True, production=True)


def test_the_builtin_row_joins_into_the_messages_url() -> None:
    assert ANTHROPIC_MESSAGES_ENDPOINT.url() == "https://api.anthropic.com/v1/messages"
    assert (
        ANTHROPIC_MESSAGES_ENDPOINT.url("/v1/messages/count_tokens")
        == "https://api.anthropic.com/v1/messages/count_tokens"
    )


def test_the_builtin_row_validates_as_an_endpoint_and_survives_a_round_trip() -> None:
    """It is data, so the proof that it is well formed is that it re-validates."""
    again = Endpoint.model_validate_json(ANTHROPIC_MESSAGES_ENDPOINT.model_dump_json())
    assert again == ANTHROPIC_MESSAGES_ENDPOINT


def test_a_row_with_a_malformed_provider_slug_is_rejected() -> None:
    """`ProviderField` validates; the built-in row passing that is not an accident."""
    with pytest.raises(ValidationError) as excinfo:
        Endpoint.model_validate(
            ANTHROPIC_MESSAGES_ENDPOINT.model_dump() | {"provider": "Anthropic Inc"}
        )
    assert excinfo.value.errors()[0]["loc"] == ("provider",)
