"""Executable specification for `AnthropicMessagesAdapter`, the M0 text subset.

An adapter is the only place in franca that knows what a provider's JSON looks like,
so what these tests pin is the JSON itself: whole bodies, compared as dicts, rather
than a key at a time. A body assertion that names every key is what makes an
accidental extra field -- a `stream: false`, a stray `system: ""` -- fail loudly
instead of riding along.

Four behaviours here are not visible in the adapter's signature. Anthropic rejects two
consecutive same-role messages, so adjacent IR items of one wire role must arrive as
*one* message with several content blocks. `max_tokens` is required by the API and is
a budget, so it is always emitted and never gated on sampling. `provider_options` is
merged last, which is the only reason it can be called an escape hatch. And the
adapter never stamps `trace`: the leaf does that, after the call, and an adapter that
guessed would be overwritten.

Anything the M0 subset cannot express -- a tool result, an image, a system-role turn
-- raises rather than being skipped. That choice is tested, because the alternative
failure is invisible: a silently dropped turn is a shorter prompt and a worse answer,
with nothing in the trace to say so.
"""

from typing import Any

import pytest

from franca.chat.adapter import DialectAdapter
from franca.chat.dialects.anthropic_messages import (
    DEFAULT_MAX_TOKENS,
    AnthropicMessagesAdapter,
)
from franca.chat.ir import Item, ModelResponse, PromptPackage, SystemBlock
from franca.chat.profile import ChatProfile
from franca.core.adapter import WireRequest
from franca.core.errors import ModelError
from franca.core.ids import ANTHROPIC, ANTHROPIC_MESSAGES, CHAT, OPENAI_CHAT
from franca.core.profile import BaseProfile
from franca.core.types import CallTrace, Usage

ADAPTER = AnthropicMessagesAdapter()
MODEL = "claude-opus-4-5"

PROFILE = ChatProfile(provider=ANTHROPIC, model_prefix="claude-opus")
"""A row that claims nothing: every flag unverified, no output-budget default."""

BUDGETED = ChatProfile(
    provider=ANTHROPIC,
    model_prefix="claude-opus",
    default_max_output_tokens=99,
)
"""The same row with an output budget, which is the only field this subset reads."""

PLAIN = BaseProfile(provider=ANTHROPIC, model_prefix="claude-opus")
"""A profile from another capability: it has no chat fields at all."""

# A real Anthropic 200, trimmed of nothing that matters: the served model carries the
# snapshot date the request did not, usage reports all four counters, and the content
# array holds a block this subset does not translate.
ANTHROPIC_200: dict[str, Any] = {
    "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4-5-20251101",
    "content": [
        {"type": "thinking", "thinking": "The user greeted me.", "signature": "EqQBCkYIBRgC"},
        {"type": "text", "text": "Hello! How can I help "},
        {"type": "text", "text": "you today?"},
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 12,
        "cache_creation_input_tokens": 4,
        "cache_read_input_tokens": 8,
        "output_tokens": 9,
        "service_tier": "standard",
    },
}


def user(text: str) -> Item:
    return Item(role="user", kind="text", text=text)


def assistant(text: str) -> Item:
    return Item(role="assistant", kind="text", text=text)


def body_of(wire: WireRequest) -> dict[str, Any]:
    assert wire.body is not None
    return wire.body


def render(pkg: PromptPackage, *, profile: BaseProfile = PROFILE, stream: bool = False) -> Any:
    return body_of(ADAPTER.to_request(pkg, MODEL, profile, stream=stream))


def as_dialect_adapter(adapter: DialectAdapter) -> DialectAdapter:
    """Pass the adapter through the chat-wide alias, so mypy checks it fits the seam."""
    return adapter


def test_the_class_carries_the_metadata_the_registry_indexes_on() -> None:
    assert AnthropicMessagesAdapter.capability == CHAT
    assert AnthropicMessagesAdapter.dialect == ANTHROPIC_MESSAGES
    assert AnthropicMessagesAdapter.status == "stable"


def test_the_adapter_satisfies_the_chat_dialect_alias() -> None:
    assert as_dialect_adapter(ADAPTER) is ADAPTER


def test_a_text_only_package_renders_exactly_this_body() -> None:
    wire = ADAPTER.to_request(PromptPackage(items=(user("hello"),)), MODEL, PROFILE)

    assert wire.method == "POST"
    assert wire.path is None, "the endpoint row owns /v1/messages"
    assert wire.stream is False
    assert wire.unverified == frozenset(), "nothing in this subset is profile-gated"
    assert wire.body == {
        "model": MODEL,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    }


def test_adjacent_items_of_one_wire_role_become_one_message_of_two_blocks() -> None:
    body = render(PromptPackage(items=(user("first"), user("second"))))

    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        }
    ]


def test_the_google_spelling_of_assistant_merges_with_assistant() -> None:
    items = (assistant("one"), Item(role="model", kind="text", text="two"))

    assert render(PromptPackage(items=items))["messages"] == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}],
        }
    ]


def test_alternating_roles_stay_separate_messages() -> None:
    pkg = PromptPackage(items=(user("q1"), assistant("a1"), user("q2")))

    assert render(pkg)["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "q1"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
        {"role": "user", "content": [{"type": "text", "text": "q2"}]},
    ]


def test_an_item_with_no_text_becomes_an_empty_block_rather_than_a_null() -> None:
    pkg = PromptPackage(items=(Item(role="user", kind="text"),))

    assert render(pkg)["messages"] == [{"role": "user", "content": [{"type": "text", "text": ""}]}]


def test_the_system_blocks_become_the_top_level_system_string() -> None:
    pkg = PromptPackage(
        system=(SystemBlock(text="be terse", position=1), SystemBlock(text="be kind")),
        items=(user("hi"),),
    )

    assert render(pkg)["system"] == "be kind\n\nbe terse"


def test_an_empty_system_omits_the_key_entirely() -> None:
    assert "system" not in render(PromptPackage(items=(user("hi"),)))
    assert "system" not in render(PromptPackage(system=(SystemBlock(text=""),), items=()))


@pytest.mark.parametrize(
    ("budget", "profile", "expected"),
    [
        (7, BUDGETED, 7),
        (7, PROFILE, 7),
        (None, BUDGETED, 99),
        (None, PROFILE, 1024),
        (None, PLAIN, 1024),
    ],
)
def test_max_tokens_prefers_the_package_then_the_profile_then_the_floor(
    budget: int | None,
    profile: BaseProfile,
    expected: int,
) -> None:
    pkg = PromptPackage(items=(user("hi"),), max_output_tokens=budget)

    assert render(pkg, profile=profile)["max_tokens"] == expected


def test_max_tokens_is_a_budget_and_survives_a_profile_that_forbids_sampling() -> None:
    profile = ChatProfile(
        provider=ANTHROPIC,
        model_prefix="claude-opus",
        sampling_allowed=False,
        default_max_output_tokens=256,
    )

    assert render(PromptPackage(items=(user("hi"),)), profile=profile)["max_tokens"] == 256


def test_streaming_adds_the_flag_and_not_streaming_omits_the_key() -> None:
    pkg = PromptPackage(items=(user("hi"),))
    streamed = ADAPTER.to_request(pkg, MODEL, PROFILE, stream=True)

    assert body_of(streamed)["stream"] is True
    assert streamed.stream is True
    assert "stream" not in render(pkg), "an absent key and a false are not the same body"


def test_provider_options_are_merged_last_and_win_over_the_adapter() -> None:
    pkg = PromptPackage(
        items=(user("hi"),),
        max_output_tokens=64,
        provider_options={
            ANTHROPIC_MESSAGES: {"max_tokens": 4096, "metadata": {"user_id": "u1"}},
        },
    )
    body = render(pkg)

    assert body["max_tokens"] == 4096, "the override wins over the computed budget"
    assert body["metadata"] == {"user_id": "u1"}, "and an unknown key is added verbatim"


def test_another_dialects_provider_options_are_left_alone() -> None:
    pkg = PromptPackage(
        items=(user("hi"),),
        provider_options={OPENAI_CHAT: {"max_tokens": 4096}},
    )

    assert render(pkg)["max_tokens"] == DEFAULT_MAX_TOKENS


@pytest.mark.parametrize(
    ("item", "needle"),
    [
        (Item(role="tool", kind="tool_result", call_id="c1", text="42"), "role='tool'"),
        (Item(role="system", kind="text", text="be terse"), "role='system'"),
        (Item(role="developer", kind="text", text="be terse"), "role='developer'"),
        (Item(role="user", kind="image"), "kind='image'"),
        (Item(role="assistant", kind="tool_call", name="lookup"), "kind='tool_call'"),
    ],
)
def test_an_item_this_subset_cannot_express_raises_rather_than_vanishing(
    item: Item,
    needle: str,
) -> None:
    pkg = PromptPackage(items=(user("hi"), item))

    with pytest.raises(ModelError, match=needle) as caught:
        ADAPTER.to_request(pkg, MODEL, PROFILE)

    assert caught.value.failure_class == "unsupported"
    assert caught.value.retryable is False
    assert caught.value.status is None
    assert caught.value.provider == ANTHROPIC
    assert "items[1]" in str(caught.value), "the message names which item, not its content"


def test_from_response_reads_the_items_meters_and_identity_off_a_real_body() -> None:
    res = ADAPTER.from_response(ANTHROPIC_200, PromptPackage(), MODEL, ANTHROPIC)

    assert res.items == (
        Item(role="assistant", kind="text", text="Hello! How can I help "),
        Item(role="assistant", kind="text", text="you today?"),
    ), "only text blocks; the thinking block is not mistranslated in this subset"
    assert res.text == "Hello! How can I help you today?"
    assert res.usage == Usage(
        input_tokens=12,
        output_tokens=9,
        cache_read_tokens=8,
        cache_write_tokens=4,
    )
    assert res.stop_reason == "end_turn"
    assert res.model == MODEL
    assert res.served_model == "claude-opus-4-5-20251101", "the snapshot the wire actually ran"
    assert res.provider == ANTHROPIC
    assert res.dialect == ANTHROPIC_MESSAGES
    assert res.raw == ANTHROPIC_200


def test_from_response_leaves_the_trace_for_the_leaf_to_stamp() -> None:
    res = ADAPTER.from_response(ANTHROPIC_200, PromptPackage(), MODEL, ANTHROPIC)
    trace = CallTrace(
        endpoint_id="anthropic/chat/messages",
        dialect=ANTHROPIC_MESSAGES,
        selection_via="native",
    )

    assert res.trace is None, "latency and attempts are facts about the call, not the body"
    assert res.with_trace(trace).trace == trace


def test_the_vendor_envelope_is_kept_for_debugging_but_never_dumped() -> None:
    res = ADAPTER.from_response(ANTHROPIC_200, PromptPackage(), MODEL, ANTHROPIC)

    assert res.raw["id"] == "msg_01XFDUDYJgAACzvnptvVoYEL"
    assert "raw" not in res.model_dump()


@pytest.mark.parametrize(
    ("raw", "why"),
    [
        ({}, "a body with no keys at all"),
        ({"content": []}, "an empty content array"),
        ({"content": [{"type": "thinking", "thinking": "..."}]}, "no text blocks"),
    ],
)
def test_a_body_with_no_text_blocks_yields_an_empty_items_tuple(
    raw: dict[str, Any],
    why: str,
) -> None:
    res = ADAPTER.from_response(raw, PromptPackage(), MODEL, ANTHROPIC)

    assert res.items == (), why
    assert res.text == ""
    assert res.stop_reason is None
    assert res.served_model is None


@pytest.mark.parametrize(
    ("usage", "why"),
    [
        ({}, "an empty usage object"),
        ({"input_tokens": 0}, "one counter reported, three absent"),
        ({"cache_read_input_tokens": None}, "a counter explicitly null"),
    ],
)
def test_every_missing_usage_counter_defaults_to_zero(usage: dict[str, Any], why: str) -> None:
    res = ADAPTER.from_response({"usage": usage}, PromptPackage(), MODEL, ANTHROPIC)

    assert res.usage == Usage(), why


def test_a_response_round_trips_back_into_the_next_request() -> None:
    pkg = PromptPackage(system=(SystemBlock(text="be terse"),), items=(user("hi"),))
    first = body_of(ADAPTER.to_request(pkg, MODEL, PROFILE))
    res: ModelResponse = ADAPTER.from_response(ANTHROPIC_200, pkg, MODEL, ANTHROPIC)

    next_turn = PromptPackage(
        system=pkg.system,
        items=(*pkg.items, *res.items, user("thanks")),
    )
    second = render(next_turn)

    assert first["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert second["system"] == "be terse"
    assert second["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello! How can I help "},
                {"type": "text", "text": "you today?"},
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "thanks"}]},
    ], "the answer's own items are legal turns, which is what makes a tool loop portable"
