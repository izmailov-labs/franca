"""The OpenAI Chat Completions dialect, and where it diverges from Anthropic's.

Every wire shape below was captured from the live API rather than transcribed from
documentation, including the two error bodies -- both of which were obtained without
spending a token, since a bad key and an exhausted balance are both rejected before
inference.
"""

from typing import Any

import pytest

from franca.chat.dialects.openai_chat import DEFAULT_MAX_TOKENS, OpenAIChatAdapter
from franca.chat.ir import Item, PromptPackage, SystemBlock
from franca.chat.profile import ChatProfile
from franca.core.errors import ModelError
from franca.core.ids import OPENAI, OPENAI_CHAT

PROFILE = ChatProfile(provider=OPENAI, model_prefix="")
MODEL = "gpt-4.1-nano"

# Captured live, trimmed to the fields the adapter reads.
LIVE_200: dict[str, Any] = {
    "id": "chatcmpl-Bx1",
    "object": "chat.completion",
    "model": "gpt-4.1-nano-2025-04-14",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "OK"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 2,
        "total_tokens": 14,
        "prompt_tokens_details": {"cached_tokens": 0},
    },
}

# Captured live from a 429 on an account with no credits.
LIVE_429_QUOTA: dict[str, Any] = {
    "error": {
        "message": "You have no credits remaining. Add credits to continue using the API...",
        "type": "insufficient_quota",
        "param": None,
        "code": "credit_balance_exhausted",
    }
}

LIVE_429_RATE: dict[str, Any] = {
    "error": {
        "message": "Rate limit reached for gpt-4.1-nano in organization org-x on requests per min.",
        "type": "requests",
        "param": None,
        "code": "rate_limit_exceeded",
    }
}


def _adapter() -> OpenAIChatAdapter:
    return OpenAIChatAdapter()


# ---------------------------------------------------------------- to_request


def test_a_text_package_renders_the_whole_expected_body() -> None:
    pkg = PromptPackage(
        system=(SystemBlock(text="Be terse."),),
        items=(Item(role="user", kind="text", text="Hello"),),
        max_output_tokens=64,
    )
    wire = _adapter().to_request(pkg, MODEL, PROFILE)
    assert wire.body == {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hello"},
        ],
        "max_completion_tokens": 64,
    }
    assert wire.method == "POST"
    assert wire.path is None  # the endpoint row owns /v1/chat/completions


def test_the_system_prompt_is_a_message_not_a_top_level_field() -> None:
    """The first divergence from Anthropic, which takes a top-level `system` string."""
    pkg = PromptPackage(
        system=(SystemBlock(text="A"), SystemBlock(text="B")),
        items=(Item(role="user", kind="text", text="hi"),),
    )
    body = _adapter().to_request(pkg, MODEL, PROFILE).body
    assert body is not None
    assert "system" not in body
    assert body["messages"][0]["role"] == "system"


def test_no_system_text_emits_no_system_message() -> None:
    pkg = PromptPackage(items=(Item(role="user", kind="text", text="hi"),))
    body = _adapter().to_request(pkg, MODEL, PROFILE).body
    assert body is not None
    assert [m["role"] for m in body["messages"]] == ["user"]


def test_adjacent_same_role_turns_are_not_merged() -> None:
    """The second divergence: Anthropic rejects this shape, OpenAI accepts it."""
    pkg = PromptPackage(
        items=(
            Item(role="user", kind="text", text="one"),
            Item(role="user", kind="text", text="two"),
        )
    )
    body = _adapter().to_request(pkg, MODEL, PROFILE).body
    assert body is not None
    assert body["messages"] == [
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
    ]


def test_the_output_budget_uses_the_current_key_not_the_legacy_one() -> None:
    """The third divergence: `max_tokens` is rejected outright by reasoning models."""
    pkg = PromptPackage(items=(Item(role="user", kind="text", text="hi"),))
    body = _adapter().to_request(pkg, MODEL, PROFILE).body
    assert body is not None
    assert "max_completion_tokens" in body
    assert "max_tokens" not in body


@pytest.mark.parametrize(
    ("pkg_budget", "profile_budget", "expected"),
    [
        (64, 128, 64),
        (None, 128, 128),
        (None, None, DEFAULT_MAX_TOKENS),
        (0, 128, 128),
    ],
)
def test_output_budget_precedence(
    pkg_budget: int | None, profile_budget: int | None, expected: int
) -> None:
    pkg = PromptPackage(
        items=(Item(role="user", kind="text", text="hi"),), max_output_tokens=pkg_budget
    )
    profile = ChatProfile(
        provider=OPENAI, model_prefix="", default_max_output_tokens=profile_budget
    )
    body = _adapter().to_request(pkg, MODEL, profile).body
    assert body is not None
    assert body["max_completion_tokens"] == expected


def test_stream_true_adds_the_key_and_false_omits_it() -> None:
    pkg = PromptPackage(items=(Item(role="user", kind="text", text="hi"),))
    streamed = _adapter().to_request(pkg, MODEL, PROFILE, stream=True)
    plain = _adapter().to_request(pkg, MODEL, PROFILE, stream=False)
    assert streamed.body is not None
    assert plain.body is not None
    assert streamed.body["stream"] is True
    assert streamed.stream is True
    assert "stream" not in plain.body


def test_provider_options_are_merged_last_and_win() -> None:
    pkg = PromptPackage(
        items=(Item(role="user", kind="text", text="hi"),),
        max_output_tokens=64,
        provider_options={OPENAI_CHAT: {"max_completion_tokens": 999, "temperature": 0.0}},
    )
    body = _adapter().to_request(pkg, MODEL, PROFILE).body
    assert body is not None
    assert body["max_completion_tokens"] == 999  # overrode what the adapter computed
    assert body["temperature"] == 0.0  # and added a key the adapter has no field for


def test_provider_options_for_another_dialect_are_ignored() -> None:
    from franca.core.ids import ANTHROPIC_MESSAGES

    pkg = PromptPackage(
        items=(Item(role="user", kind="text", text="hi"),),
        provider_options={ANTHROPIC_MESSAGES: {"temperature": 1.0}},
    )
    body = _adapter().to_request(pkg, MODEL, PROFILE).body
    assert body is not None
    assert "temperature" not in body


@pytest.mark.parametrize(
    ("role", "kind"),
    [("user", "image"), ("tool", "text"), ("user", "tool_call")],
)
def test_an_item_outside_the_subset_raises_rather_than_dropping_the_turn(
    role: str, kind: str
) -> None:
    pkg = PromptPackage(items=(Item(role=role, kind=kind, text="x"),))  # type: ignore[arg-type]
    with pytest.raises(ModelError) as exc:
        _adapter().to_request(pkg, MODEL, PROFILE)
    assert exc.value.failure_class == "unsupported"
    assert exc.value.retryable is False
    assert "items[0]" in str(exc.value)


# --------------------------------------------------------------- from_response


def test_a_live_200_parses_into_the_ir() -> None:
    pkg = PromptPackage(items=(Item(role="user", kind="text", text="hi"),))
    res = _adapter().from_response(LIVE_200, pkg, MODEL, OPENAI)
    assert res.text == "OK"
    assert res.stop_reason == "stop"
    assert res.served_model == "gpt-4.1-nano-2025-04-14"
    assert res.dialect == OPENAI_CHAT
    assert res.provider == OPENAI


def test_the_differently_named_usage_meters_are_normalised() -> None:
    """`prompt_tokens`/`completion_tokens` here; `input_tokens`/`output_tokens` there."""
    pkg = PromptPackage(items=(Item(role="user", kind="text", text="hi"),))
    usage = _adapter().from_response(LIVE_200, pkg, MODEL, OPENAI).usage
    assert usage.input_tokens == 12
    assert usage.output_tokens == 2
    assert usage.cache_read_tokens == 0


def test_the_nested_cached_token_counter_is_read() -> None:
    raw = {
        **LIVE_200,
        "usage": {**LIVE_200["usage"], "prompt_tokens_details": {"cached_tokens": 7}},
    }
    pkg = PromptPackage(items=(Item(role="user", kind="text", text="hi"),))
    assert _adapter().from_response(raw, pkg, MODEL, OPENAI).usage.cache_read_tokens == 7


def test_missing_usage_defaults_every_meter_to_zero() -> None:
    pkg = PromptPackage(items=(Item(role="user", kind="text", text="hi"),))
    usage = _adapter().from_response({"choices": []}, pkg, MODEL, OPENAI).usage
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (0, 0, 0)


def test_an_empty_choices_list_yields_no_items() -> None:
    pkg = PromptPackage(items=(Item(role="user", kind="text", text="hi"),))
    res = _adapter().from_response({"choices": []}, pkg, MODEL, OPENAI)
    assert res.items == ()
    assert res.text == ""


def test_the_adapter_never_stamps_a_trace() -> None:
    """The leaf owns the trace; an adapter that guessed at it would be overwritten."""
    pkg = PromptPackage(items=(Item(role="user", kind="text", text="hi"),))
    assert _adapter().from_response(LIVE_200, pkg, MODEL, OPENAI).trace is None


def test_the_whole_body_survives_on_raw() -> None:
    pkg = PromptPackage(items=(Item(role="user", kind="text", text="hi"),))
    res = _adapter().from_response(LIVE_200, pkg, MODEL, OPENAI)
    assert res.raw["id"] == "chatcmpl-Bx1"
    assert "raw" not in res.model_dump()  # excluded from dumps, kept in memory


# ------------------------------------------------------------------ error_from


def test_a_quota_429_is_reclassified_as_permanent() -> None:
    """The divergence that matters most: two conditions, one status code."""
    refined = _adapter().error_from(429, LIVE_429_QUOTA, OPENAI)
    assert refined is not None
    assert refined.retryable is False
    assert refined.failure_class == "auth"
    assert refined.status == 429


def test_a_rate_limit_429_is_left_alone_for_the_connector_to_retry() -> None:
    assert _adapter().error_from(429, LIVE_429_RATE, OPENAI) is None


@pytest.mark.parametrize("status", [400, 401, 404, 500, 503])
def test_every_other_status_is_left_to_the_connector(status: int) -> None:
    assert _adapter().error_from(status, {"error": {"type": "whatever"}}, OPENAI) is None


def test_an_undecodable_429_body_is_left_retryable() -> None:
    """Absent evidence of a quota problem, the safe assumption is that waiting helps."""
    assert _adapter().error_from(429, None, OPENAI) is None


def test_a_429_whose_body_is_not_an_error_envelope_is_left_alone() -> None:
    assert _adapter().error_from(429, {"unexpected": "shape"}, OPENAI) is None
    assert _adapter().error_from(429, {"error": "a bare string"}, OPENAI) is None


def test_the_quota_message_does_not_echo_the_providers_text() -> None:
    """Franca writes its own message here, so nothing provider-controlled is copied."""
    refined = _adapter().error_from(429, LIVE_429_QUOTA, OPENAI)
    assert refined is not None
    assert "no remaining quota" in str(refined)
    assert refined.raw == LIVE_429_QUOTA  # the original stays available for inspection
