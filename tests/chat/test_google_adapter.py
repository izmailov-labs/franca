"""The Google `generateContent` dialect, and the three ways it is unlike the other two.

The model in the path, the streaming method in the path, and camelCase everywhere. Each
one is asserted here rather than argued for, because each is a place where "just add a
flag to the OpenAI adapter" would have produced a request Gemini rejects.

On provenance, which this repo cares about: **the two error bodies below were captured
live** on 2026-09-16, by sending a deliberately wrong key to
`generativelanguage.googleapis.com` -- free, since a bad credential is rejected before
inference. **The 200 body was not.** No working Gemini key was available, so it is
written from the wire's documented shape, which makes it a specification of what the
adapter must read rather than a recording of what one model happened to send. The
difference is why `_GOOGLE_DEFAULT` claims no capabilities: a shape can be written down,
a capability has to be sent.
"""

from typing import Any

import pytest

from franca.chat.dialects.google_generate_content import (
    DEFAULT_MAX_TOKENS,
    GoogleGenerateContentAdapter,
    path_for,
)
from franca.chat.ir import Item, PromptPackage, SystemBlock
from franca.chat.profile import ChatProfile
from franca.core.errors import ModelError
from franca.core.ids import GOOGLE, GOOGLE_GENERATE_CONTENT

PROFILE = ChatProfile(provider=GOOGLE, model_prefix="")
MODEL = "gemini-2.0-flash"

# Written from the documented shape; see the module docstring.
WIRE_200: dict[str, Any] = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "Blue."}]},
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 17,
        "candidatesTokenCount": 5,
        "cachedContentTokenCount": 8,
        "totalTokenCount": 22,
    },
    "modelVersion": "gemini-2.0-flash-001",
}

# Captured live: a bad API key, which this wire answers with 400, not 401.
LIVE_400_BAD_KEY: dict[str, Any] = {
    "error": {
        "code": 400,
        "message": "API key not valid. Please pass a valid API key.",
        "status": "INVALID_ARGUMENT",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "API_KEY_INVALID",
                "domain": "googleapis.com",
                "metadata": {"service": "generativelanguage.googleapis.com"},
            }
        ],
    }
}

# Captured live: an OAuth-shaped token in the x-goog-api-key header. Already 401, so the
# connector classifies it correctly on its own and the adapter must leave it alone.
LIVE_401_WRONG_CREDENTIAL_TYPE: dict[str, Any] = {
    "error": {
        "code": 401,
        "message": "Request had invalid authentication credentials.",
        "status": "UNAUTHENTICATED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "ACCESS_TOKEN_TYPE_UNSUPPORTED",
            }
        ],
    }
}

# A real 400 about the request rather than the key: no ErrorInfo reason to latch onto.
LIVE_400_BAD_REQUEST: dict[str, Any] = {
    "error": {
        "code": 400,
        "message": "Invalid JSON payload received. Unknown name 'model'.",
        "status": "INVALID_ARGUMENT",
    }
}


def _adapter() -> GoogleGenerateContentAdapter:
    return GoogleGenerateContentAdapter()


def _package(*texts: str, system: str | None = None, budget: int | None = None) -> PromptPackage:
    return PromptPackage(
        system=(SystemBlock(text=system),) if system else (),
        items=tuple(Item(role="user", kind="text", text=text) for text in texts),
        max_output_tokens=budget,
    )


# ------------------------------------------------------------------ the path


def test_the_model_travels_in_the_path_and_never_in_the_body() -> None:
    """The one claim that makes this a separate dialect rather than a flag."""
    wire = _adapter().to_request(_package("hi"), MODEL, PROFILE)

    assert wire.path == "/v1beta/models/gemini-2.0-flash:generateContent"
    assert wire.body is not None
    assert "model" not in wire.body


def test_streaming_swaps_the_method_and_adds_no_body_key() -> None:
    """No `stream: true` on this wire: the method name and `alt=sse` are the signal."""
    wire = _adapter().to_request(_package("hi"), MODEL, PROFILE, stream=True)

    assert wire.path == "/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
    assert wire.stream is True
    assert wire.body is not None
    assert "stream" not in wire.body


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-2.0-flash", "/v1beta/models/gemini-2.0-flash:generateContent"),
        ("models/gemini-2.0-flash", "/v1beta/models/gemini-2.0-flash:generateContent"),
        ("tunedModels/abc-123", "/v1beta/models/tunedModels/abc-123:generateContent"),
    ],
)
def test_the_qualified_and_bare_spellings_reach_the_same_path(model: str, expected: str) -> None:
    assert path_for(model, stream=False) == expected


def test_a_model_name_cannot_rewrite_the_query_string() -> None:
    """`?` and `#` are escaped; a caller's model id must not reshape the request."""
    assert path_for("x?alt=json#z", stream=False) == (
        "/v1beta/models/x%3Falt%3Djson%23z:generateContent"
    )


# ------------------------------------------------------------------ the body


def test_a_text_package_renders_the_whole_expected_body() -> None:
    wire = _adapter().to_request(
        _package("Say hi.", system="You are terse.", budget=64), MODEL, PROFILE
    )

    assert wire.body == {
        "contents": [{"role": "user", "parts": [{"text": "Say hi."}]}],
        "generationConfig": {"maxOutputTokens": 64},
        "systemInstruction": {"parts": [{"text": "You are terse."}]},
    }


def test_the_system_prompt_is_its_own_field_not_a_turn() -> None:
    wire = _adapter().to_request(_package("hi", system="Be brief."), MODEL, PROFILE)

    assert wire.body is not None
    assert wire.body["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
    assert [c["role"] for c in wire.body["contents"]] == ["user"]


def test_no_system_text_emits_no_system_instruction() -> None:
    wire = _adapter().to_request(_package("hi"), MODEL, PROFILE)

    assert wire.body is not None
    assert "systemInstruction" not in wire.body


def test_the_assistant_is_spelled_model() -> None:
    package = PromptPackage(
        items=(
            Item(role="user", kind="text", text="one"),
            Item(role="assistant", kind="text", text="two"),
        )
    )
    wire = _adapter().to_request(package, MODEL, PROFILE)

    assert wire.body is not None
    assert [c["role"] for c in wire.body["contents"]] == ["user", "model"]


def test_adjacent_same_role_turns_merge_into_one_entry() -> None:
    """Gemini expects alternation, so a run of user items becomes one multi-part entry."""
    wire = _adapter().to_request(_package("one", "two"), MODEL, PROFILE)

    assert wire.body is not None
    assert wire.body["contents"] == [{"role": "user", "parts": [{"text": "one"}, {"text": "two"}]}]


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(None, DEFAULT_MAX_TOKENS), (0, DEFAULT_MAX_TOKENS), (16, 16)],
)
def test_output_budget_precedence(budget: int | None, expected: int) -> None:
    wire = _adapter().to_request(_package("hi", budget=budget), MODEL, PROFILE)

    assert wire.body is not None
    assert wire.body["generationConfig"] == {"maxOutputTokens": expected}


def test_the_profile_default_fills_an_unset_budget() -> None:
    profile = ChatProfile(provider=GOOGLE, model_prefix="", default_max_output_tokens=99)
    wire = _adapter().to_request(_package("hi"), MODEL, profile)

    assert wire.body is not None
    assert wire.body["generationConfig"] == {"maxOutputTokens": 99}


def test_provider_options_are_merged_last_and_win() -> None:
    package = PromptPackage(
        items=(Item(role="user", kind="text", text="hi"),),
        provider_options={
            GOOGLE_GENERATE_CONTENT: {
                "generationConfig": {"maxOutputTokens": 8, "temperature": 0.2},
                "safetySettings": [],
            }
        },
    )
    wire = _adapter().to_request(package, MODEL, PROFILE)

    assert wire.body is not None
    assert wire.body["generationConfig"] == {"maxOutputTokens": 8, "temperature": 0.2}
    assert wire.body["safetySettings"] == []


def test_provider_options_for_another_dialect_are_ignored() -> None:
    package = PromptPackage(
        items=(Item(role="user", kind="text", text="hi"),),
        provider_options={GOOGLE_GENERATE_CONTENT: {}, "anthropic_messages": {"top_k": 5}},  # type: ignore[dict-item]
    )
    wire = _adapter().to_request(package, MODEL, PROFILE)

    assert wire.body is not None
    assert "top_k" not in wire.body


@pytest.mark.parametrize(
    "item",
    [
        Item(role="system", kind="text", text="x"),
        Item(role="tool", kind="text", text="x"),
        Item(role="user", kind="image", text="x"),
        Item(role="user", kind="tool_result", text="x"),
    ],
)
def test_an_item_outside_the_subset_raises_rather_than_dropping_the_turn(item: Item) -> None:
    with pytest.raises(ModelError) as excinfo:
        _adapter().to_request(PromptPackage(items=(item,)), MODEL, PROFILE)

    assert excinfo.value.failure_class == "unsupported"
    assert excinfo.value.retryable is False
    assert excinfo.value.status is None


# --------------------------------------------------------------- the response


def test_a_200_parses_into_the_ir() -> None:
    res = _adapter().from_response(WIRE_200, _package("hi"), MODEL, GOOGLE)

    assert res.text == "Blue."
    assert res.stop_reason == "STOP"
    assert res.served_model == "gemini-2.0-flash-001"
    assert res.model == MODEL
    assert res.provider == GOOGLE
    assert res.dialect == GOOGLE_GENERATE_CONTENT
    assert res.trace is None


def test_the_differently_named_usage_meters_are_normalised() -> None:
    res = _adapter().from_response(WIRE_200, _package("hi"), MODEL, GOOGLE)

    assert res.usage.input_tokens == 17
    assert res.usage.output_tokens == 5
    assert res.usage.cache_read_tokens == 8
    assert res.usage.cache_write_tokens == 0, "this wire reports no cache-write meter"


def test_missing_usage_defaults_every_meter_to_zero() -> None:
    res = _adapter().from_response({"candidates": []}, _package("hi"), MODEL, GOOGLE)

    assert res.usage.input_tokens == 0
    assert res.usage.output_tokens == 0
    assert res.items == ()
    assert res.stop_reason is None


def test_a_multi_part_answer_stays_a_transcript_rather_than_a_concatenation() -> None:
    raw = {"candidates": [{"content": {"parts": [{"text": "one "}, {"text": "two"}]}}]}
    res = _adapter().from_response(raw, _package("hi"), MODEL, GOOGLE)

    assert len(res.items) == 2
    assert res.text == "one two"


def test_a_thought_part_is_not_folded_into_the_answer() -> None:
    """Gemini returns reasoning as text flagged `thought`; it must not reach `text`."""
    raw = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "deliberating privately", "thought": True},
                        {"text": "Blue."},
                    ]
                }
            }
        ]
    }
    res = _adapter().from_response(raw, _package("hi"), MODEL, GOOGLE)

    assert res.text == "Blue."
    assert len(res.items) == 1
    assert "deliberating" in str(res.raw), "but it survives on raw"


def test_a_function_call_part_is_passed_over_until_the_ir_can_carry_it() -> None:
    raw = {
        "candidates": [
            {
                "content": {"parts": [{"functionCall": {"name": "get_weather", "args": {}}}]},
                "finishReason": "STOP",
            }
        ]
    }
    res = _adapter().from_response(raw, _package("hi"), MODEL, GOOGLE)

    assert res.items == ()
    assert res.raw["candidates"][0]["content"]["parts"][0]["functionCall"]["name"] == "get_weather"


def test_an_absent_model_version_reads_as_unknown_not_as_a_guess() -> None:
    raw = {"candidates": [{"content": {"parts": [{"text": "x"}]}}]}
    res = _adapter().from_response(raw, _package("hi"), MODEL, GOOGLE)

    assert res.served_model is None


def test_the_whole_body_survives_on_raw_but_never_reaches_a_dump() -> None:
    res = _adapter().from_response(WIRE_200, _package("hi"), MODEL, GOOGLE)

    assert res.raw == WIRE_200
    assert "raw" not in res.model_dump()


# ------------------------------------------------------------------ the errors


def test_a_bad_key_reported_as_400_is_reclassified_as_auth() -> None:
    """The refinement that exists because this wire calls a bad credential a bad request."""
    error = _adapter().error_from(400, LIVE_400_BAD_KEY, GOOGLE)

    assert error is not None
    assert error.failure_class == "auth"
    assert error.retryable is False
    assert error.status == 400
    assert error.raw == LIVE_400_BAD_KEY


def test_the_reclassified_message_does_not_echo_the_providers_text() -> None:
    error = _adapter().error_from(400, LIVE_400_BAD_KEY, GOOGLE)

    assert error is not None
    assert "API key not valid" not in str(error)
    assert "GEMINI_API_KEY" in str(error)


def test_a_401_is_already_auth_and_is_left_to_the_connector() -> None:
    assert _adapter().error_from(401, LIVE_401_WRONG_CREDENTIAL_TYPE, GOOGLE) is None


def test_a_403_is_left_alone_too() -> None:
    assert _adapter().error_from(403, LIVE_400_BAD_KEY, GOOGLE) is None


def test_a_400_about_the_request_is_left_as_a_request_error() -> None:
    """No ErrorInfo reason means no evidence, and no evidence means no reclassification."""
    assert _adapter().error_from(400, LIVE_400_BAD_REQUEST, GOOGLE) is None


@pytest.mark.parametrize("status", [404, 429, 500, 503])
def test_every_other_status_is_left_to_the_connector(status: int) -> None:
    assert _adapter().error_from(status, LIVE_400_BAD_REQUEST, GOOGLE) is None


def test_an_undecodable_body_is_left_alone() -> None:
    assert _adapter().error_from(400, None, GOOGLE) is None


@pytest.mark.parametrize(
    "raw",
    [
        {"error": "a bare string"},
        {"error": {"details": "not a list"}},
        {"error": {"details": [{"reason": "SOMETHING_ELSE"}]}},
        {"error": {"details": ["not a mapping"]}},
        {"nothing": "useful"},
    ],
)
def test_an_error_body_of_the_wrong_shape_never_reclassifies(raw: dict[str, Any]) -> None:
    assert _adapter().error_from(400, raw, GOOGLE) is None
