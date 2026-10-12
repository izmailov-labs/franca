"""Executable specification for `franca.chat.dialects.common`.

These are the decisions every chat wire must make identically, so they are pinned once
here and each adapter's own tests only check where it puts the result. Three of them
carry a cost when they drift.

`pick_strategy` must never substitute a strategy the caller named: a caller who asked
for `tool_calling` and silently got JSON mode would be measuring the wrong thing. It
must record what it assumed, because an unmeasured profile is the common case and the
trace is the only place that says so. And `merge_provider_options` must merge deeply
without touching either side, since the options belong to a frozen package that a retry
will render again.
"""

from typing import Any

import pytest

from franca.chat.dialects.common import (
    DEFAULT_SCHEMA_NAME,
    STRATEGY_ORDER,
    decode_object,
    merge_provider_options,
    pick_strategy,
    read_structured,
    schema_name,
)
from franca.chat.ir import OutputFormat
from franca.chat.profile import ChatProfile
from franca.core.enums import Strategy
from franca.core.errors import ModelError
from franca.core.ids import ANTHROPIC, ANTHROPIC_MESSAGES
from franca.core.profile import BaseProfile

SCHEMA: dict[str, Any] = {
    "title": "Colour",
    "type": "object",
    "properties": {"colour": {"type": "string"}},
    "required": ["colour"],
}

EVERYTHING = frozenset(Strategy)
NATIVE_AND_TOOLS = frozenset({Strategy.native, Strategy.tool_calling})

UNMEASURED = ChatProfile(provider=ANTHROPIC, model_prefix="")
"""A row that claims nothing: how this model does structured output is unknown."""


def measured(**flags: Any) -> ChatProfile:
    return ChatProfile(provider=ANTHROPIC, model_prefix="", **flags)


def pick(
    fmt: OutputFormat,
    *,
    profile: BaseProfile = UNMEASURED,
    supports: frozenset[Strategy] = EVERYTHING,
) -> tuple[Strategy | None, set[str]]:
    unverified: set[str] = set()
    chosen = pick_strategy(fmt, profile, supports, ANTHROPIC_MESSAGES, unverified)
    return chosen, unverified


def json_format(**kwargs: Any) -> OutputFormat:
    return OutputFormat(kind="json", json_schema=SCHEMA, **kwargs)


# --------------------------------------------------------------------- pick_strategy


@pytest.mark.parametrize("fmt", [OutputFormat(), OutputFormat(kind="text")])
def test_no_json_answer_means_no_strategy_and_no_assumption(fmt: OutputFormat) -> None:
    assert pick(fmt) == (None, set())


def test_the_fallback_order_is_strongest_guarantee_first() -> None:
    assert (Strategy.native, Strategy.tool_calling, Strategy.json_mode) == STRATEGY_ORDER


def test_an_unmeasured_profile_gets_native_and_records_the_assumption() -> None:
    assert pick(json_format()) == (Strategy.native, {"structured_output"})


def test_the_profile_preference_wins_and_is_not_an_assumption() -> None:
    profile = measured(structured_output=Strategy.tool_calling)

    assert pick(json_format(), profile=profile) == (Strategy.tool_calling, set())


def test_a_profile_preference_the_wire_lacks_falls_back_in_order() -> None:
    """The profile is a hint about the model; the wire still decides what exists."""
    profile = measured(structured_output=Strategy.json_mode)

    assert pick(json_format(), profile=profile, supports=NATIVE_AND_TOOLS) == (
        Strategy.native,
        set(),
    )


def test_an_explicit_strategy_is_honoured_over_the_profile() -> None:
    fmt = json_format(strategy=Strategy.json_mode)
    profile = measured(structured_output=Strategy.native)

    assert pick(fmt, profile=profile) == (Strategy.json_mode, set())


def test_an_explicit_strategy_on_an_unmeasured_profile_is_still_an_assumption() -> None:
    assert pick(json_format(strategy=Strategy.tool_calling)) == (
        Strategy.tool_calling,
        {"structured_output"},
    )


def test_an_explicit_strategy_the_wire_lacks_raises_rather_than_being_substituted() -> None:
    fmt = json_format(strategy=Strategy.json_mode)

    with pytest.raises(ModelError) as caught:
        pick(fmt, supports=NATIVE_AND_TOOLS)

    error = caught.value
    assert error.failure_class == "unsupported"
    assert error.retryable is False
    assert error.status is None
    assert error.provider == ANTHROPIC
    assert "output_format.strategy=json_mode" in str(error), "the message names the field"
    assert "native, tool_calling" in str(error), "and what the wire offers instead"


def test_schema_free_json_takes_json_mode() -> None:
    assert pick(OutputFormat(kind="json")) == (Strategy.json_mode, {"structured_output"})


def test_schema_free_json_on_a_wire_without_json_mode_raises() -> None:
    with pytest.raises(ModelError, match="only json_mode applies") as caught:
        pick(OutputFormat(kind="json"), supports=NATIVE_AND_TOOLS)

    assert caught.value.failure_class == "unsupported"


def test_strict_rules_out_json_mode_when_the_wire_offers_nothing_else() -> None:
    with pytest.raises(ModelError, match="rules out json_mode"):
        pick(json_format(strict=True), supports=frozenset({Strategy.json_mode}))


def test_strict_never_falls_back_to_json_mode() -> None:
    profile = measured(structured_output=Strategy.json_mode, strict_enforced=True)

    assert pick(json_format(strict=True), profile=profile) == (Strategy.native, set())


def test_a_wire_with_no_strategy_at_all_says_so() -> None:
    with pytest.raises(ModelError, match="supports nothing"):
        pick(json_format(), supports=frozenset())


def test_strict_on_an_unmeasured_profile_records_strict_enforced() -> None:
    assert pick(json_format(strict=True)) == (
        Strategy.native,
        {"structured_output", "strict_enforced"},
    )


def test_strict_on_a_profile_that_enforces_it_records_nothing_more() -> None:
    profile = measured(strict_enforced=True)

    assert pick(json_format(strict=True), profile=profile) == (
        Strategy.native,
        {"structured_output"},
    )


def test_strict_on_a_profile_measured_not_to_enforce_it_raises() -> None:
    """A guarantee the caller demanded and the model is known not to give."""
    profile = measured(strict_enforced=False)

    with pytest.raises(ModelError, match="treats a strict schema as a hint") as caught:
        pick(json_format(strict=True), profile=profile)

    assert caught.value.failure_class == "unsupported"


def test_strict_false_is_not_a_demand_and_reads_no_flag() -> None:
    profile = measured(strict_enforced=False, structured_output=Strategy.native)

    assert pick(json_format(strict=False), profile=profile) == (Strategy.native, set())


def test_a_non_chat_profile_claims_nothing() -> None:
    plain = BaseProfile(provider=ANTHROPIC, model_prefix="")

    assert pick(json_format(strict=True), profile=plain) == (
        Strategy.native,
        {"structured_output", "strict_enforced"},
    )


# ----------------------------------------------------------------------- schema_name


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Colour", "Colour"),
        ("_private-name_2", "_private-name_2"),
        ("x" * 64, "x" * 64),
        ("x" * 65, DEFAULT_SCHEMA_NAME),
        ("Two words", DEFAULT_SCHEMA_NAME),
        ("2fast", DEFAULT_SCHEMA_NAME),
        ("dotted.name", DEFAULT_SCHEMA_NAME),
        ("", DEFAULT_SCHEMA_NAME),
        (7, DEFAULT_SCHEMA_NAME),
    ],
)
def test_the_title_names_the_schema_only_when_every_wire_accepts_it(
    title: object, expected: str
) -> None:
    fmt = OutputFormat(kind="json", json_schema={"type": "object", "title": title})

    assert schema_name(fmt) == expected


def test_a_schema_without_a_title_or_no_schema_at_all_gets_the_default() -> None:
    assert schema_name(OutputFormat(kind="json", json_schema={"type": "object"})) == "response"
    assert schema_name(OutputFormat(kind="json")) == "response"


# --------------------------------------------------------------------- decode_object


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"colour": "blue"}', {"colour": "blue"}),
        ("  {}\n", {}),
        ('{"colour": "bl', None),
        ("[1, 2]", None),
        ('"blue"', None),
        ("I can't help with that.", None),
        ('```json\n{"colour": "blue"}\n```', None),
        ("", None),
        (None, None),
    ],
)
def test_only_a_json_object_decodes(text: str | None, expected: dict[str, Any] | None) -> None:
    """Truncation, prose and fences all come back as `None`, never as an exception."""
    assert decode_object(text) == expected


# ------------------------------------------------------------------- read_structured


def test_read_structured_is_none_when_no_json_answer_was_asked_for() -> None:
    assert read_structured(OutputFormat(), '{"colour": "blue"}', [("Colour", {})]) is None


def test_read_structured_decodes_the_text_when_there_is_no_forced_call() -> None:
    assert read_structured(json_format(), '{"colour": "blue"}', []) == {"colour": "blue"}


@pytest.mark.parametrize(
    "arguments",
    [{"colour": "blue"}, '{"colour": "blue"}'],
    ids=["an object", "a json string"],
)
def test_read_structured_prefers_the_forced_tools_arguments(arguments: object) -> None:
    calls = [("something_else", {"colour": "red"}), ("Colour", arguments)]

    assert read_structured(json_format(), "ignored prose", calls) == {"colour": "blue"}


@pytest.mark.parametrize("arguments", [None, ["blue"], "not json"])
def test_unusable_forced_arguments_are_none_rather_than_the_text(arguments: object) -> None:
    """The forced call *is* the answer; falling back to prose would mix two replies."""
    assert read_structured(json_format(), '{"colour": "blue"}', [("Colour", arguments)]) is None


# ------------------------------------------------------------ merge_provider_options


def test_options_merge_into_a_nested_object_instead_of_replacing_it() -> None:
    body = {"generationConfig": {"maxOutputTokens": 64, "responseMimeType": "application/json"}}

    merged = merge_provider_options(body, {"generationConfig": {"temperature": 0.2}})

    assert merged == {
        "generationConfig": {
            "maxOutputTokens": 64,
            "responseMimeType": "application/json",
            "temperature": 0.2,
        }
    }


def test_the_caller_wins_on_every_leaf_and_on_every_type_mismatch() -> None:
    body = {"max_tokens": 64, "a": {"b": 1}, "c": {"d": 1}, "e": [1, 2]}

    merged = merge_provider_options(body, {"max_tokens": 9, "a": {"b": 2}, "c": 5, "e": [3]})

    assert merged == {"max_tokens": 9, "a": {"b": 2}, "c": 5, "e": [3]}


def test_merging_modifies_neither_side() -> None:
    body: dict[str, Any] = {"output_config": {"format": {"type": "json_schema"}}}
    options: dict[str, Any] = {"output_config": {"effort": "high"}}

    merged = merge_provider_options(body, options)

    assert merged["output_config"] == {"format": {"type": "json_schema"}, "effort": "high"}
    assert body == {"output_config": {"format": {"type": "json_schema"}}}
    assert options == {"output_config": {"effort": "high"}}


def test_merging_nothing_is_a_copy() -> None:
    body = {"model": "m"}
    merged = merge_provider_options(body, {})

    assert merged == body
    assert merged is not body
