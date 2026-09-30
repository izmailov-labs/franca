"""Executable specification for `franca.chat.profile`.

`ChatProfile` carries no behaviour, so what is worth pinning is its shape. Three
properties are load-bearing and each is easy to break by accident.

Every fact starts tri-state at `None`. A default of `False` anywhere would turn
"nobody checked" into "verified absent", which silences the very warning the
unverified set exists to raise.

A partially-specified row must dump *only* what it was given. `ProfileTable` overlays
a row on a provider default with `model_dump(exclude_unset=True)`, so a stray default
in that dump would overwrite the default it was supposed to inherit -- and an
explicitly written `None` must survive as a set field, because that is how a row says
"verified absent, do not inherit".

And a typo must raise. Without `extra="forbid"`, `sampling_allwed=True` in a
hand-written table would be accepted, ignored, and read as unverified forever.
"""

from datetime import date

import pytest
from pydantic import ValidationError

from franca.chat.profile import ChatProfile
from franca.core.enums import Strategy, ThinkingMode
from franca.core.ids import ANTHROPIC, OPENAI, OPENAI_RESPONSES
from franca.core.profile import BaseProfile, ProfileTable

IDENTITY_FIELDS = frozenset({"provider", "model_prefix"})


def bare() -> ChatProfile:
    """The provider-wide fallback row the M0 acceptance test builds: claims nothing."""
    return ChatProfile(provider=ANTHROPIC, model_prefix="")


# --------------------------------------------------------------------------------------
# Defaults: everything starts unverified
# --------------------------------------------------------------------------------------


def test_a_bare_row_is_valid() -> None:
    row = bare()

    assert isinstance(row, BaseProfile)
    assert row.provider == ANTHROPIC
    assert row.model_prefix == ""


def test_every_non_identity_field_defaults_to_unverified_or_empty() -> None:
    row = bare()

    for name in ChatProfile.model_fields:
        if name in IDENTITY_FIELDS:
            continue
        value = getattr(row, name)
        assert value is None or not value, f"{name} defaults to {value!r}"


def test_a_flag_is_none_and_not_false() -> None:
    row = bare()

    # The whole point of the tri-state: unverified is not "verified absent".
    assert row.sampling_allowed is None
    assert row.sampling_allowed is not False

    assert row.thinking is None
    assert row.structured_output is None
    assert row.tools_require_dialect is None
    assert row.retired is None
    assert row.token_factor is None
    assert row.default_max_output_tokens is None
    assert row.tool_calling is None
    assert row.parallel_tool_calls is None
    assert row.streaming is None
    assert row.prefill_allowed is None
    assert row.budget_tokens_allowed is None
    assert row.strict_enforced is None
    assert row.image_input is None
    assert row.image_output is None
    assert row.history_append_only is None
    assert row.mid_conversation_effort is None


def test_the_lint_path_facts_are_present_and_unset() -> None:
    row = bare()

    assert row.verification is None
    assert row.format_bias is None
    assert row.verbosity_bias is None
    assert row.emphasis_sensitive is None


def test_container_fields_start_empty() -> None:
    row = bare()

    assert row.effort_levels == ()
    assert row.ignored_params == frozenset()


def test_identity_and_verification_come_from_base_profile() -> None:
    row = bare()

    assert row.name == "anthropic:"
    assert row.is_verified is False

    checked = ChatProfile(
        provider=ANTHROPIC,
        model_prefix="claude-opus",
        verified=date(2026, 1, 9),
        notes="checked against the live API",
    )

    assert checked.name == "anthropic:claude-opus"
    assert checked.is_verified is True


# --------------------------------------------------------------------------------------
# Field vocabularies
# --------------------------------------------------------------------------------------


def test_the_typed_fields_accept_their_vocabularies() -> None:
    row = ChatProfile(
        provider=OPENAI,
        model_prefix="gpt-5",
        thinking=ThinkingMode.effort,
        effort_levels=("low", "medium", "high"),
        default_effort="medium",
        structured_output=Strategy.native,
        tools_require_dialect=OPENAI_RESPONSES,
        default_max_output_tokens=32_000,
        ignored_params=frozenset({"top_k"}),
        token_factor=1.15,
    )

    assert row.thinking is ThinkingMode.effort
    assert row.structured_output is Strategy.native
    assert row.tools_require_dialect == OPENAI_RESPONSES
    assert row.effort_levels == ("low", "medium", "high")
    assert row.ignored_params == frozenset({"top_k"})
    assert row.token_factor == pytest.approx(1.15)


# --------------------------------------------------------------------------------------
# The table contract: forbid, frozen, round-trip
# --------------------------------------------------------------------------------------


def test_a_misspelled_flag_is_rejected_and_named() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ChatProfile.model_validate(
            {"provider": "anthropic", "model_prefix": "", "sampling_allwed": True},
        )

    assert any("sampling_allwed" in error["loc"] for error in exc_info.value.errors())


@pytest.mark.parametrize("field", ["sampling_allowed", "thinking", "ignored_params", "notes"])
def test_a_row_is_frozen(field: str) -> None:
    row = bare()

    # setattr, not `row.x = ...`: the fields are read-only to the type checker too.
    with pytest.raises(ValidationError):
        setattr(row, field, None)


def test_container_fields_survive_a_round_trip() -> None:
    row = ChatProfile(
        provider=ANTHROPIC,
        model_prefix="claude-opus",
        ignored_params=frozenset({"top_k", "seed"}),
        effort_levels=("low", "high"),
    )

    # Python mode keeps the containers; JSON mode flattens them to a list, and the
    # overlay only works if validating that back restores the exact same types.
    for dumped in (row.model_dump(), row.model_dump(mode="json")):
        back = ChatProfile.model_validate(dumped)

        assert back == row
        assert isinstance(back.ignored_params, frozenset)
        assert back.ignored_params == frozenset({"top_k", "seed"})
        assert isinstance(back.effort_levels, tuple)
        assert back.effort_levels == ("low", "high")


def test_exclude_unset_dumps_only_what_was_passed() -> None:
    row = ChatProfile(
        provider=ANTHROPIC,
        model_prefix="claude-opus",
        thinking=ThinkingMode.budget,
        prefill_allowed=True,
    )

    # Load-bearing: anything extra here would overwrite a provider default that the
    # row never meant to touch.
    assert row.model_dump(exclude_unset=True) == {
        "provider": ANTHROPIC,
        "model_prefix": "claude-opus",
        "thinking": ThinkingMode.budget,
        "prefill_allowed": True,
    }


def test_an_explicit_none_counts_as_set() -> None:
    row = ChatProfile(provider=ANTHROPIC, model_prefix="claude-opus", thinking=None)
    dumped = row.model_dump(exclude_unset=True)

    assert dumped["thinking"] is None
    assert "prefill_allowed" not in dumped


def test_the_overlay_keeps_an_explicit_none_and_inherits_the_rest() -> None:
    default = ChatProfile(
        provider=ANTHROPIC,
        model_prefix="",
        thinking=ThinkingMode.budget,
        sampling_allowed=True,
        streaming=True,
        ignored_params=frozenset({"top_k"}),
    )
    row = ChatProfile(
        provider=ANTHROPIC,
        model_prefix="claude-opus",
        thinking=None,
        prefill_allowed=False,
    )
    table = ProfileTable[ChatProfile](profiles=(row,), defaults={ANTHROPIC: default})

    resolved = table.resolve(ANTHROPIC, "anthropic.claude-opus-4-5-20251101")

    assert resolved.thinking is None  # written as None: verified absent, not inherited
    assert resolved.sampling_allowed is True  # never mentioned: inherited
    assert resolved.streaming is True
    assert resolved.prefill_allowed is False
    assert resolved.model_prefix == "claude-opus"
    assert resolved.ignored_params == frozenset({"top_k"})
