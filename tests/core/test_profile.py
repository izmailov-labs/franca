"""Executable specification for `franca.core.profile`.

Two things here are easy to get wrong and expensive to get wrong. `normalize_model_id`
decides which row a dated, prefixed or aliased model id lands on, so it is pinned as a
table and for idempotence. And the overlay distinguishes a row that *set* a field to
`None` -- "verified absent, do not inherit" -- from a row that never mentioned it,
which is the difference between disabling a feature and inheriting it.
"""

from datetime import date

import pytest

from franca.core.enums import ThinkingMode
from franca.core.errors import ProfileError
from franca.core.ids import ANTHROPIC, OPENAI, Provider
from franca.core.profile import BaseProfile, ProfileResolver, ProfileTable, normalize_model_id


class DemoProfile(BaseProfile, frozen=True):
    """A stand-in for `ChatProfile`: one tri-state flag, one number, two containers."""

    thinking: ThinkingMode | None = None
    max_output_tokens: int | None = None
    ignored_params: frozenset[str] = frozenset()
    effort_levels: tuple[str, ...] = ()


def _through_the_seam(resolver: ProfileResolver[DemoProfile], model: str) -> DemoProfile:
    """Resolve only through the protocol, so mypy proves `ProfileTable` satisfies it."""
    return resolver.resolve(ANTHROPIC, model)


def table(
    *rows: DemoProfile,
    defaults: dict[Provider, DemoProfile] | None = None,
) -> ProfileTable[DemoProfile]:
    return ProfileTable[DemoProfile](profiles=rows, defaults=defaults or {})


NORMALISED = [
    ("claude-opus-4-5-20251101", "claude-opus-4-5"),
    ("claude-opus-4-5[1m]", "claude-opus-4-5"),
    ("anthropic.claude-sonnet-4-5", "claude-sonnet-4-5"),
    ("gemini-2.5-pro@20250219", "gemini-2.5-pro"),
    ("gpt-5.2-latest", "gpt-5.2"),
    ("openai.gpt-5.2-2025-11-01", "openai.gpt-5.2-2025-11-01".removeprefix("openai.")),
    ("google.gemini-3-pro-latest", "gemini-3-pro"),
    ("anthropic.claude-opus-4-5-20251101[1m]", "claude-opus-4-5"),
    ("  claude-haiku-4-5  ", "claude-haiku-4-5"),
    ("ANTHROPIC.Claude-Opus-4-5-LATEST", "Claude-Opus-4-5"),
    ("grok-4-fast", "grok-4-fast"),
    ("", ""),
]


@pytest.mark.parametrize(("given", "expected"), NORMALISED)
def test_normalize_model_id_table(given: str, expected: str) -> None:
    assert normalize_model_id(given) == expected


@pytest.mark.parametrize(("given", "expected"), NORMALISED)
def test_normalize_model_id_is_idempotent(given: str, expected: str) -> None:
    once = normalize_model_id(given)

    assert normalize_model_id(once) == once
    assert once == expected


def test_normalize_model_id_keeps_an_embedded_date() -> None:
    # Only a *trailing* snapshot date is a decoration; one in the middle is the name.
    assert normalize_model_id("gpt-4o-20240513-preview") == "gpt-4o-20240513-preview"


def test_base_profile_verification_and_name() -> None:
    row = DemoProfile(provider=ANTHROPIC, model_prefix="claude-opus")
    checked = DemoProfile(
        provider=ANTHROPIC,
        model_prefix="claude-opus",
        source="https://docs.anthropic.com/en/docs/about-claude/models",  # type: ignore[arg-type]
        verified=date(2026, 9, 1),
    )

    assert row.is_verified is False
    assert checked.is_verified is True
    assert row.name == "anthropic:claude-opus"
    assert DemoProfile(provider=ANTHROPIC, model_prefix="").name == "anthropic:"


def test_longest_prefix_wins() -> None:
    short = DemoProfile(provider=ANTHROPIC, model_prefix="claude", max_output_tokens=4096)
    long = DemoProfile(provider=ANTHROPIC, model_prefix="claude-opus", max_output_tokens=64000)

    resolved = table(short, long).resolve(ANTHROPIC, "claude-opus-4-5-20251101")

    assert resolved.model_prefix == "claude-opus"
    assert resolved.max_output_tokens == 64000


def test_shorter_prefix_still_wins_when_the_longer_one_does_not_match() -> None:
    short = DemoProfile(provider=ANTHROPIC, model_prefix="claude", max_output_tokens=4096)
    long = DemoProfile(provider=ANTHROPIC, model_prefix="claude-opus", max_output_tokens=64000)

    resolved = table(short, long).resolve(ANTHROPIC, "claude-haiku-4-5")

    assert resolved.model_prefix == "claude"


def test_empty_prefix_matches_everything_as_a_fallback_row() -> None:
    catch_all = DemoProfile(provider=ANTHROPIC, model_prefix="", thinking=ThinkingMode.budget)

    resolved = table(catch_all).resolve(ANTHROPIC, "a-model-nobody-has-heard-of")

    assert resolved.model_prefix == ""
    assert resolved.thinking is ThinkingMode.budget


def test_a_row_for_another_provider_never_matches() -> None:
    foreign = DemoProfile(provider=OPENAI, model_prefix="", max_output_tokens=1)
    base = DemoProfile(provider=ANTHROPIC, model_prefix="", max_output_tokens=4096)

    resolved = table(foreign, defaults={ANTHROPIC: base}).resolve(ANTHROPIC, "claude-opus-4-5")

    assert resolved.max_output_tokens == 4096


def test_a_row_with_no_default_is_returned_as_is() -> None:
    row = DemoProfile(provider=ANTHROPIC, model_prefix="claude", thinking=ThinkingMode.budget)

    resolved = table(row).resolve(ANTHROPIC, "claude-opus-4-5")

    assert resolved is row


def test_the_default_is_returned_when_no_row_matches() -> None:
    base = DemoProfile(provider=ANTHROPIC, model_prefix="", thinking=ThinkingMode.effort)
    row = DemoProfile(provider=ANTHROPIC, model_prefix="claude-opus")

    resolved = table(row, defaults={ANTHROPIC: base}).resolve(ANTHROPIC, "gpt-5.2")

    assert resolved is base


def test_overlay_keeps_an_explicit_none_and_inherits_an_unset_field() -> None:
    base = DemoProfile(
        provider=ANTHROPIC,
        model_prefix="",
        thinking=ThinkingMode.budget,
        max_output_tokens=4096,
    )
    # `thinking=None` is *set*: this model verifiably has no thinking mode.
    # `max_output_tokens` is never mentioned, so it must come from the default.
    row = DemoProfile(provider=ANTHROPIC, model_prefix="claude-haiku", thinking=None)

    resolved = table(row, defaults={ANTHROPIC: base}).resolve(ANTHROPIC, "claude-haiku-4-5")

    assert resolved.thinking is None
    assert resolved.max_output_tokens == 4096
    assert resolved.model_prefix == "claude-haiku"


def test_overlay_lets_a_set_value_win_over_the_default() -> None:
    base = DemoProfile(provider=ANTHROPIC, model_prefix="", thinking=ThinkingMode.budget)
    row = DemoProfile(
        provider=ANTHROPIC, model_prefix="claude-opus", thinking=ThinkingMode.adaptive
    )

    resolved = table(row, defaults={ANTHROPIC: base}).resolve(ANTHROPIC, "claude-opus-4-5")

    assert resolved.thinking is ThinkingMode.adaptive


def test_overlay_survives_container_fields() -> None:
    base = DemoProfile(
        provider=ANTHROPIC,
        model_prefix="",
        ignored_params=frozenset({"top_k"}),
        effort_levels=("low", "high"),
    )
    row = DemoProfile(
        provider=ANTHROPIC,
        model_prefix="claude-opus",
        ignored_params=frozenset({"temperature", "top_p"}),
    )

    resolved = table(row, defaults={ANTHROPIC: base}).resolve(ANTHROPIC, "claude-opus-4-5[1m]")

    # `model_copy(update=...)` does not validate, so the python-mode dump has to have
    # handed back a real frozenset rather than a set.
    assert isinstance(resolved.ignored_params, frozenset)
    assert resolved.ignored_params == frozenset({"temperature", "top_p"})
    assert isinstance(resolved.effort_levels, tuple)
    assert resolved.effort_levels == ("low", "high")


def test_the_overlay_does_not_mutate_the_row_or_the_default() -> None:
    base = DemoProfile(provider=ANTHROPIC, model_prefix="", max_output_tokens=4096)
    row = DemoProfile(provider=ANTHROPIC, model_prefix="claude", thinking=ThinkingMode.budget)

    resolved = table(row, defaults={ANTHROPIC: base}).resolve(ANTHROPIC, "claude-opus-4-5")

    assert resolved is not row
    assert resolved is not base
    assert row.max_output_tokens is None
    assert base.thinking is None


def test_a_dated_model_id_resolves_to_the_undated_row() -> None:
    row = DemoProfile(provider=ANTHROPIC, model_prefix="claude-opus-4-5", max_output_tokens=64000)
    rows = table(row)

    for spelling in (
        "claude-opus-4-5",
        "claude-opus-4-5-20251101",
        "claude-opus-4-5[1m]",
        "anthropic.claude-opus-4-5",
    ):
        assert rows.resolve(ANTHROPIC, spelling).max_output_tokens == 64000


def test_no_row_and_no_default_raises_profile_error() -> None:
    empty = table()

    with pytest.raises(ProfileError) as excinfo:
        empty.resolve(ANTHROPIC, "claude-opus-4-5")

    message = str(excinfo.value)
    assert "anthropic" in message
    assert "claude-opus-4-5" in message
    assert "Registrar.profiles" in message


def test_profile_table_satisfies_the_resolver_protocol() -> None:
    row = DemoProfile(provider=ANTHROPIC, model_prefix="claude", max_output_tokens=8192)

    # The protocol is not `runtime_checkable` -- satisfaction is a static claim, and
    # passing the table through `_through_the_seam` is what makes mypy check it.
    resolved = _through_the_seam(table(row), "claude-opus-4-5-20251101")

    assert resolved.max_output_tokens == 8192


def test_profile_table_defaults() -> None:
    empty = ProfileTable[DemoProfile]()

    assert empty.verified_on is None
    assert empty.profiles == ()
    assert empty.defaults == {}
