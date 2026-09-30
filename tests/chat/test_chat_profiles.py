"""The measured profile table, and the divergence it exists to record.

These tests pin an empirical claim, not a design preference: six Anthropic models
answer the same request three different ways. If a future measurement contradicts a
row, this file is where that shows up -- and it already has. The prefill cases below
asserted the opposite until 2026-09-16, when a re-probe found the 5 generation
rejecting a prefill it had accepted nine days earlier. The table was corrected to the
wire, not the wire argued with.
"""

import pytest

from franca.chat.profile import ChatProfile
from franca.chat.profiles import CHAT_PROFILES
from franca.core.enums import ThinkingMode
from franca.core.ids import ANTHROPIC, OPENAI, Provider

LEGACY = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5")
ADAPTIVE = ("claude-sonnet-5", "claude-opus-5", "claude-fable-5-1")


def _resolve(model: str, provider: Provider = ANTHROPIC) -> ChatProfile:
    return CHAT_PROFILES.resolve(provider, model)


@pytest.mark.parametrize("model", LEGACY)
def test_the_older_generation_takes_sampling_parameters(model: str) -> None:
    assert _resolve(model).sampling_allowed is True


@pytest.mark.parametrize("model", ADAPTIVE)
def test_the_newer_generation_rejects_sampling_parameters(model: str) -> None:
    """Measured: `temperature` and `top_p` are both a 400 on these models."""
    assert _resolve(model).sampling_allowed is False


@pytest.mark.parametrize("model", LEGACY)
def test_the_older_generation_thinks_on_a_token_budget(model: str) -> None:
    profile = _resolve(model)
    assert profile.thinking is ThinkingMode.budget
    assert profile.budget_tokens_allowed is True


@pytest.mark.parametrize("model", ADAPTIVE)
def test_the_newer_generation_thinks_adaptively_and_refuses_a_budget(model: str) -> None:
    profile = _resolve(model)
    assert profile.thinking is ThinkingMode.adaptive
    assert profile.budget_tokens_allowed is False


def test_effort_splits_the_generations_differently_from_everything_else() -> None:
    """The finding that proves this cannot be a two-way per-generation branch.

    `opus-4-5` sits with the OLD models on sampling and thinking, and with the NEW
    ones on effort. Any scheme with a single generation boundary gets it wrong.
    """
    assert _resolve("claude-haiku-4-5").effort_levels == ()
    assert _resolve("claude-sonnet-4-5").effort_levels == ()
    assert _resolve("claude-opus-4-5").effort_levels == ("high",)
    assert _resolve("claude-sonnet-5").effort_levels == ("high",)

    # ... while on sampling, opus-4-5 goes back to the other side.
    assert _resolve("claude-opus-4-5").sampling_allowed is True
    assert _resolve("claude-sonnet-5").sampling_allowed is False


@pytest.mark.parametrize(
    ("spelling", "expected_prefix"),
    [
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
        ("claude-opus-4-5-20251101", "claude-opus-4-5"),
        ("claude-opus-5", "claude-opus-5"),
        ("claude-opus-5[1m]", "claude-opus-5"),
        ("anthropic.claude-sonnet-5", "claude-sonnet-5"),
    ],
)
def test_dated_and_decorated_ids_land_on_the_same_row(spelling: str, expected_prefix: str) -> None:
    """A snapshot id, a context-window suffix and a Bedrock prefix are one model."""
    assert _resolve(spelling).model_prefix == expected_prefix


def test_longest_prefix_wins_over_the_provider_default() -> None:
    assert _resolve("claude-opus-5").model_prefix == "claude-opus-5"
    assert _resolve("claude-opus-5").verified is not None


def test_an_unmeasured_model_resolves_but_claims_nothing() -> None:
    """`None` is not `False`: unknown must never be reported as verified-absent."""
    profile = _resolve("claude-something-unreleased")
    assert profile.model_prefix == ""
    assert profile.sampling_allowed is None
    assert profile.thinking is None
    assert profile.budget_tokens_allowed is None
    assert profile.verified is None
    assert profile.is_verified is False


def test_openai_rows_record_a_measurement_gap_not_a_finding() -> None:
    """The account had no credits, so nothing about OpenAI could be verified."""
    profile = _resolve("gpt-4.1-nano", OPENAI)
    assert profile.sampling_allowed is None
    assert profile.is_verified is False
    assert "unmeasured" in profile.notes


def test_every_measured_row_carries_the_date_it_was_measured() -> None:
    for profile in CHAT_PROFILES.profiles:
        assert profile.is_verified, f"{profile.name} claims capabilities without a date"


@pytest.mark.parametrize("model", LEGACY)
def test_the_older_generation_still_takes_an_assistant_prefill(model: str) -> None:
    assert _resolve(model).prefill_allowed is True


@pytest.mark.parametrize("model", ADAPTIVE)
def test_the_newer_generation_rejects_an_assistant_prefill(model: str) -> None:
    """Measured 2026-09-16: 400, "the conversation must end with a user message".

    This assertion is inverted from what it was, and the inversion is the reason the
    file exists. Nothing in franca changed between the two sweeps; the provider did.
    A row that had been left at its first reading would now refuse nothing and send a
    request that is a guaranteed 400.
    """
    assert _resolve(model).prefill_allowed is False
