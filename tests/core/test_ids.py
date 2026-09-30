"""Identifier vocabulary, and the security property its error message must hold."""

import pytest
from pydantic import BaseModel, ValidationError

from franca.core.errors import format_validation
from franca.core.ids import (
    ANTHROPIC,
    ANTHROPIC_MESSAGES,
    CHAT,
    Capability,
    Dialect,
    Provider,
    ProviderField,
    validate_slug,
)


class _Row(BaseModel, frozen=True):
    """A model with a validated provider field, standing in for Endpoint."""

    provider: ProviderField


@pytest.mark.parametrize(
    "value",
    ["anthropic", "a", "my_shim", "openai2", "a" * 64, "anthropic_messages"],
)
def test_accepts_env_var_spellable_slugs(value: str) -> None:
    assert validate_slug(value) == value


@pytest.mark.parametrize(
    "value",
    ["", "My_Shim", "my-shim", "my.shim", "9x", "_x", "a" * 65, "anthropic\n", "a b"],
)
def test_rejects_anything_that_is_not_an_env_var_segment(value: str) -> None:
    with pytest.raises(ValueError, match="not a slug"):
        validate_slug(value)


def test_the_rejected_value_never_reaches_the_error_message() -> None:
    """A mistyped key pasted into a provider slug must not survive into a message.

    `format_validation` promises a path and a message and never the input, because
    the input may hold a secret. A validator that echoes its argument silently
    breaks that promise for every caller downstream.
    """
    canary = "sk-ant-CANARY-40f1e9"
    with pytest.raises(ValidationError) as exc:
        _Row(provider=Provider(canary))  # a caller that skipped the slug rule

    rendered = format_validation(exc.value)
    assert canary not in rendered
    assert "provider" in rendered  # the field path still identifies what went wrong


def test_the_rejected_value_never_reaches_the_raw_validation_error_either() -> None:
    canary = "sk-ant-CANARY-40f1e9"
    with pytest.raises(ValueError, match="not a slug") as exc:
        validate_slug(canary)
    assert canary not in str(exc.value)


def test_newtypes_are_transparent_strings_at_runtime() -> None:
    """`NewType` costs nothing: the constant is the string, usable as a dict key."""
    assert ANTHROPIC == "anthropic"
    assert isinstance(ANTHROPIC, str)
    assert type(ANTHROPIC) is str
    assert {ANTHROPIC: 1}[Provider("anthropic")] == 1
    # A NewType is erased at runtime, so a plain str key hits the same entry. mypy
    # rejects that spelling directly, which is the whole point -- go through a
    # str-keyed alias to show the runtime behaviour without asking for a suppression.
    erased: dict[str, int] = {str(ANTHROPIC): 1}
    assert erased["anthropic"] == 1
    assert f"FRANCA_{ANTHROPIC.upper()}_API_KEY" == "FRANCA_ANTHROPIC_API_KEY"


def test_well_known_constants_satisfy_their_own_validator() -> None:
    """Every shipped identifier must pass the rule the library enforces on plugins."""
    for value in (ANTHROPIC, CHAT, ANTHROPIC_MESSAGES):
        assert validate_slug(value) == value


def test_constructors_are_identity_functions() -> None:
    assert Provider("x") == "x"
    assert Dialect("x") == "x"
    assert Capability("x") == "x"
