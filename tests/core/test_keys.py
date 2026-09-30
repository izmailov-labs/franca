"""Tests for `franca.core.keys`: the key seam, the static provider and the vendor table."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import SecretStr

from franca.core import keys
from franca.core.ids import ANTHROPIC, DEEPSEEK, GOOGLE, OPENAI, XAI, Provider
from franca.core.keys import VENDOR_ENV, KeyProvider, StaticKeyProvider

if TYPE_CHECKING:
    from collections.abc import MutableMapping

PLAINTEXT = "sk-ant-plaintext-must-never-leak"
MASK = "**********"


def _provider_with_anthropic_key() -> StaticKeyProvider:
    return StaticKeyProvider({ANTHROPIC: SecretStr(PLAINTEXT)})


# --------------------------------------------------------------------------------------
# StaticKeyProvider.key_for
# --------------------------------------------------------------------------------------


def test_key_for_known_provider_returns_the_configured_secret() -> None:
    secret = SecretStr(PLAINTEXT)
    provider = StaticKeyProvider({ANTHROPIC: secret})

    assert provider.key_for(ANTHROPIC) == secret


def test_key_for_unknown_provider_returns_none() -> None:
    provider = _provider_with_anthropic_key()

    assert provider.key_for(OPENAI) is None
    assert provider.key_for(Provider("some_plugin")) is None


def test_key_for_on_an_empty_provider_returns_none() -> None:
    assert StaticKeyProvider({}).key_for(ANTHROPIC) is None


def test_key_for_hands_out_a_secret_str_not_plaintext() -> None:
    key = _provider_with_anthropic_key().key_for(ANTHROPIC)

    assert isinstance(key, SecretStr)
    assert str(key) == MASK
    assert PLAINTEXT not in repr(key)


def test_key_for_matches_by_slug_value_not_identity() -> None:
    provider = _provider_with_anthropic_key()

    assert provider.key_for(Provider("anthropic")) is not None


# --------------------------------------------------------------------------------------
# StaticKeyProvider never leaks or aliases
# --------------------------------------------------------------------------------------


def test_repr_and_str_never_contain_the_plaintext_key() -> None:
    provider = _provider_with_anthropic_key()

    assert PLAINTEXT not in repr(provider)
    assert PLAINTEXT not in str(provider)


def test_repr_names_the_configured_providers() -> None:
    provider = StaticKeyProvider({OPENAI: SecretStr("a"), ANTHROPIC: SecretStr("b")})

    assert repr(provider) == "StaticKeyProvider(providers=[anthropic, openai])"


def test_constructor_copies_the_mapping_instead_of_aliasing_it() -> None:
    mapping: dict[Provider, SecretStr] = {ANTHROPIC: SecretStr(PLAINTEXT)}
    provider = StaticKeyProvider(mapping)

    mapping[OPENAI] = SecretStr("added-later")
    mapping[ANTHROPIC] = SecretStr("replaced-later")

    assert provider.key_for(OPENAI) is None
    assert provider.key_for(ANTHROPIC) == SecretStr(PLAINTEXT)


def test_deleting_from_the_callers_mapping_does_not_remove_the_key() -> None:
    mapping: dict[Provider, SecretStr] = {ANTHROPIC: SecretStr(PLAINTEXT)}
    provider = StaticKeyProvider(mapping)

    del mapping[ANTHROPIC]

    assert provider.key_for(ANTHROPIC) == SecretStr(PLAINTEXT)


def test_static_key_provider_satisfies_the_key_provider_protocol() -> None:
    seam: KeyProvider = _provider_with_anthropic_key()

    assert seam.key_for(ANTHROPIC) == SecretStr(PLAINTEXT)


# --------------------------------------------------------------------------------------
# VENDOR_ENV
# --------------------------------------------------------------------------------------


def test_vendor_env_has_exactly_the_five_first_party_entries() -> None:
    assert dict(VENDOR_ENV) == {
        ANTHROPIC: "ANTHROPIC_API_KEY",
        OPENAI: "OPENAI_API_KEY",
        GOOGLE: "GEMINI_API_KEY",
        XAI: "XAI_API_KEY",
        DEEPSEEK: "DEEPSEEK_API_KEY",
    }
    assert len(VENDOR_ENV) == 5


@pytest.mark.parametrize(
    ("provider", "env_var"),
    [
        (ANTHROPIC, "ANTHROPIC_API_KEY"),
        (OPENAI, "OPENAI_API_KEY"),
        (GOOGLE, "GEMINI_API_KEY"),
        (XAI, "XAI_API_KEY"),
        (DEEPSEEK, "DEEPSEEK_API_KEY"),
    ],
)
def test_vendor_env_maps_each_provider_to_its_variable(provider: Provider, env_var: str) -> None:
    assert VENDOR_ENV[provider] == env_var


def test_vendor_env_rejects_item_assignment() -> None:
    mutable = cast("MutableMapping[Provider, str]", VENDOR_ENV)

    with pytest.raises(TypeError):
        mutable[ANTHROPIC] = "SOMETHING_ELSE"


def test_vendor_env_rejects_item_deletion() -> None:
    mutable = cast("MutableMapping[Provider, str]", VENDOR_ENV)

    with pytest.raises(TypeError):
        del mutable[ANTHROPIC]


def test_vendor_env_rejects_new_entries() -> None:
    mutable = cast("MutableMapping[Provider, str]", VENDOR_ENV)

    with pytest.raises(TypeError):
        mutable[Provider("mistral")] = "MISTRAL_API_KEY"


# --------------------------------------------------------------------------------------
# The secrets rule (plan section 9.1): this module hands SecretStr out, never unwraps it
# --------------------------------------------------------------------------------------


def test_keys_module_source_never_unwraps_a_secret() -> None:
    assert keys.__file__ is not None
    source = Path(keys.__file__).read_text(encoding="utf-8")

    assert "get_secret_value" not in source
