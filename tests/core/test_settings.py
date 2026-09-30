"""Executable specification for `franca.core.settings`."""

from pathlib import Path

import pytest
from pydantic import SecretStr

from franca.core.enums import UnverifiedPolicy
from franca.core.errors import ConfigError
from franca.core.ids import ANTHROPIC, OPENAI, Provider
from franca.core.settings import (
    ENV_ALIASES,
    ProviderSettings,
    RetryPolicy,
    RoutePolicy,
    Settings,
    SettingsKeyProvider,
    load_settings,
)

MISTRAL = Provider("mistral")

TOML = """
[providers.anthropic]
timeout_s = 90
[providers.mistral]
base_url = "https://api.mistral.ai"
api_key_env = "MISTRAL_KEY"
dialect = "openai_chat"
[providers.mistral.options]
rpm = 120
[routes.default]
layers = ["trace", "tools", "retry"]
[routes."openai:*"]
layers = ["trace", "tools", "budget", "retry"]
fallback = ["anthropic:claude-sonnet-5"]
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "franca.toml").write_text(TOML, encoding="utf-8")
    return tmp_path


# ------------------------------------------------------------------ defaults


def test_defaults_need_no_configuration_at_all(tmp_path: Path) -> None:
    settings = load_settings(env={}, cwd=tmp_path)
    assert settings.providers == {}
    assert settings.routes["default"] == RoutePolicy()
    assert settings.plugins is True
    assert settings.allow_stubs is False


def test_a_provider_keeps_its_own_defaults() -> None:
    provider = ProviderSettings()
    assert provider.timeout_s == 60.0
    assert provider.on_unverified is UnverifiedPolicy.warn
    assert provider.api_key is None


def test_retry_defaults_are_declared_not_implied() -> None:
    assert RetryPolicy() == RetryPolicy(max_attempts=3, base_s=0.5, cap_s=8.0, jitter_s=0.25)


# --------------------------------------------------------------------- files


def test_a_toml_file_beside_the_working_directory_is_found(project: Path) -> None:
    settings = load_settings(env={}, cwd=project)
    assert settings.providers[ANTHROPIC].timeout_s == 90.0
    assert settings.providers[MISTRAL].dialect == "openai_chat"
    assert settings.providers[MISTRAL].options == {"rpm": 120}


def test_an_explicit_file_wins_and_must_exist(project: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.toml"
    other.write_text("allow_stubs = true\n", encoding="utf-8")
    assert load_settings(env={}, file=other, cwd=project).allow_stubs is True

    with pytest.raises(ConfigError):
        load_settings(env={}, file=tmp_path / "absent.toml", cwd=project)


def test_the_config_env_var_points_at_a_file(project: Path, tmp_path: Path) -> None:
    other = tmp_path / "elsewhere.toml"
    other.write_text("job_timeout_s = 7\n", encoding="utf-8")
    settings = load_settings(env={"FRANCA_CONFIG": str(other)}, cwd=project)
    assert settings.job_timeout_s == 7.0


def test_a_pyproject_table_is_the_lowest_file_layer(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.franca]\npoll_interval_s = 11\n", encoding="utf-8"
    )
    assert load_settings(env={}, cwd=tmp_path).poll_interval_s == 11.0


# --------------------------------------------------------------- environment


def test_the_short_vendor_alias_works(project: Path) -> None:
    settings = load_settings(env={"FRANCA_ANTHROPIC_TIMEOUT_S": "5"}, cwd=project)
    assert settings.providers[ANTHROPIC].timeout_s == 5.0


def test_the_alias_table_is_finite_and_inspectable() -> None:
    """An explicit table rather than a regex whose tie-break has to be reasoned about."""
    assert ENV_ALIASES["FRANCA_ANTHROPIC_TIMEOUT_S"] == "providers.anthropic.timeout_s"
    assert ENV_ALIASES[f"FRANCA_{OPENAI.upper()}_BASE_URL"] == "providers.openai.base_url"
    assert all(name.startswith("FRANCA_") for name in ENV_ALIASES)


def test_a_provider_outside_the_table_uses_the_nested_form(project: Path) -> None:
    """`__` separates levels and a single `_` never does, so `my_shim` is unambiguous."""
    settings = load_settings(env={"FRANCA_PROVIDERS__MY_SHIM__BASE_URL": "http://s"}, cwd=project)
    assert settings.providers[Provider("my_shim")].base_url == "http://s"


def test_the_environment_beats_the_file(project: Path) -> None:
    settings = load_settings(env={"FRANCA_ANTHROPIC_TIMEOUT_S": "1"}, cwd=project)
    assert settings.providers[ANTHROPIC].timeout_s == 1.0


def test_overrides_beat_the_environment(project: Path) -> None:
    settings = load_settings(
        overrides={"providers": {"anthropic": {"timeout_s": 2}}},
        env={"FRANCA_ANTHROPIC_TIMEOUT_S": "1"},
        cwd=project,
    )
    assert settings.providers[ANTHROPIC].timeout_s == 2.0


def test_an_empty_environment_is_honoured_not_replaced(project: Path) -> None:
    """`env or os.environ` would reinstate the real environment, keys included."""
    settings = load_settings(env={}, cwd=project)
    assert settings.providers[ANTHROPIC].api_key is None


# -------------------------------------------------------------------- routes


def test_route_for_prefers_exact_then_longest_glob_then_default(project: Path) -> None:
    settings = load_settings(env={}, cwd=project)
    assert settings.route_for("openai:gpt-5").layers == ("trace", "tools", "budget", "retry")
    assert settings.route_for("anthropic:claude").layers == ("trace", "tools", "retry")


def test_the_longest_matching_glob_wins() -> None:
    settings = Settings(
        routes={
            "openai:*": RoutePolicy(layers=("broad",)),
            "openai:gpt-5*": RoutePolicy(layers=("narrow",)),
            "default": RoutePolicy(),
        }
    )
    assert settings.route_for("openai:gpt-5-mini").layers == ("narrow",)
    assert settings.route_for("openai:o3").layers == ("broad",)


def test_route_for_falls_back_when_there_is_no_default() -> None:
    assert Settings(routes={}).route_for("anything") == RoutePolicy()


# --------------------------------------------------------------------- typos


def test_a_misspelled_provider_field_fails_with_a_path(tmp_path: Path) -> None:
    """`timeout` instead of `timeout_s` must not silently leave the default in place."""
    (tmp_path / "franca.toml").write_text("[providers.anthropic]\ntimeout = 90\n", encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_settings(env={}, cwd=tmp_path)
    message = str(caught.value)
    assert "providers.anthropic.timeout" in message
    assert "franca.toml" in message


def test_an_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "franca.toml").write_text("nonsense = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="nonsense"):
        load_settings(env={}, cwd=tmp_path)


def test_a_bad_value_fails_without_echoing_it(tmp_path: Path) -> None:
    """A rejected value may be a key, and an exception is a widely logged object."""
    (tmp_path / "franca.toml").write_text(
        '[providers.anthropic]\ntimeout_s = "sk-ant-CANARY"\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError) as caught:
        load_settings(env={}, cwd=tmp_path)
    reasons = [ln for ln in str(caught.value).splitlines() if "Reason:" in ln]
    assert reasons
    assert not any("CANARY" in line for line in reasons)


# ---------------------------------------------------------------------- keys


def test_keys_are_never_folded_into_settings(project: Path) -> None:
    """A rotated key must not need a rebuild, and no model may hold one."""
    settings = load_settings(env={"ANTHROPIC_API_KEY": "sk-ant"}, cwd=project)
    assert settings.providers[ANTHROPIC].api_key is None

    keys = SettingsKeyProvider(settings, {"ANTHROPIC_API_KEY": "sk-ant"})
    found = keys.key_for(ANTHROPIC)
    assert found is not None
    assert found.get_secret_value() == "sk-ant"


def test_the_key_lookup_chain_is_ordered(project: Path) -> None:
    settings = load_settings(env={}, cwd=project)
    env = {
        "MISTRAL_KEY": "from-api-key-env",
        "FRANCA_MISTRAL_API_KEY": "from-franca",
    }
    keys = SettingsKeyProvider(settings, env)
    got = keys.key_for(MISTRAL)
    assert got is not None
    assert got.get_secret_value() == "from-api-key-env"

    fallback = SettingsKeyProvider(settings, {"FRANCA_MISTRAL_API_KEY": "from-franca"})
    got = fallback.key_for(MISTRAL)
    assert got is not None
    assert got.get_secret_value() == "from-franca"


def test_an_explicit_api_key_wins_over_every_variable() -> None:
    settings = Settings(providers={ANTHROPIC: ProviderSettings(api_key=SecretStr("inline"))})
    keys = SettingsKeyProvider(settings, {"ANTHROPIC_API_KEY": "env"})
    got = keys.key_for(ANTHROPIC)
    assert got is not None
    assert got.get_secret_value() == "inline"


def test_an_unconfigured_provider_has_no_key() -> None:
    assert SettingsKeyProvider(Settings(), {}).key_for(Provider("nobody")) is None


def test_no_secret_reaches_any_rendering_surface() -> None:
    canary = "sk-ant-CANARY-40f1e9"
    settings = Settings(providers={ANTHROPIC: ProviderSettings(api_key=SecretStr(canary))})
    keys = SettingsKeyProvider(settings, {"ANTHROPIC_API_KEY": canary})
    for surface in (repr(keys), repr(settings), str(settings.model_dump())):
        assert canary not in surface
    assert "anthropic" in repr(keys)
