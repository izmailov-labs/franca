"""Settings: the schema franca owns, and one call into whence to fill it.

The shapes here -- `RetryPolicy`, `RoutePolicy`, `ProviderSettings`, `Settings` --
are franca's, and they are the whole of what this module decides. Everything about
*getting* values into them (which file, which environment variables, in what order,
and what wins) belongs to `whence`, which exists for exactly this and is published as its own
package.

Three consequences worth knowing.

**Errors carry a file and a line.** whence keeps every value's origin through the
merge and into pydantic's `loc` paths, so a bad `timeout_s` reports
`franca.toml:4:9` rather than just naming the field. `ConfigError` still wraps it,
and no message ever contains the rejected value -- a rejected value may be a key.

**API keys are deliberately absent from `Settings`.** `SettingsKeyProvider` reads
them at call time from the environment, so rotating a key needs no rebuild and no
key is ever held in a model that something might dump. `api_key_env` names the
variable; it never holds the secret.

**The environment spelling is an explicit table, not a pattern.** The original plan
matched `FRANCA_<PROVIDER>_<KEY>` with a regex whose tie-break was "the shortest
provider prefix that leaves a known key wins" -- a heuristic, and one that cannot
distinguish a provider named `my_shim` from a provider `my` with a key `shim_...`.
`ENV_ALIASES` below is finite, generated from the known vendor slugs and the known
field names, and greppable. Any other provider uses whence's unambiguous nested
spelling, `FRANCA_PROVIDERS__MY_SHIM__BASE_URL`, where `__` separates levels and a
single `_` never does.
"""

import fnmatch
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, SecretStr, ValidationError
from whence import Config, Discovery, WhenceError

from franca.core.enums import Severity, UnverifiedPolicy
from franca.core.errors import ConfigError, FailureClass, format_validation
from franca.core.ids import Dialect, Provider, ProviderField
from franca.core.keys import VENDOR_ENV

__all__ = [
    "ENV_ALIASES",
    "ProviderSettings",
    "RetryPolicy",
    "RoutePolicy",
    "Settings",
    "SettingsKeyProvider",
    "load_settings",
]


class RetryPolicy(BaseModel, frozen=True):
    """Exponential backoff with jitter, in seconds."""

    max_attempts: int = 3
    base_s: float = 0.5
    cap_s: float = 8.0
    jitter_s: float = 0.25


class RoutePolicy(BaseModel, frozen=True):
    """How one route is built: which middleware, what it retries, where it falls back.

    `layers` names middleware in the layer registry, so a plugin's layer is reachable
    from a TOML file without any code change. `fallback` holds model specs walked in
    order when a call fails with a class listed in `fallback_on`; each is built without
    a fallback of its own, so a chain cannot loop.
    """

    layers: tuple[str, ...] = ("trace", "tools", "retry")
    retry: RetryPolicy = RetryPolicy()
    fallback: tuple[str, ...] = ()
    fallback_on: tuple[FailureClass, ...] = ("rate_limit", "provider", "transport", "timeout")
    lint_fail_on: Severity | None = None
    poll_interval_s: float | None = None
    job_timeout_s: float | None = None
    options: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ProviderSettings(BaseModel, frozen=True, extra="forbid"):
    """Per-provider connection settings.

    `extra="forbid"` is the point of this model: `timeout = 90` instead of
    `timeout_s = 90` is a typo that fails with a field path rather than being
    silently ignored, which is how a production timeout stays at its default.
    """

    api_key: SecretStr | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    timeout_s: float = 60.0
    extra_headers: dict[str, str] = Field(default_factory=dict)
    dialect: Dialect | None = None
    on_unverified: UnverifiedPolicy = UnverifiedPolicy.warn
    options: dict[str, Any] = Field(default_factory=dict)


class Settings(BaseModel, frozen=True, extra="forbid"):
    """Everything franca reads from configuration."""

    providers: dict[ProviderField, ProviderSettings] = Field(default_factory=dict)
    routes: dict[str, RoutePolicy] = Field(default_factory=lambda: {"default": RoutePolicy()})
    transport: str = "franca.transports.httpx:HttpxTransport"
    transport_options: dict[str, Any] = Field(default_factory=dict)
    allow_stubs: bool = False
    plugins: bool = True
    poll_interval_s: float = 5.0
    job_timeout_s: float = 600.0

    def route_for(self, spec: str) -> RoutePolicy:
        """Find the route policy governing a model spec.

        An exact key wins; otherwise the longest glob key that matches; otherwise
        `default`. Longest-first makes `openai:gpt-5*` beat `openai:*` without the
        caller having to order the table.

        Args:
            spec: A model spec such as `openai:gpt-5`.

        Returns:
            The matching policy, or the default one.
        """
        if spec in self.routes:
            return self.routes[spec]
        globbed = sorted(
            (key for key in self.routes if "*" in key and fnmatch.fnmatchcase(spec, key)),
            key=len,
            reverse=True,
        )
        if globbed:
            return self.routes[globbed[0]]
        return self.routes.get("default", RoutePolicy())


def _aliases() -> dict[str, str]:
    """Build the explicit environment alias table.

    Finite and generated rather than matched: every entry is a literal variable
    name mapped to a literal key path, so there is no tie-break rule to get wrong
    and `ENV_ALIASES` can simply be printed.
    """
    fields = ("base_url", "dialect", "timeout_s", "on_unverified", "api_key_env")
    return {
        f"FRANCA_{provider.upper()}_{field.upper()}": f"providers.{provider}.{field}"
        for provider in VENDOR_ENV
        for field in fields
    }


ENV_ALIASES: Mapping[str, str] = _aliases()
"""Short environment spellings for the known vendors, e.g. `FRANCA_ANTHROPIC_TIMEOUT_S`.

Any provider outside this table -- a plugin's, or an OpenAI-compatible shim -- uses
whence's nested form, `FRANCA_PROVIDERS__<NAME>__<FIELD>`.
"""


def load_settings(
    *,
    overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    file: Path | None = None,
    cwd: Path | None = None,
) -> Settings:
    """Load settings from overrides, the environment and a TOML file.

    Precedence is overrides, then the environment, then the file, then the field
    defaults. Discovery looks at `file` if given, then `$FRANCA_CONFIG`, then
    `franca.toml` beside the working directory, then `[tool.franca]` in
    `pyproject.toml`.

    Args:
        overrides: Values that outrank every source.
        env: The environment to read; defaults to `os.environ`.
        file: An explicit configuration file, which must exist.
        cwd: Where discovery starts; defaults to the process working directory.

    Returns:
        A validated, frozen `Settings`.

    Raises:
        ConfigError: If discovery, parsing or validation fails. The message names
            the offending field and, where the format reports one, its line.
    """
    # `env or os.environ` would quietly reinstate the real environment -- and with
    # it real API keys -- for a test that deliberately passed an empty one.
    environ = os.environ if env is None else env
    discovery = Discovery(
        app="franca",
        file=file,
        formats=("toml",),
        user_config=False,
        secrets_dir=None,
        dotenv=(),
    )
    try:
        config = Config.load(
            discovery=discovery,
            environ=environ,
            overrides=overrides,
            aliases=ENV_ALIASES,
            argv=(),
            cwd=cwd,
        )
        # strict=False: `providers` and `routes` are open mappings keyed by names
        # franca cannot know in advance, so unknown-key detection is left to the
        # models' own `extra="forbid"`, which is closed exactly where it should be.
        bound: Settings = config.bind(Settings, strict=False)
        return bound
    except WhenceError as exc:
        # whence is an implementation detail of how settings arrive; `ConfigError`
        # is what franca promises. The message is kept verbatim because whence has
        # already put the field path and the file in it.
        raise ConfigError(str(exc)) from exc
    except ValidationError as exc:  # pragma: no cover - whence maps these first
        raise ConfigError(format_validation(exc)) from exc


class SettingsKeyProvider:
    """Resolves API keys from settings and the environment, at call time.

    Keys are read on each lookup rather than folded into `Settings`, so a rotated
    key takes effect without rebuilding anything and no key is ever held in a model
    that something might serialise.

    The chain is: the provider's `api_key` if one was set explicitly, then the
    variable named by `api_key_env`, then `FRANCA_<PROVIDER>_API_KEY`, then the
    vendor's own conventional variable from `VENDOR_ENV`, so an existing shell
    configuration works unchanged.
    """

    def __init__(self, settings: Settings, environ: Mapping[str, str] | None = None) -> None:
        """Bind a key provider to settings and an environment.

        Args:
            settings: The loaded settings.
            environ: The environment to read; defaults to `os.environ`. A hermetic
                test passes the same mapping it gave `load_settings`.
        """
        self._settings = settings
        self._environ = os.environ if environ is None else environ

    def key_for(self, provider: Provider) -> SecretStr | None:
        """Return the key for a provider, or `None` when none is configured.

        Args:
            provider: The provider slug.

        Returns:
            The key wrapped in `SecretStr`, or `None`.
        """
        configured = self._settings.providers.get(provider)
        if configured is not None and configured.api_key is not None:
            return configured.api_key
        names = []
        if configured is not None and configured.api_key_env:
            names.append(configured.api_key_env)
        names.append(f"FRANCA_{provider.upper()}_API_KEY")
        vendor = VENDOR_ENV.get(provider)
        if vendor:
            names.append(vendor)
        for name in names:
            value = self._environ.get(name)
            if value:
                return SecretStr(value)
        return None

    def __repr__(self) -> str:
        """Name the configured providers only; no secret ever appears in the repr."""
        return f"{type(self).__name__}(providers=[{', '.join(sorted(self._settings.providers))}])"
