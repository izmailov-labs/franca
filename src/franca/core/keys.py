"""Key resolution: who hands out the API key for a provider, and from where.

`KeyProvider` is the seam. It is synchronous on purpose: `infer_provider` and
`ModelRegistry.client()` are synchronous, and a key lookup that had to be awaited would
force `async` up through both. Implementations hand out `pydantic.SecretStr` and never
unwrap it -- the one runtime unwrap site is `Connector.headers()` (plan section 9.1), and
a source-text test holds this module to that rule.

`VENDOR_ENV` names each first-party vendor's conventional environment variable. It is the
last fallback in the lookup chain (`FRANCA_<PROVIDER>_API_KEY` always wins), there so an
existing shell configuration works unchanged.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol

from pydantic import SecretStr

from franca.core.ids import ANTHROPIC, DEEPSEEK, GOOGLE, OPENAI, XAI, Provider


class KeyProvider(Protocol):
    """Resolves the API key for a provider slug.

    Structural: anything with a matching `key_for` satisfies it, so tests can pass a
    plain object and a plugin can source keys from a vault without subclassing.
    """

    def key_for(self, provider: Provider) -> SecretStr | None:
        """Return the key for `provider`, or `None` when none is configured.

        Synchronous by design; see the module docstring.

        Args:
            provider: The provider slug to look up.

        Returns:
            The key wrapped in `SecretStr`, or `None`.
        """


class StaticKeyProvider:
    """A `KeyProvider` over a fixed mapping, for tests and programmatic configuration.

    The mapping is copied at construction, so later changes to the caller's dict are not
    observed. Keys are handed out as the `SecretStr` they arrived as, never unwrapped.
    """

    def __init__(self, keys: Mapping[Provider, SecretStr]) -> None:
        """Copy `keys`; the instance does not alias the caller's mapping.

        Args:
            keys: One API key per provider slug.
        """
        self._keys: dict[Provider, SecretStr] = dict(keys)

    def key_for(self, provider: Provider) -> SecretStr | None:
        """Return the configured key for `provider`, or `None` when there is none.

        Args:
            provider: The provider slug to look up.

        Returns:
            The key wrapped in `SecretStr`, or `None`.
        """
        return self._keys.get(provider)

    def __repr__(self) -> str:
        """Name the configured providers only; no secret ever appears in the repr."""
        providers = ", ".join(sorted(self._keys))
        return f"{type(self).__name__}(providers=[{providers}])"


VENDOR_ENV: Mapping[Provider, str] = MappingProxyType(
    {
        ANTHROPIC: "ANTHROPIC_API_KEY",
        OPENAI: "OPENAI_API_KEY",
        GOOGLE: "GEMINI_API_KEY",
        XAI: "XAI_API_KEY",
        DEEPSEEK: "DEEPSEEK_API_KEY",
    }
)
"""Each first-party vendor's conventional key variable; read-only, keyed by provider slug."""
