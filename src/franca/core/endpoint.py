"""Endpoint rows and the contracts around choosing one.

An `Endpoint` is one row of data: which provider you authenticate with, which capability
it serves, which wire dialect it speaks, where it lives and how the key travels. Plugins
add endpoints as rows, not code, so nothing here branches on a vendor name.

`EndpointFeatures` is deliberately tri-state. `True` means franca has verified the feature
against the live provider, `False` means it has verified the feature is absent, and `None`
means nobody has checked. Selection treats those three differently (see `Requirements`),
and the distinction is what lets an untested plugin endpoint be usable without being
silently trusted.

Everything in this module is a version-bump-protected contract: `Requirements` and
`Selection` are the inputs and outputs of dialect selection, which lives in a later
milestone but is declared here because this is its home.
"""

from typing import Literal

from pydantic import BaseModel, Field

from franca.core.enums import AuthScheme
from franca.core.ids import CapabilityField, Dialect, DialectField, ProviderField

type SelectionVia = Literal["pin", "profile", "native", "fallback"]
"""How a `Selection` was reached: a caller pin, a profile hint, native support, or fallback."""


class EndpointFeatures(BaseModel, frozen=True, extra="allow"):
    """What an endpoint is known to support, as a tri-state per feature.

    `True` is verified present, `False` is verified absent, and `None` is unverified.
    `extra="allow"` lets a plugin attach features by name without subclassing; those are
    reachable through `get` alongside the declared ones.
    """

    strict: bool | None = None
    server_state: bool | None = None
    artifacts: bool | None = None
    caching: bool | None = None
    hoisting: bool | None = None
    streaming: bool | None = None
    production: bool = True

    def get(self, name: str) -> bool | None:
        """Look a feature up by name.

        Declared fields are consulted first, then the extras a plugin supplied. An
        extra counts only if it is a `bool` or `None`; anything else, and any name
        nobody set, reads as unverified.

        Args:
            name: The feature name, e.g. `"strict"` or a plugin's own key.

        Returns:
            The feature's tri-state value, or `None` when unknown.
        """
        if name in type(self).model_fields:
            declared: bool | None = getattr(self, name)
            return declared
        extra: object = (self.model_extra or {}).get(name)
        if extra is None or isinstance(extra, bool):
            return extra
        return None


class Endpoint(BaseModel, frozen=True):
    """One addressable provider surface: who, what, which wire, where and how to sign.

    `id` is the trace key and the cassette directory name, so it must be stable across
    releases; `"anthropic/chat/messages"` is the canonical shape.
    """

    id: str
    provider: ProviderField
    capability: CapabilityField
    dialect: DialectField
    base_url: str
    path: str
    auth: AuthScheme
    extra_headers: dict[str, str] = Field(default_factory=dict)
    features: EndpointFeatures = Field(default_factory=EndpointFeatures)

    def url(self, path: str | None = None) -> str:
        """Join the base URL with a path by plain concatenation.

        No normalisation happens: neither side is stripped or slashed, so a row is
        responsible for its own `base_url` / `path` boundary.

        Args:
            path: A path overriding `self.path`; `None` means use the endpoint's own.
                An empty string is honoured as an override, not treated as `None`.

        Returns:
            The full request URL.
        """
        return self.base_url + (path if path is not None else self.path)


class Requirements(BaseModel, frozen=True):
    """What a caller demands of the endpoint that dialect selection may pick.

    `needs` names `EndpointFeatures` that must not be `False`; `must_be_verified`
    additionally rejects `None`. `allow_stubs` admits adapters whose status is `"stub"`.
    """

    pin: Dialect | None = None
    needs: frozenset[str] = frozenset()
    must_be_production: bool = True
    must_be_verified: bool = False
    allow_stubs: bool = False


class Selection(BaseModel, frozen=True):
    """The outcome of dialect selection, with enough context to explain itself.

    `unverified` lists the needed features the chosen endpoint leaves at `None`;
    `rejected` records, in order, the candidates that were passed over and why.
    """

    endpoint: Endpoint
    dialect: Dialect
    adapter: str
    via: SelectionVia
    unverified: frozenset[str]
    rejected: tuple[str, ...]
