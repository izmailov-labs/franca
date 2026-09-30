"""Model profiles: what franca believes about a model, and how a belief is found.

A profile is *data about a model*, not about an endpoint. `EndpointFeatures` says
what a URL supports; a profile says what `claude-opus-4-5` does with a prefill, or
whether `gpt-5.2` honours `temperature`. Adapters read profiles and never branch on
a model name, which is what keeps a new model a table row rather than a code change.

Every flag on a subclass is tri-state for the same reason `EndpointFeatures` is:
`True` verified present, `False` verified absent, `None` nobody checked. An adapter
that acts on a `None` records the flag name in `WireRequest.unverified`, so an
assumption is visible in the trace instead of silently baked in.

Resolution is longest-prefix over rows, overlaid on a per-provider default. The
overlay uses `model_dump(exclude_unset=True)`, which is the whole reason a row may
say `thinking=None` and mean "verified absent, do not inherit the default": an
explicit `None` is *set*, an omitted field is not.
"""

import re
from datetime import date
from typing import Protocol

from pydantic import BaseModel, Field, HttpUrl

from franca.core.errors import ProfileError
from franca.core.ids import Provider, ProviderField

_VENDOR_PREFIX = re.compile(r"^(?:(?:anthropic|openai|google)\.)+", re.IGNORECASE)
"""A Bedrock or Vertex vendor namespace glued onto the front of the real model id."""

_SUFFIXES = re.compile(r"(?:\[1m\]|@\d{8}|-\d{8}|-latest)+$", re.IGNORECASE)
"""Trailing decorations that name a variant of one model: context window, date, alias."""


def normalize_model_id(model: str) -> str:
    """Reduce a model identifier to the stem that profile rows are keyed on.

    Surrounding whitespace goes first. Then a leading `anthropic.`, `openai.` or
    `google.` vendor namespace is removed -- Bedrock and Vertex prefix the same model
    franca already has a row for -- followed by any run of trailing decorations: the
    `[1m]` context-window marker, an `@YYYYMMDD` or `-YYYYMMDD` snapshot date, and
    the `-latest` alias. Matching is case-insensitive; whatever survives keeps the
    case it was given.

    The trailing decorations are stripped as one greedy run rather than one pass per
    kind, which makes the function idempotent for *any* input rather than only for
    realistic ones: `normalize_model_id(normalize_model_id(m)) == normalize_model_id(m)`.

    Args:
        model: A model identifier as a caller or a provider spelled it.

    Returns:
        The normalised stem, e.g. `"claude-opus-4-5"` for
        `"anthropic.claude-opus-4-5-20251101"`.
    """
    return _SUFFIXES.sub("", _VENDOR_PREFIX.sub("", model.strip()))


class BaseProfile(BaseModel, frozen=True, extra="forbid"):
    """What every profile carries, whatever capability it describes.

    `extra="forbid"` is load-bearing: a misspelled flag in a hand-written table would
    otherwise be accepted, ignored, and read as "unverified" forever.

    Attributes:
        provider: Who serves the models this row describes.
        model_prefix: The normalised model stem this row applies to. Matching is by
            prefix, so `"claude-opus"` covers every `claude-opus-*`; the empty string
            matches everything and is how a provider-wide fallback row is written.
        source: Where the claims came from -- vendor documentation, a changelog entry
            or a conformance run.
        verified: The day the claims were last checked against the live provider.
        notes: Anything a reader of the table needs that no flag captures.
    """

    provider: ProviderField
    model_prefix: str
    source: HttpUrl | None = None
    verified: date | None = None
    notes: str = ""

    @property
    def is_verified(self) -> bool:
        """Whether anyone has checked this row against the live provider."""
        return self.verified is not None

    @property
    def name(self) -> str:
        """The row's identity as `"<provider>:<model_prefix>"`, e.g. `"anthropic:claude-opus"`.

        A provider-wide fallback row, whose `model_prefix` is empty, therefore reads
        as `"anthropic:"`.
        """
        return f"{self.provider}:{self.model_prefix}"


class ProfileResolver[P: BaseProfile](Protocol):
    """The seam between "which model is this" and "what do we believe about it".

    `ProfileTable` satisfies this structurally, and so does a resolver that reads a
    database or asks a service, which is why the leaf depends on the protocol.
    """

    def resolve(self, provider: Provider, model: str) -> P:
        """Return the profile that applies to one model.

        Args:
            provider: The provider the call will be addressed to.
            model: The model identifier, un-normalised.

        Returns:
            The applicable profile.

        Raises:
            ProfileError: If nothing applies and there is no default to fall back on.
        """


class ProfileTable[P: BaseProfile](BaseModel, frozen=True):
    """An in-memory `ProfileResolver`: rows plus a per-provider default to overlay.

    Attributes:
        verified_on: When the table as a whole was last reviewed.
        profiles: The rows, in no particular order; resolution sorts by prefix length.
        defaults: One fallback row per provider, carrying the beliefs that hold for
            everything that provider serves.
    """

    verified_on: date | None = None
    profiles: tuple[P, ...] = ()
    defaults: dict[ProviderField, P] = Field(default_factory=dict)

    def resolve(self, provider: Provider, model: str) -> P:
        """Find the longest matching row and overlay it on the provider's default.

        The model id is normalised first, so a dated or region-prefixed spelling
        matches the same row as the bare one. Among the rows for `provider` whose
        `model_prefix` the normalised id starts with, the longest wins.

        The overlay is `model_copy(update=row.model_dump(exclude_unset=True))`: a
        field the row set -- *including* one it set to `None` -- replaces the
        default's, and a field the row omitted inherits. `model_copy` does not
        validate, which is exactly why the row was validated when it was built.

        Args:
            provider: The provider the call will be addressed to.
            model: The model identifier, un-normalised.

        Returns:
            The row, the default, or the row overlaid on the default.

        Raises:
            ProfileError: If no row matches and the provider has no default.
        """
        key = normalize_model_id(model)
        rows = [
            p for p in self.profiles if p.provider == provider and key.startswith(p.model_prefix)
        ]
        row = max(rows, key=lambda p: len(p.model_prefix), default=None)
        base = self.defaults.get(provider)
        if row is None:
            if base is None:
                raise ProfileError(
                    f"no profile or default for {provider}:{model};"
                    " register one via Registrar.profiles"
                )
            return base
        if base is None:
            return row
        return base.model_copy(update=row.model_dump(exclude_unset=True))
