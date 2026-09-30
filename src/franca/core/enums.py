"""Closed vocabularies: every member corresponds to a branch in franca's own code.

Closed on purpose, and mechanically so. A `StrEnum` cannot be extended at runtime --
`EnumMeta.__call__` rejects non-member `_missing_` returns, and pydantic compiles enum
validators at class creation -- which is exactly the guarantee wanted here. Adding an
`AuthScheme` member without adding the matching branch to `Connector.headers()` would
produce a value nothing honours.

Vocabularies a plugin may extend as data are open instead; see `core.ids`.
"""

from enum import StrEnum


class Severity(StrEnum):
    """How loudly a `Finding` speaks; `LintClient` raises at its configured level."""

    error = "error"
    warn = "warn"
    info = "info"


class AuthScheme(StrEnum):
    """Which header carries the API key. One member per branch in `Connector.headers`."""

    x_api_key = "x-api-key"
    bearer = "bearer"
    x_goog_api_key = "x-goog-api-key"


class Strategy(StrEnum):
    """How structured output is obtained when a model lacks native support."""

    native = "native"
    tool_calling = "tool_calling"
    json_mode = "json_mode"


class JobStatus(StrEnum):
    """Lifecycle of an asynchronous generation job, as the poll loop sees it."""

    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    expired = "expired"


class StateMode(StrEnum):
    """Whether conversation state lives with the caller or on the provider."""

    stateless = "stateless"
    server = "server"


class ThinkingMode(StrEnum):
    """How a reasoning budget is expressed on the wire; adapters branch on this."""

    off = "off"
    budget = "budget"
    effort = "effort"
    level = "level"
    adaptive = "adaptive"


class UnverifiedPolicy(StrEnum):
    """What to do when a selected endpoint leaves a needed feature unverified."""

    allow = "allow"
    warn = "warn"
    reject = "reject"
