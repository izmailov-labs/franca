"""Exception hierarchy and the one safe way to render a pydantic `ValidationError`.

Everything franca raises derives from `FrancaError`, so a caller can catch the whole
library in one clause. Below it the tree splits by *who* has to act:

- `ModelError` -- something went wrong talking to, or selecting, a model. It carries a
  `failure_class` so middleware and the eval runner can branch on the kind of failure
  without string-matching messages, and a `retryable` flag that `RetryClient` honours.
  `SelectionError`, `JobTimeoutError` and `JobFailedError` are `ModelError`s with fixed
  status and failure class.
- `ProfileError` -- a profile lookup failed; the fix is registering or correcting a row.
- `ConfigError` -- settings, plugin wiring or layer scoping is wrong; the fix is in
  configuration, not at the call site. `PluginConflictError` is the duplicate-name case.

`format_validation` exists because `str(ValidationError)` prints the rejected input, and
inputs routinely hold API keys. Every wrapped `ValidationError` in `src/` goes through it.
"""

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import ValidationError

from franca.core.ids import Provider

type FailureClass = Literal[
    "transport",
    "auth",
    "rate_limit",
    "provider",
    "parse",
    "timeout",
    "selection",
    "unsupported",
    "request",
    "lint",
]
"""Coarse kind of a `ModelError`, stable enough to branch on and to put in a trace row."""


class FrancaError(Exception):
    """Base class for every exception franca raises."""


class ModelError(FrancaError):
    """A model call, or the selection leading up to one, failed.

    `str(err)` is the message alone; the structured fields are attributes. `raw` holds
    the provider's decoded error payload when there was one and is deliberately left
    out of `repr()`, because such payloads can echo request content.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None,
        provider: Provider,
        retryable: bool,
        failure_class: FailureClass,
        raw: Mapping[str, Any] | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        """Record what failed and how the caller may react to it.

        Args:
            message: Human-readable description; becomes `str(err)`.
            status: HTTP status the provider answered with, or `None` when the failure
                happened before or without an HTTP exchange.
            provider: The provider the call was addressed to.
            retryable: Whether repeating the same call may succeed.
            failure_class: Coarse kind of failure; see `FailureClass`.
            raw: The provider's decoded error payload, if any.
            retry_after_s: Provider-requested backoff in seconds, if it sent one.
        """
        super().__init__(message)
        self.message = message
        self.status = status
        self.provider = provider
        self.retryable = retryable
        self.failure_class: FailureClass = failure_class
        self.raw = raw
        self.retry_after_s = retry_after_s

    def __str__(self) -> str:
        """Return the message and nothing else."""
        return self.message

    def __repr__(self) -> str:
        """Return the class name, message and structured fields; never `raw`."""
        fields = ", ".join(self._repr_fields())
        return f"{type(self).__name__}({self.message!r}, {fields})"

    def _repr_fields(self) -> list[str]:
        return [
            f"status={self.status!r}",
            f"provider={self.provider!r}",
            f"retryable={self.retryable!r}",
            f"failure_class={self.failure_class!r}",
            f"retry_after_s={self.retry_after_s!r}",
        ]


class SelectionError(ModelError):
    """No endpoint, dialect or provider satisfied the request's requirements.

    Always `status=None`, `failure_class="selection"` and not retryable: repeating the
    same call cannot help, the requirements or the registry have to change.
    """

    def __init__(self, message: str, *, provider: Provider, reasons: Iterable[str]) -> None:
        """Record which provider was searched and why each candidate was rejected.

        Args:
            message: Human-readable summary; becomes `str(err)`.
            provider: The provider whose endpoints were considered.
            reasons: One entry per rejected candidate, saying why. Stored as a tuple.
        """
        super().__init__(
            message,
            status=None,
            provider=provider,
            retryable=False,
            failure_class="selection",
        )
        self.reasons: tuple[str, ...] = tuple(reasons)

    def _repr_fields(self) -> list[str]:
        return [*super()._repr_fields(), f"reasons={self.reasons!r}"]


class JobTimeoutError(ModelError):
    """An asynchronous job did not reach a terminal state before the poll deadline.

    Always `status=408`, `failure_class="timeout"` and not retryable at this layer: the
    job may still finish on the provider's side, and re-submitting would duplicate it.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Provider,
        raw: Mapping[str, Any] | None = None,
    ) -> None:
        """Record which provider's job timed out.

        Args:
            message: Human-readable summary; becomes `str(err)`.
            provider: The provider running the job.
            raw: The last decoded job payload seen before the deadline, if any.
        """
        super().__init__(
            message,
            status=408,
            provider=provider,
            retryable=False,
            failure_class="timeout",
            raw=raw,
        )


class JobFailedError(ModelError):
    """The provider reported an asynchronous job as failed or expired.

    Always `status=422`, `failure_class="provider"` and not retryable: the provider has
    already given a final answer for this job.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Provider,
        raw: Mapping[str, Any] | None = None,
    ) -> None:
        """Record which provider's job failed.

        Args:
            message: Human-readable summary; becomes `str(err)`.
            provider: The provider that ran the job.
            raw: The decoded terminal job payload, if any.
        """
        super().__init__(
            message,
            status=422,
            provider=provider,
            retryable=False,
            failure_class="provider",
            raw=raw,
        )


class ProfileError(FrancaError):
    """A profile lookup failed or a profile row is unusable for the request."""


class ConfigError(FrancaError):
    """Settings, plugin wiring or layer scoping is invalid; fix the configuration."""


class PluginConflictError(ConfigError):
    """Two registrants claimed the same name and neither asked to replace the other."""


def format_validation(exc: ValidationError) -> str:
    """Render a `ValidationError` as path + message per error, never the input value.

    `str(exc)` includes the rejected input, and inputs may hold secrets, so every
    wrapped `ValidationError` in franca goes through this instead.

    Args:
        exc: The error to render.

    Returns:
        One `"<dotted.loc>: <msg> [<type>]"` clause per error, joined with `"; "`.
    """
    return "; ".join(
        f"{'.'.join(map(str, e['loc']))}: {e['msg']} [{e['type']}]"
        for e in exc.errors(include_input=False, include_url=False)
    )
