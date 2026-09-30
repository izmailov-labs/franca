"""Value types every capability package shares: assets, usage, jobs and the call trace.

These are the nouns that cross layer boundaries. An adapter produces them, a leaf
stamps a `CallTrace` onto them, middleware reads them and a runner persists them, so
they carry no behaviour beyond what all four need. Everything here is a frozen
pydantic model: a response that middleware could mutate in place would make a retry
or a fallback unreproducible.

Two fields are deliberately asymmetric about privacy. `Job.raw` is the provider's
untranslated payload -- kept for debugging, excluded from `model_dump()` so a
persisted transcript does not silently grow a vendor envelope. `Asset.b64` is the
opposite case: it is hidden from `repr()` because it is megabytes of base64, but it
*is* the result of an image or video call, so it must survive a dump.

`HasTrace` and `HasUsage` are the structural seams middleware reasons through. They
are protocols rather than base classes because a capability package's response type
already has a base class, and because a plugin's response should qualify without
importing anything from here.
"""

from typing import Any, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, Field

from franca.core.endpoint import SelectionVia
from franca.core.enums import JobStatus, Severity
from franca.core.ids import Dialect


class Asset(BaseModel, frozen=True):
    """One binary result: an image, a video, an audio clip or an opaque file.

    A provider hands back exactly one of three shapes -- a URL to fetch, inline
    base64, or a file handle to reference in a later call -- so all three fields are
    optional and an adapter fills the one the wire gave it.

    Attributes:
        kind: What the bytes are, independent of how they are carried.
        url: Where to fetch the asset, when the provider returned a link.
        b64: The asset inline, base64-encoded. Excluded from `repr()` because it can
            be megabytes, but *not* excluded from `model_dump()`: for an image or
            video call this field is the result.
        file_id: A provider-side handle, when the asset stays on their storage.
        mime: The media type the provider reported, e.g. `"image/png"`.
        meta: Anything else worth keeping, such as width, height or duration.
    """

    kind: Literal["image", "video", "audio", "file"]
    url: str | None = None
    b64: str | None = Field(default=None, repr=False)
    file_id: str | None = None
    mime: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel, frozen=True):
    """Token counts for one call, in franca's own vocabulary rather than a vendor's.

    Every field defaults to zero so a provider that reports nothing still yields a
    usable record, and so `sum` over an empty round list has an identity.

    Attributes:
        input_tokens: Tokens billed for the prompt.
        output_tokens: Tokens billed for the completion.
        cache_read_tokens: Prompt tokens served from a provider-side cache.
        cache_write_tokens: Prompt tokens written into a provider-side cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        """Add two usage records field by field, returning a new one.

        This is for accumulating across tool rounds, where each round is a separate
        billed call. It is *not* for streams: a stream's usage frames restate the
        running total, so a stream replaces the record rather than adding to it.

        Args:
            other: The usage to add to this one.

        Returns:
            A new `Usage` with each field summed; neither operand is modified.
        """
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


class Job(BaseModel, frozen=True):
    """An asynchronous generation job as the poll loop sees it.

    Video, and some image endpoints, answer a submit with a handle and make the
    caller poll. This is that handle plus whatever the last poll said about it.

    Attributes:
        id: The provider's job identifier, as it must be echoed back when polling.
        status: Where the job is in its lifecycle.
        progress: Completion fraction when the provider reports one, else `None`.
        error: The provider's failure message, when it failed.
        raw: The last decoded poll payload. Kept for debugging, excluded from
            `model_dump()` and from `repr()` so a persisted job stays franca-shaped.
    """

    id: str
    status: JobStatus
    progress: float | None = None
    error: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    @property
    def finished(self) -> bool:
        """Whether the job reached a terminal state and polling should stop."""
        return self.status in {JobStatus.done, JobStatus.failed, JobStatus.expired}


class CallTrace(BaseModel, frozen=True):
    """What one call actually did, recorded once so nobody re-parses a raw body.

    A leaf stamps this onto every response it returns, middleware refines the fields
    it owns, and a runner reads it for metrics. Adapters never construct one.

    Attributes:
        endpoint_id: The `Endpoint.id` the call went to; stable across releases, so
            it doubles as the trace key and the cassette directory name.
        dialect: The wire dialect that was spoken.
        selection_via: How that dialect was arrived at.
        unverified: Profile flags the adapter applied while their value was `None`,
            i.e. assumptions nobody has confirmed against the live provider.
        latency_ms: Wall time of the final, successful request.
        ttft_ms: Time to first token; streams only, `None` otherwise.
        attempts: How many requests were made; re-stamped by `RetryClient`.
        tool_rounds: Model round trips spent invoking tools.
        tool_calls: Individual tool invocations across those rounds.
        partial: Set when a stream failed after at least one delta had arrived.
    """

    endpoint_id: str
    dialect: Dialect
    selection_via: SelectionVia
    unverified: frozenset[str] = frozenset()
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    attempts: int = 1
    tool_rounds: int = 0
    tool_calls: int = 0
    partial: bool = False


class Traced(BaseModel, frozen=True):
    """Base for anything that leaves a leaf carrying the trace of the call that made it.

    `trace` is `None` only in the gap between the adapter parsing a response and the
    leaf stamping it, so code above the leaf may treat it as present.
    """

    trace: CallTrace | None = None

    def with_trace(self, trace: CallTrace) -> Self:
        """Return a copy of this object carrying `trace`.

        Args:
            trace: The trace to stamp on.

        Returns:
            A new instance of the same concrete type; the original is unchanged.
        """
        return self.model_copy(update={"trace": trace})


@runtime_checkable
class HasTrace(Protocol):
    """Anything a leaf can stamp a `CallTrace` onto and middleware can read back."""

    @property
    def trace(self) -> CallTrace | None:
        """The trace of the call that produced this object, if it has been stamped."""

    def with_trace(self, trace: CallTrace) -> Self:
        """Return a copy of this object carrying `trace`.

        Args:
            trace: The trace to stamp on.

        Returns:
            A new instance of the same concrete type.
        """


@runtime_checkable
class HasUsage(Protocol):
    """Anything that can report token usage, so metering never branches on a type."""

    @property
    def usage(self) -> Usage | None:
        """Token counts for the call that produced this object, when known."""


class Finding(BaseModel, frozen=True):
    """One thing a lint rule objects to about a request, before it reaches the wire.

    Attributes:
        rule_id: Stable identifier of the rule that fired, e.g. `"prefill_denied"`.
        severity: How loudly it speaks; `LintClient` raises at its configured level.
        message: What is wrong, in one sentence.
        where: The location in the request, as a dotted path, when there is one.
        fix: What to change, when the rule can say.
    """

    rule_id: str
    severity: Severity
    message: str
    where: str | None = None
    fix: str | None = None
