"""Executable specification for `franca.core.types`.

Three behaviours here are contracts other layers rely on and none of them is
obvious from the field list: `Job.raw` must not survive `model_dump()`, `Asset.b64`
must, and `Traced.with_trace` must copy rather than mutate. The protocol checks pin
that a plain frozen response qualifies structurally, so middleware never has to
import a base class to recognise one.
"""

import pytest
from pydantic import ValidationError

from franca.core.enums import JobStatus, Severity
from franca.core.ids import ANTHROPIC_MESSAGES
from franca.core.types import (
    Asset,
    CallTrace,
    Finding,
    HasTrace,
    HasUsage,
    Job,
    Traced,
    Usage,
)

B64 = "aGVsbG8td29ybGQtdGhpcy1pcy1ub3QtcmVhbGx5LWFuLWltYWdl"


class Answer(Traced, frozen=True):
    """A minimal leaf response: `Traced`'s trace plus a usage field of its own."""

    text: str = ""
    usage: Usage | None = None


def _stamp(obj: HasTrace, trace: CallTrace) -> HasTrace:
    """Push a value through the structural seam middleware uses, so mypy checks it."""
    return obj.with_trace(trace)


def _meter(obj: HasUsage) -> int:
    """Read tokens off anything that reports usage, without naming its type."""
    return 0 if obj.usage is None else obj.usage.output_tokens


def trace() -> CallTrace:
    return CallTrace(
        endpoint_id="anthropic/chat/messages",
        dialect=ANTHROPIC_MESSAGES,
        selection_via="native",
    )


def test_usage_defaults_are_zero() -> None:
    usage = Usage()

    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.cache_read_tokens == 0
    assert usage.cache_write_tokens == 0


def test_usage_add_is_field_wise() -> None:
    left = Usage(input_tokens=1, output_tokens=2, cache_read_tokens=3, cache_write_tokens=4)
    right = Usage(input_tokens=10, output_tokens=20, cache_read_tokens=30, cache_write_tokens=40)

    total = left + right

    assert total == Usage(
        input_tokens=11, output_tokens=22, cache_read_tokens=33, cache_write_tokens=44
    )


def test_usage_add_returns_a_new_frozen_instance() -> None:
    left = Usage(input_tokens=1)
    right = Usage(output_tokens=2)

    total = left + right

    assert total is not left
    assert total is not right
    assert left == Usage(input_tokens=1)
    assert right == Usage(output_tokens=2)
    with pytest.raises(ValidationError):
        total.input_tokens = 99  # type: ignore[misc]


def test_usage_add_with_default_is_identity() -> None:
    usage = Usage(input_tokens=7, cache_write_tokens=1)

    assert usage + Usage() == usage
    assert Usage() + usage == usage


@pytest.mark.parametrize(
    ("status", "finished"),
    [
        (JobStatus.queued, False),
        (JobStatus.running, False),
        (JobStatus.done, True),
        (JobStatus.failed, True),
        (JobStatus.expired, True),
    ],
)
def test_job_finished_covers_every_status(status: JobStatus, finished: bool) -> None:
    assert Job(id="vid_1", status=status).finished is finished


def test_job_defaults() -> None:
    job = Job(id="vid_1", status=JobStatus.queued)

    assert job.progress is None
    assert job.error is None
    assert job.raw == {}


def test_job_raw_is_excluded_from_dump_and_repr() -> None:
    job = Job(id="vid_1", status=JobStatus.running, raw={"internal_handle": "op/42"})

    assert job.raw == {"internal_handle": "op/42"}
    assert "raw" not in job.model_dump()
    assert job.model_dump() == {
        "id": "vid_1",
        "status": JobStatus.running,
        "progress": None,
        "error": None,
    }
    assert "internal_handle" not in repr(job)
    assert "raw=" not in repr(job)


def test_asset_b64_is_hidden_from_repr_but_kept_in_dump() -> None:
    asset = Asset(kind="image", b64=B64, mime="image/png")

    assert B64 not in repr(asset)
    assert "b64=" not in repr(asset)
    assert asset.model_dump()["b64"] == B64
    # The point of the asymmetry: a dumped asset is still the result of the call.
    assert Asset.model_validate(asset.model_dump()) == asset


def test_asset_defaults() -> None:
    asset = Asset(kind="video")

    assert asset.url is None
    assert asset.b64 is None
    assert asset.file_id is None
    assert asset.mime is None
    assert asset.meta == {}


def test_asset_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        Asset(kind="hologram")  # type: ignore[arg-type]


def test_call_trace_defaults() -> None:
    call = trace()

    assert call.endpoint_id == "anthropic/chat/messages"
    assert call.dialect == ANTHROPIC_MESSAGES
    assert call.selection_via == "native"
    assert call.unverified == frozenset()
    assert isinstance(call.unverified, frozenset)
    assert call.latency_ms == 0.0
    assert call.ttft_ms is None
    assert call.attempts == 1
    assert call.tool_rounds == 0
    assert call.tool_calls == 0
    assert call.partial is False


def test_call_trace_rejects_an_unknown_selection_via() -> None:
    with pytest.raises(ValidationError):
        CallTrace(
            endpoint_id="anthropic/chat/messages",
            dialect=ANTHROPIC_MESSAGES,
            selection_via="vibes",  # type: ignore[arg-type]
        )


def test_with_trace_copies_and_leaves_the_original_untouched() -> None:
    original = Answer(text="hi")
    call = trace()

    stamped = original.with_trace(call)

    assert original.trace is None
    assert stamped is not original
    assert stamped.trace == call
    assert stamped.text == "hi"
    assert isinstance(stamped, Answer)


def test_with_trace_replaces_an_existing_trace() -> None:
    first = trace()
    second = first.model_copy(update={"attempts": 3})

    stamped = Answer().with_trace(first).with_trace(second)

    assert stamped.trace == second


def test_a_traced_subclass_satisfies_both_protocols() -> None:
    answer = Answer(usage=Usage(output_tokens=5))

    assert isinstance(answer, HasTrace)
    assert isinstance(answer, HasUsage)
    assert _meter(answer) == 5
    assert _stamp(answer, trace()).trace == trace()


def test_a_plain_object_satisfies_neither_protocol() -> None:
    assert not isinstance(object(), HasTrace)
    assert not isinstance(object(), HasUsage)
    assert not isinstance(Usage(), HasTrace)
    # Traced carries no usage, so the two protocols really are independent.
    assert not isinstance(Traced(), HasUsage)


def test_finding_defaults() -> None:
    finding = Finding(rule_id="prefill_denied", severity=Severity.error, message="no prefill")

    assert finding.severity is Severity.error
    assert finding.where is None
    assert finding.fix is None
