"""Executable specification for `franca.core.adapter.WireRequest`.

`WireRequest` is the object every adapter hands the `Connector`, so its defaults and
its immutability are load-bearing for every capability package. The adapter ABCs are
not covered here yet; they land alongside `BaseProfile`, `SseEvent` and `Job`.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from franca.core.adapter import WireRequest

SECRET = b"sk-ant-api03-do-not-leak-me"


def test_defaults() -> None:
    wire = WireRequest()

    assert wire.method == "POST"
    assert wire.path is None
    assert wire.body is None
    assert wire.content is None
    assert wire.headers == {}
    assert wire.stream is False
    assert wire.unverified == frozenset()
    assert isinstance(wire.unverified, frozenset)


def test_frozen_rejects_assignment() -> None:
    wire = WireRequest()

    # The static signature already forbids this; the test proves the runtime does too.
    with pytest.raises(ValidationError):
        wire.method = "GET"  # type: ignore[misc]


def test_content_is_absent_from_repr_and_str() -> None:
    wire = WireRequest(content=SECRET, headers={"Content-Type": "multipart/form-data"})

    assert wire.content == SECRET
    assert SECRET.decode() not in repr(wire)
    assert SECRET.decode() not in str(wire)
    assert "content=" not in repr(wire)
    # The other fields still show, so the redaction is targeted rather than wholesale.
    assert "multipart/form-data" in repr(wire)


def test_method_rejects_put() -> None:
    with pytest.raises(ValidationError):
        WireRequest(method="PUT")  # type: ignore[arg-type]


def test_method_accepts_get() -> None:
    assert WireRequest(method="GET").method == "GET"


@pytest.mark.parametrize(
    "given",
    [{"sampling_allowed", "prefill_allowed"}, ["sampling_allowed", "prefill_allowed"]],
)
def test_unverified_coerces_to_frozenset(given: set[str] | list[str]) -> None:
    # Statically the field is a frozenset; at runtime pydantic coerces any iterable of str.
    wire = WireRequest(unverified=given)  # type: ignore[arg-type]

    assert isinstance(wire.unverified, frozenset)
    assert wire.unverified == frozenset({"sampling_allowed", "prefill_allowed"})


def test_headers_default_is_fresh_per_instance() -> None:
    first = WireRequest()
    second = WireRequest()

    assert first.headers is not second.headers
    first.headers["x-tier"] = "wire"
    assert second.headers == {}
    assert WireRequest().headers == {}


def test_get_without_body_or_content_is_valid() -> None:
    wire = WireRequest(method="GET", path="/v1/videos/vid_123")

    assert wire.method == "GET"
    assert wire.path == "/v1/videos/vid_123"
    assert wire.body is None
    assert wire.content is None


def test_body_and_content_may_both_be_set() -> None:
    wire = WireRequest(body={"ignored": True}, content=b"--boundary--")

    assert wire.body == {"ignored": True}
    assert wire.content == b"--boundary--"


def test_body_keeps_nested_structure() -> None:
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    wire = WireRequest(body=body)

    assert wire.body == body


def test_stream_flag_round_trips() -> None:
    assert WireRequest(body={}, stream=True).stream is True
    assert WireRequest(body={}).stream is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"body": {}},
        {"body": {}, "path": "/v1/responses"},
        {"method": "GET", "path": "/v1/jobs/abc"},
        {"body": {"ignored": True}, "content": b"--boundary--"},
        {"body": {}, "headers": {"x-tier": "wire"}},
        {"body": {}, "stream": True},
    ],
)
def test_connector_spec_constructions_validate(kwargs: dict[str, Any]) -> None:
    """Every shape the `Connector` specification builds must validate unchanged."""
    wire = WireRequest(**kwargs)

    for name, value in kwargs.items():
        assert getattr(wire, name) == value
