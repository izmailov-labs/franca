"""Tests for `franca.core._json`: canonical, byte-deterministic request bodies."""

import json

import pytest

from franca.core._json import dumps


def test_returns_bytes() -> None:
    assert isinstance(dumps({"a": 1}), bytes)
    assert isinstance(dumps([]), bytes)


def test_sorts_keys_at_every_nesting_level() -> None:
    body = dumps({"z": {"b": 1, "a": 2}, "a": [{"y": 1, "x": 2}]})
    assert body == b'{"a":[{"x":2,"y":1}],"z":{"a":2,"b":1}}'


def test_compact_separators_have_no_whitespace() -> None:
    body = dumps({"a": [1, 2, {"b": None}]})
    assert b" " not in body
    assert body == b'{"a":[1,2,{"b":null}]}'


def test_non_ascii_is_kept_as_utf8_not_escaped() -> None:
    text = "héllo — 日本語 \U0001f642"
    body = dumps({"text": text})
    assert body == f'{{"text":"{text}"}}'.encode()
    assert b"\\u" not in body


def test_insertion_order_does_not_change_bytes() -> None:
    first = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    second = {"max_tokens": 1, "messages": [{"content": "hi", "role": "user"}], "model": "m"}
    assert first == second
    assert dumps(first) == dumps(second)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, b"null"),
        (True, b"true"),
        (False, b"false"),
        (0, b"0"),
        (-42, b"-42"),
        (1.5, b"1.5"),
        ("", b'""'),
        ([], b"[]"),
        ([1, "two", None, False], b'[1,"two",null,false]'),
        ({}, b"{}"),
    ],
)
def test_scalar_and_list_rendering(value: object, expected: bytes) -> None:
    assert dumps(value) == expected


def test_output_round_trips_through_stdlib() -> None:
    payload = {"a": [1, {"b": "ü"}], "c": None, "d": 2.5}
    assert json.loads(dumps(payload)) == payload


def test_non_serialisable_raises_type_error_unwrapped() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        dumps({1, 2})


def test_nested_non_serialisable_raises_type_error() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        dumps({"ok": [1, {"bad": {1, 2}}]})
