"""WHATWG `EventSource` framing cases for `franca.core.sse.parse_sse`.

Every case feeds a small async generator of pre-split lines, exactly the shape a
`RawStream` hands the parser, and asserts on the dispatched `SseEvent`s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from franca.core.sse import SseEvent, parse_sse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


async def _lines(*items: bytes) -> AsyncIterator[bytes]:
    """Yield the given lines one at a time, as a transport would."""
    for item in items:
        yield item


async def _collect(*items: bytes) -> list[SseEvent]:
    """Run the parser over `items` and gather every dispatched event."""
    return [event async for event in parse_sse(_lines(*items))]


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------


async def test_single_event() -> None:
    assert await _collect(b"data: hello\n", b"\n") == [SseEvent(data="hello")]


async def test_two_events_back_to_back() -> None:
    events = await _collect(b"data: one\n", b"\n", b"data: two\n", b"\n")
    assert events == [SseEvent(data="one"), SseEvent(data="two")]


async def test_multi_line_data_is_joined_with_newline() -> None:
    events = await _collect(b"data: first\n", b"data: second\n", b"data: third\n", b"\n")
    assert events == [SseEvent(data="first\nsecond\nthird")]


async def test_comment_lines_are_ignored() -> None:
    events = await _collect(b": keep-alive\n", b"data: x\n", b":another comment\n", b"\n")
    assert events == [SseEvent(data="x")]


async def test_empty_data_buffer_is_not_dispatched() -> None:
    """A blank line with nothing buffered dispatches nothing and resets the event type."""
    events = await _collect(b"\n", b"event: ping\n", b"\n", b"data: x\n", b"\n")
    assert events == [SseEvent(event=None, data="x")]


async def test_explicit_empty_data_field_dispatches_an_empty_event() -> None:
    """WHATWG: `data:` with an empty value fills the buffer, so the block is dispatched."""
    assert await _collect(b"data:\n", b"\n") == [SseEvent(data="")]


async def test_trailing_event_without_final_blank_line_is_discarded() -> None:
    events = await _collect(b"data: complete\n", b"\n", b"data: pending\n")
    assert events == [SseEvent(data="complete")]


async def test_empty_input_yields_nothing() -> None:
    assert await _collect() == []


# --------------------------------------------------------------------------------------
# Line endings and decoding
# --------------------------------------------------------------------------------------


async def test_crlf_line_endings() -> None:
    events = await _collect(b"event: e\r\n", b"data: a\r\n", b"data: b\r\n", b"\r\n")
    assert events == [SseEvent(event="e", data="a\nb")]


async def test_lf_line_endings() -> None:
    events = await _collect(b"event: e\n", b"data: a\n", b"data: b\n", b"\n")
    assert events == [SseEvent(event="e", data="a\nb")]


async def test_bare_cr_line_endings() -> None:
    events = await _collect(b"event: e\r", b"data: a\r", b"data: b\r", b"\r")
    assert events == [SseEvent(event="e", data="a\nb")]


async def test_already_stripped_lines() -> None:
    events = await _collect(b"event: e", b"data: a", b"data: b", b"")
    assert events == [SseEvent(event="e", data="a\nb")]


async def test_only_one_line_ending_is_stripped() -> None:
    """A payload that itself ends in CR keeps it: exactly one terminator is removed."""
    assert await _collect(b"data: a\r\r\n", b"\n") == [SseEvent(data="a\r")]


async def test_bom_on_first_line_is_stripped() -> None:
    events = await _collect(b"\xef\xbb\xbfdata: a\n", b"\n")
    assert events == [SseEvent(data="a")]


async def test_bom_on_a_later_line_is_not_stripped() -> None:
    """Only the first line loses a BOM; later it is part of the field name, which is then unknown."""
    events = await _collect(b"data: a\n", b"\n", b"\xef\xbb\xbfdata: b\n", b"\n")
    assert events == [SseEvent(data="a")]


async def test_invalid_utf8_is_replaced_not_raised() -> None:
    assert await _collect(b"data: caf\xff\n", b"\n") == [SseEvent(data="caf\ufffd")]


async def test_utf8_payload_survives_per_line_decoding() -> None:
    assert await _collect("data: café\n".encode(), b"\n") == [SseEvent(data="café")]


# --------------------------------------------------------------------------------------
# Fields
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (b"data:x\n", "x"),
        (b"data: x\n", "x"),
        (b"data:  x\n", " x"),
    ],
)
async def test_exactly_one_leading_space_is_removed_from_the_value(
    line: bytes, expected: str
) -> None:
    assert await _collect(line, b"\n") == [SseEvent(data=expected)]


async def test_field_line_with_no_colon_has_an_empty_value() -> None:
    """WHATWG: `data` alone is the `data` field with value "", and unknown bare fields are ignored."""
    events = await _collect(b"ping\n", b"data\n", b"\n")
    assert events == [SseEvent(data="")]


async def test_value_may_contain_further_colons() -> None:
    events = await _collect(b"data: a:b:c\n", b"\n")
    assert events == [SseEvent(data="a:b:c")]


async def test_event_type_is_set_and_reset_between_events() -> None:
    events = await _collect(b"event: add\n", b"data: 1\n", b"\n", b"data: 2\n", b"\n")
    assert [e.event for e in events] == ["add", None]


async def test_id_persists_to_the_next_event_when_not_resent() -> None:
    events = await _collect(b"id: 7\n", b"data: a\n", b"\n", b"data: b\n", b"\n")
    assert [e.id for e in events] == ["7", "7"]


async def test_id_is_replaced_when_resent() -> None:
    events = await _collect(b"id: 1\n", b"data: a\n", b"\n", b"id: 2\n", b"data: b\n", b"\n")
    assert [e.id for e in events] == ["1", "2"]


async def test_id_containing_nul_is_ignored() -> None:
    events = await _collect(b"id: ok\n", b"data: a\n", b"\n", b"id: bad\x00\n", b"data: b\n", b"\n")
    assert [e.id for e in events] == ["ok", "ok"]


async def test_id_set_on_a_block_with_no_data_still_persists() -> None:
    """The id buffer is not part of the per-event reset, so a data-less block still sets it."""
    events = await _collect(b"id: 3\n", b"\n", b"data: a\n", b"\n")
    assert events == [SseEvent(data="a", id="3")]


async def test_retry_field_is_ignored() -> None:
    events = await _collect(b"retry: 10000\n", b"data: a\n", b"\n")
    assert events == [SseEvent(data="a")]


async def test_unknown_field_is_ignored() -> None:
    events = await _collect(b"x-custom: 1\n", b"data: a\n", b"\n")
    assert events == [SseEvent(data="a")]


# --------------------------------------------------------------------------------------
# [DONE]
# --------------------------------------------------------------------------------------


async def test_done_ends_the_stream_and_is_not_yielded() -> None:
    events = await _collect(b"data: a\n", b"\n", b"data: [DONE]\n", b"\n", b"data: b\n", b"\n")
    assert events == [SseEvent(data="a")]


async def test_nothing_after_done_is_read() -> None:
    async def poisoned() -> AsyncIterator[bytes]:
        yield b"data: a\n"
        yield b"\n"
        yield b"data: [DONE]\n"
        yield b"\n"
        pytest.fail("parse_sse advanced the source past [DONE]")

    events = [event async for event in parse_sse(poisoned())]
    assert events == [SseEvent(data="a")]


async def test_done_is_only_recognised_as_a_whole_payload() -> None:
    """`[DONE]` inside a larger or multi-line payload is ordinary data."""
    events = await _collect(
        b"data: [DONE]\n", b"data: extra\n", b"\n", b"data: not [DONE]\n", b"\n"
    )
    assert events == [SseEvent(data="[DONE]\nextra"), SseEvent(data="not [DONE]")]


# --------------------------------------------------------------------------------------
# WHATWG worked examples and a real dialect
# --------------------------------------------------------------------------------------


async def test_whatwg_example_ids_and_leading_spaces() -> None:
    """The specification's `: test stream` example, terminated so the third block dispatches."""
    events = await _collect(
        b": test stream\n",
        b"\n",
        b"data: first event\n",
        b"id: 1\n",
        b"\n",
        b"data:second event\n",
        b"id\n",
        b"\n",
        b"data:  third event\n",
        b"\n",
    )
    assert events == [
        SseEvent(data="first event", id="1"),
        SseEvent(data="second event", id=""),
        SseEvent(data=" third event", id=""),
    ]


async def test_whatwg_example_empty_and_newline_payloads() -> None:
    """The specification's `data` / `data` `data` / `data:` example."""
    events = await _collect(b"data\n", b"\n", b"data\n", b"data\n", b"\n", b"data:\n")
    assert events == [SseEvent(data=""), SseEvent(data="\n")]


async def test_whatwg_example_named_events() -> None:
    events = await _collect(
        b"event: add\n",
        b"data: 73857293\n",
        b"\n",
        b"event: remove\n",
        b"data: 2153\n",
        b"\n",
        b"event: add\n",
        b"data: 113411\n",
        b"\n",
    )
    assert events == [
        SseEvent(event="add", data="73857293"),
        SseEvent(event="remove", data="2153"),
        SseEvent(event="add", data="113411"),
    ]


async def test_anthropic_message_start_sequence() -> None:
    payload = b'{"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[]}}'
    events = await _collect(b"event: message_start\n", b"data: " + payload + b"\n", b"\n")
    assert events == [SseEvent(event="message_start", data=payload.decode())]


# --------------------------------------------------------------------------------------
# The event type itself
# --------------------------------------------------------------------------------------


def test_sse_event_is_frozen_and_slotted() -> None:
    event = SseEvent(event="e", data="d", id="i")
    assert not hasattr(event, "__dict__")
    with pytest.raises(AttributeError):
        event.data = "changed"  # type: ignore[misc]


def test_sse_event_defaults() -> None:
    assert SseEvent() == SseEvent(event=None, data="", id=None)
