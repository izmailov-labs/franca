"""Server-sent events framing: the WHATWG `EventSource` algorithm over a line iterator.

One parser serves every dialect. Anthropic Messages, OpenAI Chat Completions and
Responses, and Gemini `alt=sse` all speak standard SSE; only the JSON inside each
`data:` payload differs, and that is the adapter's business, not this module's. The
parser takes *lines* rather than a byte stream because that is what `RawStream`
yields -- the transport already did the line splitting -- and returns framed events
with the line endings, comments, multi-line `data:` joins and blank-line dispatch of
the WHATWG "interpret the stream" steps applied.

Two deliberate departures from a plain `EventSource`: the OpenAI `[DONE]` sentinel
ends the stream instead of being surfaced as an event, and `retry:` is ignored
because franca has no reconnection loop for it to configure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_BOM = b"\xef\xbb\xbf"
_DONE = "[DONE]"
_NUL = "\u0000"


@dataclass(frozen=True, slots=True)
class SseEvent:
    """One framed server-sent event.

    A frozen, slotted dataclass rather than a pydantic model on purpose: the parser
    creates one instance per token on the streaming hot path, and only the parser
    creates them, from bytes it has already validated. Pydantic's per-field
    validation would buy nothing here and cost a measurable slice of every delta.

    Attributes:
        event: The `event:` field, or `None` when the block set none.
        data: The `data:` lines of the block joined with a newline, trailing newline
            removed, exactly as WHATWG dispatches them.
        id: The most recent `id:` field seen on the stream. Per WHATWG it persists
            across events until the server sends a new one, so it may come from an
            earlier block.
    """

    event: str | None = None
    data: str = ""
    id: str | None = None


def _strip_line_ending(line: bytes) -> bytes:
    """Remove exactly one trailing line ending: CRLF, LF or a bare CR."""
    if line.endswith(b"\r\n"):
        return line[:-2]
    if line.endswith((b"\n", b"\r")):
        return line[:-1]
    return line


async def parse_sse(lines: AsyncIterator[bytes]) -> AsyncIterator[SseEvent]:
    """Frame a stream of SSE lines into events.

    Implements the WHATWG `EventSource` "interpret the stream" steps against an
    iterator that yields one line per item. Lines may carry their line ending
    (CRLF, LF or a bare CR) or arrive already stripped; each is decoded as UTF-8
    with invalid sequences replaced, and a leading byte-order mark is dropped from
    the first line only.

    Field handling follows the specification: `data:` accumulates, `event:` sets the
    type for the next dispatch, `id:` sets the last event id (ignored when the value
    contains U+0000), lines starting with `:` are comments, and `retry:` or unknown
    fields are ignored. A blank line dispatches: nothing when no `data:` was seen, an
    event otherwise. A dispatched payload equal to `[DONE]` ends the iteration
    without being yielded and without reading any further line. Data still buffered
    when the input ends is discarded, as WHATWG requires.

    The source iterator is never closed here; the transport context that produced
    it owns its lifetime.

    Args:
        lines: The raw SSE lines, one per item, as produced by `RawStream`.

    Yields:
        One `SseEvent` per dispatched block, in stream order.
    """
    event_type = ""
    data_lines: list[str] = []
    last_id: str | None = None
    first = True
    async for raw in lines:
        line = _strip_line_ending(raw)
        if first:
            first = False
            line = line.removeprefix(_BOM)
        text = line.decode("utf-8", errors="replace")

        if not text:
            if not data_lines:
                event_type = ""
                continue
            data = "\n".join(data_lines)
            event = event_type or None
            event_type = ""
            data_lines = []
            if data == _DONE:
                return
            yield SseEvent(event=event, data=data, id=last_id)
            continue

        if text.startswith(":"):
            continue

        field, sep, value = text.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]

        if field == "data":
            data_lines.append(value)
        elif field == "event":
            event_type = value
        elif field == "id" and _NUL not in value:
            last_id = value
        # "retry" and unknown fields are ignored.
