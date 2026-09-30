"""Cassette replay: a `Transport` that answers from a recorded script instead of a socket.

A `Cassette` is an ordered list of `Exchange` rows, and `ScriptedTransport` walks it one
row per call. That ordering is deliberate: a test that says "these three requests, in this
order" is a specification of the call sequence, not just of the last body, and a retry loop
that suddenly sends four requests fails loudly instead of silently reusing a response.

Matching is on `(method, url, canonical request)`. Canonical means the JSON is re-encoded
through `franca.core._json.dumps`, so key order and whitespace in the cassette are
irrelevant and only the logical content matters; a body that is not UTF-8 JSON is matched
by its SHA-256 digest instead, which keeps multipart uploads out of the file. A request
mismatch raises `AssertionError` carrying a unified diff of the two bodies -- that diff is
the whole point of this module, because an adapter regression should read as "you now send
`max_tokens: 32` where the golden says 16", not as "assertion failed".

Streaming replay and `RecordingTransport` land in M1; `post_stream` raises until then.
"""

import difflib
import hashlib
import json
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field

from franca.core import _json
from franca.core.transport import RawResponse, RawStream


class Exchange(BaseModel, frozen=True):
    """One recorded request/response pair.

    The request side is stored as a single string rather than as structured JSON so that a
    body which is not JSON at all -- a multipart upload, an image -- can be pinned by digest
    in the same field, and so that the file stays diffable in review.

    Attributes:
        method: The HTTP method the request must use.
        url: The absolute URL the request must be sent to, compared verbatim.
        request: The expected request body: canonical sorted-key JSON text, or the SHA-256
            hexdigest of the bytes when the body is not UTF-8 JSON. `None` means the
            request carried no body, which is the normal case for a `GET`.
        status: The status code to answer with. Non-2xx is allowed and expected -- the
            status map in `Connector` is tested from cassettes.
        headers: The response headers to answer with, verbatim.
        body: The response body as text; it is UTF-8 encoded on replay.
    """

    method: Literal["POST", "GET"]
    url: str
    request: str | None = None
    status: int = 200
    headers: dict[str, str] = Field(default_factory=dict)
    body: str = ""


class Cassette(BaseModel, frozen=True):
    """An ordered script of exchanges, and the JSON file it lives in.

    Attributes:
        exchanges: The exchanges in the order they are expected to happen.
    """

    exchanges: tuple[Exchange, ...] = ()

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Load a cassette from a JSON file.

        Args:
            path: The file to read, UTF-8 encoded.

        Returns:
            The validated cassette.

        Raises:
            OSError: If the file cannot be read.
            ValidationError: If the document does not match the schema. The message names
                the offending field path, which is what makes a hand-edited cassette
                cheap to fix.
        """
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def to_path(self, path: Path) -> None:
        """Write the cassette to a JSON file, indented for review.

        Synchronous on purpose: recording buffers in memory and persists here, so no
        `async def` ever touches the filesystem (ruff `ASYNC230`).

        Args:
            path: The file to write. Its parent directory must already exist.
        """
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


class ScriptedTransport:
    """Replays a `Cassette`, in order, and records what was actually sent.

    Satisfies `franca.core.transport.Transport` structurally; it is injected into a
    `Connector` exactly where `HttpxTransport` would go, and nothing above it can tell the
    difference. `aclose` is a no-op because there is nothing pooled to release.

    Attributes:
        calls: Every call made, as `(method, url, content, headers)`, appended before the
            exchange is matched so a failing call is still visible. Headers are the
            plaintext ones the connector produced -- including the API key -- which is what
            makes wire assertions possible; this list is a test-local artefact and must
            never be serialised.
        exits: How many `post_stream` contexts have been left. Streaming replay lands in
            M1; the counter exists now because the streaming tests assert on it.
    """

    def __init__(self, cassette: Cassette) -> None:
        """Start at the first exchange of `cassette` with an empty call log.

        Args:
            cassette: The script to replay.
        """
        self._cassette = cassette
        self._index = 0
        self.calls: list[tuple[str, str, bytes | None, dict[str, str]]] = []
        self.exits = 0

    def __repr__(self) -> str:
        """Name the position in the script; never the recorded bodies or headers."""
        return f"{type(self).__name__}(at={self._index}/{len(self._cassette.exchanges)})"

    async def post(
        self, url: str, content: bytes, headers: Mapping[str, str], *, timeout_s: float
    ) -> RawResponse:
        """Match a POST against the next exchange and answer with its recorded response.

        Args:
            url: Absolute URL, compared verbatim against the exchange.
            content: Request body, canonicalised before comparison.
            headers: Complete request headers; recorded in `calls`, never matched on.
            timeout_s: Ignored -- nothing here can time out.

        Returns:
            The recorded response.

        Raises:
            AssertionError: If the script is exhausted, or the method, URL or canonical
                request body does not match. A body mismatch carries a unified diff.
        """
        return _response(self._match("POST", url, content, headers))

    async def get(self, url: str, headers: Mapping[str, str], *, timeout_s: float) -> RawResponse:
        """Match a GET against the next exchange and answer with its recorded response.

        Args:
            url: Absolute URL, compared verbatim against the exchange.
            headers: Complete request headers; recorded in `calls`, never matched on.
            timeout_s: Ignored -- nothing here can time out.

        Returns:
            The recorded response.

        Raises:
            AssertionError: If the script is exhausted, or the method or URL does not
                match. A `GET` carries no body, so the exchange's `request` must be `None`.
        """
        return _response(self._match("GET", url, None, headers))

    def post_stream(
        self, url: str, content: bytes, headers: Mapping[str, str], *, timeout_s: float
    ) -> AbstractAsyncContextManager[RawStream]:
        """Not implemented yet; streaming replay is an M1 deliverable.

        Args:
            url: Absolute URL.
            content: Request body.
            headers: Complete request headers.
            timeout_s: Per-request timeout.

        Raises:
            NotImplementedError: Always, until SSE cassettes exist.
        """
        raise NotImplementedError("streaming replay lands in M1")

    async def aclose(self) -> None:
        """Release nothing: a scripted transport holds no pooled resources."""

    def _match(
        self, method: str, url: str, content: bytes | None, headers: Mapping[str, str]
    ) -> Exchange:
        """Record the call, advance the script, and return the exchange it must match."""
        self.calls.append((method, url, content, dict(headers)))
        index = self._index
        exchanges = self._cassette.exchanges
        if index >= len(exchanges):
            raise AssertionError(
                f"cassette exhausted: it holds {len(exchanges)} exchange(s) but the code "
                f"under test sent a {index + 1}th request at index {index}: {method} {url}"
            )
        expected = exchanges[index]
        self._index = index + 1
        if expected.method != method or expected.url != url:
            raise AssertionError(
                f"exchange {index} mismatch: expected {expected.method} {expected.url}, "
                f"got {method} {url}"
            )
        wanted = None if expected.request is None else _canonicalise(expected.request)
        actual = None if content is None else _canonical(content)
        if wanted != actual:
            raise AssertionError(
                f"exchange {index} request body mismatch for {method} {url}:\n"
                f"{_diff(wanted, actual)}"
            )
        return expected


def _response(exchange: Exchange) -> RawResponse:
    """Turn a recorded exchange into the `RawResponse` the transport hands back."""
    return RawResponse(
        status=exchange.status,
        headers=dict(exchange.headers),
        body=exchange.body.encode("utf-8"),
    )


def _canonical(content: bytes) -> str:
    """Canonicalise actual request bytes: sorted-key JSON text, or a SHA-256 hexdigest."""
    try:
        parsed: object = json.loads(content.decode("utf-8"))
    except ValueError:
        return hashlib.sha256(content).hexdigest()
    return _json.dumps(parsed).decode("utf-8")


def _canonicalise(text: str) -> str:
    """Canonicalise a recorded request: re-encode it when it is JSON, else leave it alone.

    Leaving a non-JSON value untouched is what lets the same field hold a SHA-256
    hexdigest; re-encoding a JSON one is what makes the key order a cassette was written
    with irrelevant.
    """
    try:
        parsed: object = json.loads(text)
    except ValueError:
        return text
    return _json.dumps(parsed).decode("utf-8")


def _pretty(text: str) -> list[str]:
    """Split `text` into diffable lines, re-indenting it first when it is JSON."""
    try:
        parsed: object = json.loads(text)
    except ValueError:
        return text.splitlines()
    return json.dumps(parsed, indent=2, sort_keys=True).splitlines()


def _diff(expected: str | None, actual: str | None) -> str:
    """Render a unified diff of two canonical request bodies, labelled expected/actual."""
    return "\n".join(
        difflib.unified_diff(
            _pretty(expected or ""),
            _pretty(actual or ""),
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )
    )
