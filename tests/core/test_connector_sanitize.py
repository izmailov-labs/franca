"""Third-party text never reaches an exception message unsanitised.

Every case here comes from a real captured exchange rather than an imagined one.
The motivating example is OpenAI's 401 body, which echoes the credential that was
just submitted: "Incorrect API key provided: sk-...". A connector that copies a
provider message verbatim therefore writes a live key into logs and spans.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from franca.core.adapter import WireRequest
from franca.core.connector import Connector
from franca.core.endpoint import Endpoint
from franca.core.enums import AuthScheme
from franca.core.errors import ModelError
from franca.core.ids import ANTHROPIC, ANTHROPIC_MESSAGES, CHAT, Provider
from franca.core.keys import StaticKeyProvider
from franca.core.transport import RawResponse
from franca.testing import FakeClock

if TYPE_CHECKING:
    from collections.abc import Mapping
    from contextlib import AbstractAsyncContextManager

KEY = "sk-live-CANARY-8f21bd"


class _Canned:
    """Transport returning one canned response, or raising one canned exception."""

    def __init__(
        self, response: RawResponse | None = None, *, raises: Exception | None = None
    ) -> None:
        """Store the canned outcome."""
        self._response = response or RawResponse(status=200, headers={}, body=b"{}")
        self._raises = raises

    async def post(
        self, url: str, content: bytes, headers: Mapping[str, str], *, timeout_s: float
    ) -> RawResponse:
        """Return the canned response, or raise."""
        if self._raises is not None:
            raise self._raises
        return self._response

    async def get(self, url: str, headers: Mapping[str, str], *, timeout_s: float) -> RawResponse:
        """Return the canned response, or raise."""
        if self._raises is not None:
            raise self._raises
        return self._response

    def post_stream(
        self, url: str, content: bytes, headers: Mapping[str, str], *, timeout_s: float
    ) -> AbstractAsyncContextManager[Any]:
        """Not used by these tests."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Nothing pooled."""


def _connector(transport: _Canned) -> Connector:
    """A connector holding the canary key, over the Anthropic endpoint row."""
    endpoint = Endpoint(
        id="anthropic/chat/messages",
        provider=ANTHROPIC,
        capability=CHAT,
        dialect=ANTHROPIC_MESSAGES,
        base_url="https://api.anthropic.com",
        path="/v1/messages",
        auth=AuthScheme.x_api_key,
    )
    return Connector(
        endpoint,
        keys=StaticKeyProvider({ANTHROPIC: SecretStr(KEY)}),
        transport=transport,
        clock=FakeClock(),
    )


def _error_response(message: str, status: int = 401) -> RawResponse:
    """A provider error body carrying `message`."""
    body = json.dumps(
        {"type": "error", "error": {"type": "authentication_error", "message": message}}
    )
    return RawResponse(status=status, headers={}, body=body.encode())


async def test_a_provider_that_echoes_the_key_does_not_leak_it() -> None:
    """The real OpenAI 401 shape: the submitted credential comes back in the message."""
    echoed = f"Incorrect API key provided: {KEY}. You can find your API key at ..."
    conn = _connector(_Canned(_error_response(echoed)))

    with pytest.raises(ModelError) as exc:
        await conn.send(WireRequest(body={}))

    assert KEY not in str(exc.value)
    assert KEY not in repr(exc.value)
    assert "<scrubbed>" in str(exc.value)
    assert exc.value.failure_class == "auth"


async def test_the_raw_payload_is_still_available_for_error_from() -> None:
    """Scrubbing the message must not blind `adapter.error_from`, which reads `raw`."""
    conn = _connector(_Canned(_error_response("Overloaded", status=529)))

    with pytest.raises(ModelError) as exc:
        await conn.send(WireRequest(body={}))

    assert exc.value.raw is not None
    assert exc.value.raw["error"]["type"] == "authentication_error"


async def test_control_characters_cannot_forge_log_structure() -> None:
    conn = _connector(_Canned(_error_response("boom\nERROR fake line\r\n2026-01-01 spoofed")))

    with pytest.raises(ModelError) as exc:
        await conn.send(WireRequest(body={}))

    assert "\n" not in str(exc.value)
    assert "\r" not in str(exc.value)


async def test_an_unbounded_provider_message_is_truncated() -> None:
    conn = _connector(_Canned(_error_response("A" * 50_000)))

    with pytest.raises(ModelError) as exc:
        await conn.send(WireRequest(body={}))

    assert len(str(exc.value)) < 600
    assert str(exc.value).endswith("...")


async def test_a_transport_exception_carrying_the_key_does_not_leak_it() -> None:
    """A URL with an embedded credential is a normal way for this to happen."""
    conn = _connector(_Canned(raises=OSError(f"connect failed for https://x/?key={KEY}")))

    with pytest.raises(ModelError) as exc:
        await conn.send(WireRequest(body={}))

    assert KEY not in str(exc.value)
    assert exc.value.failure_class == "transport"
    assert exc.value.retryable is True


async def test_the_transport_error_names_the_exception_type() -> None:
    """`str(exc)` alone is often empty; the class name is what makes it diagnosable."""
    conn = _connector(_Canned(raises=TimeoutError()))

    with pytest.raises(ModelError, match="TimeoutError") as exc:
        await conn.send(WireRequest(body={}))

    assert exc.value.failure_class == "transport"


async def test_a_message_without_a_key_is_left_readable() -> None:
    """Sanitising must not destroy the diagnostic value of an ordinary message."""
    conn = _connector(_Canned(_error_response("max_tokens: Field required", status=400)))

    with pytest.raises(ModelError, match="max_tokens: Field required") as exc:
        await conn.send(WireRequest(body={}))

    assert exc.value.status == 400


async def test_a_missing_key_still_reports_without_material() -> None:
    endpoint = Endpoint(
        id="anthropic/chat/messages",
        provider=ANTHROPIC,
        capability=CHAT,
        dialect=ANTHROPIC_MESSAGES,
        base_url="https://api.anthropic.com",
        path="/v1/messages",
        auth=AuthScheme.x_api_key,
    )
    conn = Connector(
        endpoint,
        keys=StaticKeyProvider({Provider("other"): SecretStr(KEY)}),
        transport=_Canned(),
        clock=FakeClock(),
    )

    with pytest.raises(ModelError, match="FRANCA_ANTHROPIC_API_KEY") as exc:
        conn.headers(WireRequest(body={}))

    assert KEY not in str(exc.value)
    assert exc.value.failure_class == "auth"
