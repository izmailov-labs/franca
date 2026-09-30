"""The one live call: the same M0 path as `test_first_call.py`, against the real API.

Everything the acceptance test replaces with a double is real here -- `HttpxTransport`
instead of `ScriptedTransport`, `AsyncioClock` instead of `FakeClock`, and a key out of
the environment instead of a fake one -- and nothing else changes. That is the value of
the test: a cassette can only prove that franca is self-consistent, and this is the one
place that checks the cassette was right about Anthropic.

It costs money and needs a key, so it is opt-in twice over. `-m "not contract"` in
`addopts` deselects it from every ordinary run, and even when it is selected it
skips cleanly unless `ANTHROPIC_API_KEY` is set. Run it with `uv run pytest -m contract`.

The assertions are deliberately thin. A live model's prose is not reproducible, so this
test pins only what the wire contract guarantees: the reply parses into a
`ModelResponse` at all, the provider billed for a prompt, it said why it stopped, and it
named the model that actually served the request -- which is how a silent alias or a
snapshot swap becomes visible.
"""

import os
import pathlib

import pytest
from pydantic import SecretStr

from franca.chat.dialects.anthropic_messages import AnthropicMessagesAdapter
from franca.chat.endpoints import ANTHROPIC_MESSAGES_ENDPOINT
from franca.chat.ir import Item, PromptPackage, SystemBlock
from franca.chat.model import ChatModel
from franca.chat.profile import ChatProfile
from franca.core.clock import AsyncioClock
from franca.core.connector import Connector
from franca.core.ids import ANTHROPIC, ANTHROPIC_MESSAGES
from franca.core.keys import StaticKeyProvider
from franca.transports.httpx import HttpxTransport

MODEL = "claude-opus-5"

PKG = PromptPackage(
    system=(SystemBlock(text="You are terse. Answer in three words or fewer."),),
    items=(Item(role="user", kind="text", text="Name one primary colour."),),
    max_output_tokens=1024,
)


def _key_from_dotenv(name: str) -> str | None:
    """Fall back to the gitignored repo-root `.env`, which is where keys live here.

    Kept inline rather than shared through a `conftest.py`, so this module stays
    runnable on its own. Six lines of duplication is the cheaper side of that trade.
    """
    path = pathlib.Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.strip().partition("=")
        if key.strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None


@pytest.mark.contract
async def test_a_live_anthropic_call_answers_and_bills_for_the_prompt() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY") or _key_from_dotenv("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY is not set; the live contract test needs a real key")

    transport = HttpxTransport()
    try:
        model = ChatModel(
            model=MODEL,
            connector=Connector(
                ANTHROPIC_MESSAGES_ENDPOINT,
                keys=StaticKeyProvider({ANTHROPIC: SecretStr(key)}),
                transport=transport,
                clock=AsyncioClock(),
            ),
            adapter=AnthropicMessagesAdapter(),
            profile=ChatProfile(provider=ANTHROPIC, model_prefix=""),
            clock=AsyncioClock(),
        )
        res = await model.complete(PKG)
    finally:
        await transport.aclose()

    assert res.usage.input_tokens > 0
    assert res.stop_reason is not None
    assert res.served_model is not None
    assert res.provider == ANTHROPIC
    assert res.dialect == ANTHROPIC_MESSAGES

    trace = res.trace
    assert trace is not None
    assert trace.endpoint_id == "anthropic/chat/messages"
    assert trace.selection_via == "native"
    assert trace.latency_ms > 0.0
