# franca

[![CI](https://github.com/izmailov-labs/franca/actions/workflows/ci.yml/badge.svg)](https://github.com/izmailov-labs/franca/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/franca.svg)](https://pypi.org/project/franca/)
[![Python versions](https://img.shields.io/pypi/pyversions/franca.svg)](https://pypi.org/project/franca/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

One request shape for every LLM wire dialect.

A *lingua franca* for model APIs. You build one request, and franca decides which wire
shape the target model actually speaks, translates into it, and translates the answer
back. Three chat dialects are wired today behind one intermediate representation; five,
plus the image and video wires, are the target.

Three axes are kept separate, because they vary independently:

| Axis | Meaning | Owns |
| --- | --- | --- |
| **Provider** | who you authenticate with | endpoint rows, settings, key lookup |
| **Dialect** | the shape of the bytes | the adapter, the typed request, stream mapping |
| **Model** | the weights | the profile row that adapters read |

That separation is the point. A compatibility surface that serves one provider's wire
shape under another's host, and remaps model names on the way, needs no special case
anywhere else.

> **Status:** `0.1.0` — scaffolding. No public API yet.

## What is wired today

Three dialects serve five providers, because a wire shape and a provider are different
things: xAI and DeepSeek speak OpenAI's bytes under their own hosts and keys, so they
are rows of data rather than code.

| Provider | Endpoint | Dialect | Auth header |
| --- | --- | --- | --- |
| `anthropic` | `/v1/messages` | `anthropic_messages` | `x-api-key` |
| `openai` | `/v1/chat/completions` | `openai_chat` | `Authorization` |
| `google` | `/v1beta/models/{model}:generateContent` | `google_generate_content` | `x-goog-api-key` |
| `xai` | `/v1/chat/completions` | `openai_chat` | `Authorization` |
| `deepseek` | `/v1/chat/completions` | `openai_chat` | `Authorization` |

Capability profiles are measured, never transcribed. Anthropic has six verified rows --
`haiku-4-5`, `sonnet-4-5`, `opus-4-5`, `sonnet-5`, `opus-5`, `fable-5-1` -- and every
other provider's default row claims nothing, because no probe returned a usable verdict:
no key was available for Google, xAI or DeepSeek, and OpenAI checks quota before it
validates parameters, so every probe there came back 429. An unverified capability reads
as `None`, which is not the same as `False`.

## Models

**Every model these five providers serve is callable.** Reachability is what an endpoint
row and a dialect adapter buy you, and both exist for all five: `gpt-6-astra`,
`gemini-3.8-flash`, `grok-4.6`, `deepseek-v4-pro` and the Claude models are all one request
away. Nothing below narrows that.

What varies is how much franca has *measured* about a given model, which is a different
question and the one the profile table answers. A measured row is not permission to call a
model — it is a record of what that model accepts on the wire, so franca can refuse an
illegal parameter locally, with a field path, instead of paying a round trip to find out.

A model id is normalised before it is matched, so every spelling of one model lands on one
row. An `anthropic.` / `openai.` / `google.` vendor namespace is stripped, as are the `[1m]`
context marker, an `@YYYYMMDD` or `-YYYYMMDD` snapshot date and `-latest`; the longest
matching `model_prefix` then wins, and that row is overlaid on the provider's default, so a
row states only what it measured and inherits the rest.

Six rows carry measured contracts, and they are all Anthropic today — not because the
other providers are second class, but because no working key for them was available when
the sweep was taken. Each cell below was obtained by sending the parameter to the live API
and recording whether the request was legal:

| Model prefix | Sampling | Thinking | Token budget | Effort | Prefill | Verified |
| --- | --- | --- | --- | --- | --- | --- |
| `claude-haiku-4-5` | yes | `budget` | yes | — | yes | 2026-09-07 |
| `claude-sonnet-4-5` | yes | `budget` | yes | — | yes | 2026-09-07 |
| `claude-opus-4-5` | yes | `budget` | yes | `high` | yes | 2026-09-07 |
| `claude-sonnet-5` | no | `adaptive` | no | `high` | no | 2026-09-16 |
| `claude-opus-5` | no | `adaptive` | no | `high` | no | 2026-09-16 |
| `claude-fable-5-1` | no | `adaptive` | no | `high` | no | 2026-09-16 |

Three groups, and the boundaries do not line up: `effort` splits `opus-4-5` away from its
own generation, while sampling and thinking split it the other way. That is the argument
for per-model rows over an `if provider ==` branch.

Every other model resolves to its provider's default row, which claims `streaming` and
nothing else — each remaining capability is `None`, meaning nobody has checked. `None` is
not `False`, and the difference runs the permissive way: with no measured row franca
**sends** the parameter and lets the provider judge it. An unmeasured model is fully
callable; it just gets no local pre-flight. Closing that gap is a data change in
`chat/profiles.py` and needs no code.

One genuine coverage limit, as distinct from an unmeasured one: three of the five planned
chat dialects are implemented. `openai_responses` and `google_interactions` have ids
reserved in `core/ids.py` but no adapter yet, so OpenAI models are reachable through Chat
Completions rather than `/v1/responses`.

## Install

```bash
uv add franca
```

Requires Python 3.12+. Fully typed; ships a `py.typed` marker.

## Making a call

Nothing below is re-exported from `franca` yet — the `0.1.0` status line above is honest,
and these are internal import paths that will move when the registry lands. They are the
shapes `tests/mock` drives over real HTTP, so they work today.

### One request, one response

```python
import asyncio

from franca.chat.dialects.anthropic_messages import AnthropicMessagesAdapter
from franca.chat.endpoints import ANTHROPIC_MESSAGES_ENDPOINT
from franca.chat.ir import Item, PromptPackage, SystemBlock
from franca.chat.model import ChatModel
from franca.chat.profiles import CHAT_PROFILES
from franca.core.clock import AsyncioClock
from franca.core.connector import Connector
from franca.core.ids import ANTHROPIC
from franca.core.settings import SettingsKeyProvider, load_settings
from franca.transports.httpx import HttpxTransport

MODEL = "claude-sonnet-5"


async def main() -> None:
    transport = HttpxTransport()  # needs the extra: uv add "franca[http]"
    try:
        model = ChatModel(
            model=MODEL,
            connector=Connector(
                ANTHROPIC_MESSAGES_ENDPOINT,
                # reads ANTHROPIC_API_KEY at call time, never stores it
                keys=SettingsKeyProvider(load_settings()),
                transport=transport,
                clock=AsyncioClock(),
            ),
            adapter=AnthropicMessagesAdapter(),
            profile=CHAT_PROFILES.resolve(ANTHROPIC, MODEL),
            clock=AsyncioClock(),
        )
        res = await model.complete(
            PromptPackage(
                system=(SystemBlock(text="You are terse."),),
                items=(Item(role="user", kind="text", text="Name one primary colour."),),
                max_output_tokens=32,
            )
        )
        print(res.text)  # "Blue."
        print(res.usage.input_tokens, res.usage.output_tokens)
        print(res.served_model, res.stop_reason)  # what the wire itself reported
        print(res.trace.endpoint_id, res.trace.latency_ms)
    finally:
        await transport.aclose()


asyncio.run(main())
```

`complete()` builds the wire request, sends it, translates the answer and stamps a
`CallTrace`. The response is dialect-neutral: `items` in the same vocabulary the request
used, `usage` with four counters, `stop_reason` in the wire's own words, `served_model` so
a silent alias swap is visible, and `raw` for the untranslated payload.

Switching providers changes three arguments and nothing else — the endpoint row, the
adapter and the provider the profile resolves against:

```python
from franca.chat.dialects.openai_chat import OpenAIChatAdapter
from franca.chat.endpoints import XAI_CHAT_ENDPOINT
from franca.core.ids import XAI

# ...same ChatModel call, with:
#   Connector(XAI_CHAT_ENDPOINT, ...), adapter=OpenAIChatAdapter(),
#   profile=CHAT_PROFILES.resolve(XAI, "grok-4.6")
```

### Streaming

The IR-level stream is not built yet, and the leaf says so rather than shipping half a
feature. `ChatModel.stream()` raises a `ModelError` with `failure_class="unsupported"`:

```text
streaming lands in M1; use complete() for now
```

One level down works today. The adapter shapes a streaming request, the connector opens
the response as server-sent events, and `parse_sse` frames them — so you can consume raw
events now and swap to typed deltas when M1 lands:

```python
wire = model.build(package, stream=True)  # wire.stream is True
async with model.connector.stream(wire) as events:
    async for event in events:
        print(event.event, event.data)
        # message_start {"type":"message_start","message":{...}}
```

`SseEvent` is framing, not meaning: an `event` name, its `data`, and an `id` that persists
across blocks per WHATWG. The `[DONE]` sentinel is consumed by the parser rather than
yielded, and the stream ends when the body does. Reassembling text from the deltas is the
caller's job until the chat delta type exists.

Two dialect details the adapter already handles, so this loop does not have to: Anthropic
and the OpenAI-compatible wires set a body flag, while Google signals streaming by swapping
the path to `:streamGenerateContent?alt=sse`. `model.build(..., stream=True)` produces
whichever the selected row needs.

## Development

```bash
make install     # sync the dev group, install pre-commit hooks
make all         # lint, typecheck, coverage, build
```

Live provider calls are opt-in: `make test-contract` runs them against real keys.

### Testing against a mock model

[aimock](https://aimock.copilotkit.dev) serves the OpenAI and Anthropic wire shapes from
JSON fixtures, over real HTTP and real server-sent events, with no key and no bill. It
runs through `npx`, so node is the only prerequisite.

```bash
make test-mock   # spawns aimock on a free port, runs tests/mock, stops it
make aimock      # runs it in the foreground on :4010 for your own experiments
```

Point a model at it by rebasing an endpoint row. Only the host changes; the row's `id`,
and so every trace and cassette name, stays what it is in production:

```python
from franca.chat.endpoints import OPENAI_CHAT_ENDPOINT

endpoint = OPENAI_CHAT_ENDPOINT.model_copy(update={"base_url": "http://127.0.0.1:4010"})
```

`make aimock` starts the server with `AIMOCK_API_KEYS=franca-mock-key`, so send that key.
Fixtures live in `tests/mock/fixtures/` and match on the last user message; a request that
matches none comes back as a 404 `No fixture matched`. Set `AIMOCK_BASE_URL` to run the
suite against a server you started yourself.

## License

MIT — see [LICENSE](LICENSE).
