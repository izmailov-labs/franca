"""Built-in chat endpoint rows: where franca knows how to reach a chat model.

An endpoint is a row of data, never a branch. Everything that varies between one
provider's chat surface and another's -- the host, the path, which header carries the
key, the version header the wire insists on, what the surface is known to support --
is a field on `Endpoint`, so adding a provider is adding a row here (or, for a plugin,
registering one from outside franca) rather than editing any code that sends requests.

Two fields on the row below are load-bearing beyond their obvious meaning.

`id` is the trace key and the cassette directory name, so `"anthropic/chat/messages"`
is a stable name that outlives any refactor: changing it invalidates recorded fixtures
and breaks every dashboard keyed on it.

`extra_headers` carries `anthropic-version`, which the Messages API requires on every
request. Keeping it here rather than in the adapter is what makes an API-version bump
a one-line data change that no adapter has to know about; the `Connector` merges it
under anything the adapter or the caller set, and always under the auth header.

`features` is tri-state per feature (`True` verified, `False` verified absent, `None`
unchecked), and the values below are claims about the Messages API that franca has
confirmed: it streams, it caches, and it is a production surface rather than a preview.
Everything nobody has checked is left `None` by omission rather than guessed at.
"""

from typing import Final

from franca.core.endpoint import Endpoint, EndpointFeatures
from franca.core.enums import AuthScheme
from franca.core.ids import (
    ANTHROPIC,
    ANTHROPIC_MESSAGES,
    CHAT,
    DEEPSEEK,
    GOOGLE,
    GOOGLE_GENERATE_CONTENT,
    OPENAI,
    OPENAI_CHAT,
    XAI,
)

ANTHROPIC_MESSAGES_ENDPOINT: Final = Endpoint(
    id="anthropic/chat/messages",
    provider=ANTHROPIC,
    capability=CHAT,
    dialect=ANTHROPIC_MESSAGES,
    base_url="https://api.anthropic.com",
    path="/v1/messages",
    auth=AuthScheme.x_api_key,
    extra_headers={"anthropic-version": "2023-06-01"},
    features=EndpointFeatures(streaming=True, caching=True, production=True),
)
"""Anthropic's Messages API: `POST https://api.anthropic.com/v1/messages`, `x-api-key`.

The one endpoint M0 can actually call. `AnthropicMessagesAdapter` speaks its dialect,
and the row's `path` is why that adapter leaves `WireRequest.path` at `None`.
"""

OPENAI_CHAT_ENDPOINT: Final = Endpoint(
    id="openai/chat/completions",
    provider=OPENAI,
    capability=CHAT,
    dialect=OPENAI_CHAT,
    base_url="https://api.openai.com",
    path="/v1/chat/completions",
    auth=AuthScheme.bearer,
    features=EndpointFeatures(streaming=True, caching=True, production=True),
)
"""OpenAI's Chat Completions API: `POST https://api.openai.com/v1/chat/completions`.

`auth` is `bearer` rather than `x-api-key`, which is the whole reason `AuthScheme`
is an enum on the row: the same `Connector` serves both providers and branches on
data, not on the provider's name.

`extra_headers` is empty on purpose. Unlike Anthropic, this wire carries no required
version header -- `OpenAI-Organization` and `OpenAI-Project` are optional scoping
headers, so they belong to whoever constructs the `Connector`, not to the row.

The Responses API (`/v1/responses`) is a *different dialect*, not a variant of this
one: it renames the prompt field, the output field and every usage counter. It gets
its own row and its own adapter when it lands.
"""

GOOGLE_GENERATE_CONTENT_ENDPOINT: Final = Endpoint(
    id="google/chat/generate_content",
    provider=GOOGLE,
    capability=CHAT,
    dialect=GOOGLE_GENERATE_CONTENT,
    base_url="https://generativelanguage.googleapis.com",
    path="/v1beta/models",
    auth=AuthScheme.x_goog_api_key,
    features=EndpointFeatures(streaming=True, caching=True, production=True),
)
"""Gemini's REST surface: `POST /v1beta/models/{model}:generateContent`, `x-goog-api-key`.

The only row whose `path` is not the path a request is sent to. This wire carries the
model *in* the path, so `GoogleGenerateContentAdapter` sets `WireRequest.path` on every
call and this value is never used for a completion; it names the collection those paths
are built under, which is where a models listing would go. A row cannot do better,
because a row cannot know the model.

`auth` is the third scheme, which is the case that makes the point of keeping auth on
the row: three providers, three header names, one `Connector` that branches on data.
"""

XAI_CHAT_ENDPOINT: Final = Endpoint(
    id="xai/chat/completions",
    provider=XAI,
    capability=CHAT,
    dialect=OPENAI_CHAT,
    base_url="https://api.x.ai",
    path="/v1/chat/completions",
    auth=AuthScheme.bearer,
    features=EndpointFeatures(streaming=True, production=True),
)
"""xAI's chat surface: `POST https://api.x.ai/v1/chat/completions`, an OpenAI-compatible wire.

The row that shows the three axes really are independent. `provider` is `xai` -- that is
whose key travels and whose bill it lands on -- while `dialect` is `openai_chat`, because
the bytes are OpenAI's. `OpenAIChatAdapter` serves this row unmodified, so xAI cost one
row of data and no code at all, which is the whole claim the README makes about
compatibility surfaces.

One measured divergence is recorded here rather than fixed in the adapter. Sent a bad key
on 2026-09-16, xAI answers **400**, not 401:

    {"code": "invalid-argument", "error": "Incorrect API key provided..."}

so franca reports a wrong xAI key as a non-retryable `provider` failure rather than
`auth`. Refining that would mean matching on the message text, since this body carries no
machine-readable reason the way Google's does, and a guess about wording is not worth the
false positives. The bare-string `error` is already handled: the connector reads both that
and the nested object form.

`caching` is left unverified rather than guessed. The surface reports cached prompt
tokens, but whether it does so under the `prompt_tokens_details.cached_tokens` key that
`OpenAIChatAdapter` reads has not been measured, and claiming a meter franca might not be
reading would be worse than admitting nobody checked.
"""

DEEPSEEK_CHAT_ENDPOINT: Final = Endpoint(
    id="deepseek/chat/completions",
    provider=DEEPSEEK,
    capability=CHAT,
    dialect=OPENAI_CHAT,
    base_url="https://api.deepseek.com",
    path="/v1/chat/completions",
    auth=AuthScheme.bearer,
    features=EndpointFeatures(streaming=True, production=True),
)
"""DeepSeek's chat surface: `POST https://api.deepseek.com/v1/chat/completions`, bearer.

The second compatibility surface, and a row for the same reason xAI is one. Unlike xAI it
answers a bad key with a plain **401** (measured 2026-09-16), so the connector's status
map already classifies it correctly and nothing needs refining.

Two things about this provider are known and deliberately *not* encoded as capability
claims. Its reasoning models return their chain of thought in `reasoning_content`
alongside `content`, which the M0 `OpenAIChatAdapter` does not read -- that arrives with
the full IR, as an `Item(kind="reasoning")`. And its prompt cache reports hits under
`prompt_cache_hit_tokens` rather than the nested `prompt_tokens_details.cached_tokens`
this adapter reads, so `caching` stays unverified: a `True` here would promise a meter
that comes back zero.
"""
