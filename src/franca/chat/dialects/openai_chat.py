"""The OpenAI Chat Completions wire: `PromptPackage` onto `POST /v1/chat/completions`.

The same M0 subset as the Anthropic adapter -- text turns, an output budget, the
system prompt, `stream`, and the token meters coming back -- which makes the two
directly comparable and is the point of having an IR at all.

Four differences from the Anthropic wire drive the code below, and each one is a
reason this is a separate *dialect* rather than a flag on one adapter.

**The system prompt is a message, not a field.** Anthropic takes a top-level `system`
string; OpenAI takes a `system` entry at the head of `messages`. Same IR, two shapes.

**Adjacent same-role turns are legal.** Anthropic rejects two consecutive `user`
messages, so its adapter merges runs; OpenAI accepts them, so this one does not merge
and the wire stays a faithful transcript of what the caller wrote.

**The output budget is `max_completion_tokens`.** `max_tokens` is the legacy spelling
and is rejected outright by the reasoning models, so this adapter always emits the
current name. A caller who needs the legacy key for an old deployment can put it in
`provider_options[openai_chat]`, which is merged last and wins.

**A 429 means two unrelated things.** Rate limiting is retryable; a spent balance is
not, and both arrive as HTTP 429. The status alone is therefore not enough to decide
retryability, which is exactly what `error_from` exists to refine -- see its docstring
for the captured evidence.

Structured output is where this wire is the most complete of the three: all three
strategies have a field. `native` is `response_format` of type `json_schema`, which
names the schema and carries `strict`; `json_mode` is `response_format` of type
`json_object`, which sends no schema; `tool_calling` is one function forced through
`tool_choice`, whose arguments come back as a JSON-encoded *string*, not an object.
JSON mode also needs the word "JSON" somewhere in the messages, and the provider, not
franca, enforces that -- this adapter does not edit the prompt to satisfy it.
"""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final

from franca.chat.dialects.common import (
    merge_provider_options,
    pick_strategy,
    read_structured,
    schema_name,
)
from franca.chat.ir import Item, ModelResponse, OutputFormat, PromptPackage
from franca.chat.profile import ChatProfile
from franca.core.adapter import Adapter, WireRequest
from franca.core.enums import Strategy
from franca.core.errors import ModelError
from franca.core.ids import CHAT, OPENAI_CHAT, Provider
from franca.core.profile import BaseProfile
from franca.core.types import Usage

DEFAULT_MAX_TOKENS: Final = 1024
"""The output budget sent when neither the package nor the profile names one.

Unlike Anthropic's `max_tokens`, this key is optional on the wire: omitting it lets
the model run to its own default, which on a reasoning model can be very expensive.
Sending a small explicit ceiling is the safer default for a library.
"""

_WIRE_ROLE: Final[Mapping[str, str]] = {
    "system": "system",
    "developer": "developer",
    "user": "user",
    "assistant": "assistant",
    "model": "assistant",
}
"""IR roles the Chat Completions wire accepts; `model` is Google's spelling for assistant."""

_NOT_RETRYABLE_429_TYPES: Final = frozenset({"insufficient_quota"})
"""429 error types that no amount of waiting will clear."""

_NOT_RETRYABLE_429_CODES: Final = frozenset(
    {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached"}
)
"""429 error codes that mean "add money", not "slow down"."""


def _max_tokens(pkg: PromptPackage, profile: BaseProfile) -> int:
    """Resolve the output budget: the package's, then the profile's, then the floor.

    Args:
        pkg: The request being rendered.
        profile: What franca believes about the model.

    Returns:
        The `max_completion_tokens` value to put on the wire.
    """
    fallback = profile.default_max_output_tokens if isinstance(profile, ChatProfile) else None
    return pkg.max_output_tokens or fallback or DEFAULT_MAX_TOKENS


def _messages(pkg: PromptPackage, provider: Provider) -> list[dict[str, Any]]:
    """Render the system blocks and IR items as one flat `messages` array.

    System text goes first, as a single `system` message, because that is where this
    wire carries it. Turns are then emitted in order with no merging: OpenAI accepts
    consecutive same-role messages, so preserving them keeps the wire an honest
    transcript.

    Args:
        pkg: The request whose `system` and `items` are being rendered.
        provider: The provider to name on a `ModelError`.

    Returns:
        The `messages` array, oldest turn first.

    Raises:
        ModelError: If an item is outside this M0 subset -- any kind other than
            `text`, or a role with no wire spelling. Raising rather than skipping is
            deliberate: a silently dropped turn changes the prompt, and a changed
            prompt is a measurement bug that only ever shows up as a worse answer.
    """
    messages: list[dict[str, Any]] = []
    system = pkg.system_text()
    if system:
        messages.append({"role": "system", "content": system})
    for index, item in enumerate(pkg.items):
        role = _WIRE_ROLE.get(item.role)
        if role is None or item.kind != "text":
            raise ModelError(
                f"openai_chat cannot send items[{index}]:"
                f" role={item.role!r} kind={item.kind!r};"
                " this adapter carries text items of role system, developer, user,"
                " assistant or model. Tool results, images and reasoning blocks land"
                " with the full IR",
                status=None,
                provider=provider,
                retryable=False,
                failure_class="unsupported",
            )
        messages.append({"role": role, "content": item.text or ""})
    return messages


def _structured(fmt: OutputFormat, strategy: Strategy) -> dict[str, Any]:
    """Render the chosen structured-output strategy as top-level body keys.

    `strict` is sent only when the caller set it, in either direction: the wire's own
    default applies otherwise, and a strict schema here must meet stricter rules
    (every property required, no additional properties) that franca does not impose
    on a caller who did not ask for them.

    Args:
        fmt: The package's output format, already checked by `pick_strategy`.
        strategy: Any of the three; this wire supports them all.

    Returns:
        The keys to add to the body.
    """
    if strategy is Strategy.json_mode:
        return {"response_format": {"type": "json_object"}}
    name = schema_name(fmt)
    strict = {} if fmt.strict is None else {"strict": fmt.strict}
    if strategy is Strategy.native:
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": fmt.json_schema, **strict},
            }
        }
    return {
        "tools": [
            {
                "type": "function",
                "function": {"name": name, "parameters": fmt.json_schema, **strict},
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": name}},
    }


def _tool_calls(message: Mapping[str, Any]) -> list[tuple[object, object]]:
    """Read a message's tool calls as `(name, arguments)` pairs, in wire order.

    `arguments` stays the JSON-encoded string the wire sends; `read_structured`
    decodes it. Anything not shaped like a function call is skipped, since a
    malformed entry cannot be the forced tool's answer.
    """
    calls: Sequence[Any] = message.get("tool_calls") or ()
    pairs: list[tuple[object, object]] = []
    for call in calls:
        function: object = call.get("function") if isinstance(call, Mapping) else None
        if isinstance(function, Mapping):
            pairs.append((function.get("name"), function.get("arguments")))
    return pairs


def _error_field(raw: Mapping[str, Any] | None, name: str) -> str | None:
    """Read one string field out of OpenAI's `{"error": {...}}` envelope."""
    if raw is None:
        return None
    error: object = raw.get("error")
    if not isinstance(error, Mapping):
        return None
    value: object = error.get(name)
    return value if isinstance(value, str) else None


class OpenAIChatAdapter(Adapter[PromptPackage, ModelResponse]):
    """Translate the chat IR into OpenAI's Chat Completions wire, and its reply back.

    Stateless and cheap to construct, as the `Adapter` contract requires.

    What this adapter renders today is text turns, the output budget, the system
    message, the stream flag and structured output; what it reads back is the
    assistant message, the token meters, `finish_reason`, the model the wire says
    served the request and the structured answer. Tools, images, reasoning effort and
    the Responses API surface are all deferred -- the last of those is a separate
    dialect, not a flag here.

    Attributes:
        supports: The structured-output strategies this wire can render.
    """

    capability = CHAT
    dialect = OPENAI_CHAT
    status = "stable"
    supports: ClassVar[frozenset[Strategy]] = frozenset(
        {Strategy.native, Strategy.tool_calling, Strategy.json_mode}
    )

    def to_request(
        self,
        req: PromptPackage,
        model: str,
        profile: BaseProfile,
        *,
        stream: bool = False,
    ) -> WireRequest:
        """Render a package as one `POST` to the Chat Completions endpoint.

        `path` is left `None`: the endpoint row owns `/v1/chat/completions`.

        `stream` is emitted only when streaming was asked for -- an absent key and a
        `false` differ in a cassette diff, and the smaller body is the honest one.
        The structured-output keys are added when a JSON answer was asked for; see
        `pick_strategy` for what that adds to `WireRequest.unverified`.
        `provider_options[openai_chat]` is merged last and deeply, so a caller's
        override wins over anything computed here.

        Args:
            req: The package to render.
            model: The model identifier to put on the wire, as the caller spelled it.
            profile: What franca believes about that model.
            stream: Whether to shape the request for server-sent events.

        Returns:
            The request for the `Connector` to send.

        Raises:
            ModelError: If an item is outside what this subset can express, or the
                output format cannot be rendered on this wire.
        """
        unverified: set[str] = set()
        body: dict[str, Any] = {
            "model": model,
            "messages": _messages(req, profile.provider),
            "max_completion_tokens": _max_tokens(req, profile),
        }
        strategy = pick_strategy(req.output_format, profile, self.supports, OPENAI_CHAT, unverified)
        if strategy is not None:
            body.update(_structured(req.output_format, strategy))
        if stream:
            body["stream"] = True
        body = merge_provider_options(body, req.provider_options.get(OPENAI_CHAT, {}))
        return WireRequest(body=body, stream=stream, unverified=frozenset(unverified))

    def from_response(
        self,
        raw: Mapping[str, Any],
        req: PromptPackage,
        model: str,
        provider: Provider,
    ) -> ModelResponse:
        """Parse a Chat Completions 200 body into a `ModelResponse`.

        Only the first choice is read: `n` is not in the M0 IR, so a reply with more
        than one choice keeps the rest on `raw` rather than being flattened into
        something the IR cannot represent faithfully.

        The usage names differ from every other dialect -- `prompt_tokens` and
        `completion_tokens` rather than `input_tokens` and `output_tokens`, with the
        cache counter nested under `prompt_tokens_details` -- which is precisely the
        divergence `Usage` exists to erase for everything upstream.

        When the package asked for a JSON answer, `structured` is the forced
        function's decoded `arguments` if the message carries that call, else the
        content parsed as JSON. A refusal arrives in `message.refusal` with `content`
        null, so it leaves `structured` at `None` and stays on `raw`.

        `trace` is left `None`: the leaf stamps it when the call is over.

        Args:
            raw: The decoded JSON body of the 200.
            req: The package that produced it, read only for its output format.
            model: The model identifier that was sent.
            provider: The provider that answered.

        Returns:
            The parsed response, with `trace` still `None`.
        """
        choices: Sequence[Any] = raw.get("choices") or ()
        first: Mapping[str, Any] = choices[0] if choices else {}
        message: Mapping[str, Any] = first.get("message") or {}
        text = message.get("content")

        usage: Mapping[str, Any] = raw.get("usage") or {}
        prompt_details: Mapping[str, Any] = usage.get("prompt_tokens_details") or {}

        return ModelResponse(
            model=model,
            provider=provider,
            dialect=OPENAI_CHAT,
            items=(
                (Item(role="assistant", kind="text", text=text),) if isinstance(text, str) else ()
            ),
            stop_reason=first.get("finish_reason"),
            usage=Usage(
                input_tokens=usage.get("prompt_tokens") or 0,
                output_tokens=usage.get("completion_tokens") or 0,
                cache_read_tokens=prompt_details.get("cached_tokens") or 0,
            ),
            served_model=raw.get("model"),
            structured=read_structured(
                req.output_format,
                text if isinstance(text, str) else "",
                _tool_calls(message),
            ),
            raw=dict(raw),
        )

    def error_from(
        self,
        status: int,
        raw: Mapping[str, Any] | None,
        provider: Provider,
    ) -> ModelError | None:
        """Refine a 429 that the status map cannot classify correctly on its own.

        The `Connector` maps every 429 to `rate_limit` with `retryable=True`, which is
        right for a rate limit and wrong for a spent balance. Both arrive as 429 and
        only the body separates them. Captured from the live API:

            {"error": {"type": "insufficient_quota",
                       "code": "credit_balance_exhausted",
                       "message": "You have no credits remaining..."}}

        Left as retryable, that sends `RetryClient` into its full backoff schedule
        against a wall, turning an instant, actionable failure into a slow one. So a
        quota 429 is re-raised as a non-retryable `auth` failure: the account, not the
        request, is what has to change.

        Every other status is left to the connector's map, which already handles it.

        Args:
            status: The HTTP status that came back.
            raw: The decoded error body, if it decoded at all.
            provider: The provider that answered.

        Returns:
            A replacement error for a quota 429, otherwise `None`.
        """
        if status != 429:
            return None
        type_ = _error_field(raw, "type")
        code = _error_field(raw, "code")
        if type_ not in _NOT_RETRYABLE_429_TYPES and code not in _NOT_RETRYABLE_429_CODES:
            return None
        return ModelError(
            "OpenAI reports no remaining quota for this account; retrying cannot clear"
            " it. Add credits or raise the billing limit",
            status=status,
            provider=provider,
            retryable=False,
            failure_class="auth",
            raw=raw,
        )
