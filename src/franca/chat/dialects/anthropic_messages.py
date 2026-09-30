"""The Anthropic Messages wire: `PromptPackage` onto `POST /v1/messages`, and back.

This is the M0 subset of the dialect -- text turns, `max_tokens`, the top-level
`system` string, `stream`, and the four token meters coming back -- and it is a
subset on purpose: everything here is behaviour that a walking skeleton exercises end
to end, so the adapter seam, the connector and the trace stamp are all provable
before tools, thinking, images and caching arrive.

Three things about the wire drive the shape of the code below.

`max_tokens` is **required** by the Messages API: a request without it is a 400, so
this adapter always emits one. It is an output *budget*, never a sampling parameter,
which is why it is not gated on `sampling_allowed` -- a model that ignores
`temperature` still honours a token ceiling.

Anthropic merges nothing for you: two consecutive `user` messages are an error, and
the wire expects one message per role run, carrying a list of content blocks. So
adjacent IR items of the same wire role become one message with several blocks, and
the message is what the *wire* considers a turn rather than what the caller typed.

`provider_options[ANTHROPIC_MESSAGES]` is merged **last** and validated after, so a
caller can override any key this adapter computed. That ordering is the escape hatch:
a wire feature franca has no field for is one dict away, and it never has to fight
what the adapter decided.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final

from franca.chat.ir import Item, ModelResponse, PromptPackage
from franca.chat.profile import ChatProfile
from franca.core.adapter import Adapter, WireRequest
from franca.core.errors import ModelError
from franca.core.ids import ANTHROPIC_MESSAGES, CHAT, Provider
from franca.core.profile import BaseProfile
from franca.core.types import Usage

DEFAULT_MAX_TOKENS: Final = 1024
"""The `max_tokens` floor: what is sent when neither the package nor the profile says.

The Messages API rejects a request without `max_tokens`, so *some* number is always
required and the adapter cannot decline to choose. This one is deliberately small:
the failure mode of guessing high is a bill, the failure mode of guessing low is a
visibly truncated answer that says "set `max_output_tokens`".
"""

_WIRE_ROLE: Final[Mapping[str, str]] = {
    "user": "user",
    "assistant": "assistant",
    "model": "assistant",
}
"""IR roles the Messages wire accepts as message roles; `model` is Google's spelling."""


def _max_tokens(pkg: PromptPackage, profile: BaseProfile) -> int:
    """Resolve the output budget: the package's, then the profile's, then the floor.

    The chain is `or`-shaped rather than `is None`-shaped, so a `0` from either source
    falls through to the next one. Nobody means "generate at most zero tokens", and
    the wire would reject it.

    The profile is typed as `BaseProfile` by the adapter contract, so the chat-specific
    default is read behind an `isinstance` narrowing; a non-chat profile simply
    contributes nothing.

    Args:
        pkg: The request being rendered.
        profile: What franca believes about the model.

    Returns:
        The `max_tokens` value to put on the wire.
    """
    fallback = profile.default_max_output_tokens if isinstance(profile, ChatProfile) else None
    return pkg.max_output_tokens or fallback or DEFAULT_MAX_TOKENS


def _messages(pkg: PromptPackage, provider: Provider) -> list[dict[str, Any]]:
    """Render the IR items as Messages turns, merging same-role runs into one message.

    Every message carries a list of content blocks, even when the run is one item
    long: the wire accepts a bare string too, but one shape means one code path and a
    body that stays comparable across cassettes.

    Args:
        pkg: The request whose `items` are being rendered.
        provider: The provider to name on a `ModelError`.

    Returns:
        The `messages` array, oldest turn first.

    Raises:
        ModelError: If an item is not something this M0 subset can put in a message:
            a `system`, `developer` or `tool` role, or any kind other than `text`.
            Raising rather than skipping is the whole point -- dropping a turn would
            change the prompt silently, and a truncated prompt is a measurement bug
            that only shows up as a worse answer.
    """
    messages: list[dict[str, Any]] = []
    for index, item in enumerate(pkg.items):
        role = _WIRE_ROLE.get(item.role)
        if role is None or item.kind != "text":
            raise ModelError(
                f"anthropic_messages cannot send items[{index}]:"
                f" role={item.role!r} kind={item.kind!r};"
                " this adapter carries text items of role user, assistant or model."
                " Tool results, images and reasoning blocks land with the full IR",
                status=None,
                provider=provider,
                retryable=False,
                failure_class="unsupported",
            )
        block = {"type": "text", "text": item.text or ""}
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].append(block)
        else:
            messages.append({"role": role, "content": [block]})
    return messages


class AnthropicMessagesAdapter(Adapter[PromptPackage, ModelResponse]):
    """Translate the chat IR into Anthropic's Messages wire, and its reply back.

    Stateless and cheap to construct, as the `Adapter` contract requires: the registry
    keeps one instance and shares it across every call.

    What this adapter renders today is text turns, the output budget, the system
    string and the stream flag; what it reads back is text blocks, the four token
    meters, `stop_reason` and `served_model`. Tools, thinking, images, cache control,
    structured output and refusal parsing are all part of the dialect and none of them
    is here yet -- they arrive with the full IR, together with their profile gates.
    """

    capability = CHAT
    dialect = ANTHROPIC_MESSAGES
    status = "stable"

    def to_request(
        self,
        req: PromptPackage,
        model: str,
        profile: BaseProfile,
        *,
        stream: bool = False,
    ) -> WireRequest:
        """Render a package as one `POST` to the Messages endpoint.

        `path` is left `None`: the endpoint row owns `/v1/messages`, and so does the
        `anthropic-version` header, so a version bump is a data change rather than a
        code change.

        The body is `model`, the resolved `max_tokens` and `messages`, plus `system`
        when there is any system text and `stream` only when streaming was asked for
        -- an absent key and a `false` are not the same thing to a cassette diff, and
        the smaller body is the honest one. `provider_options[anthropic_messages]` is
        then merged over all of it and the result validated, so a caller's override
        wins over anything computed here.

        Nothing in this subset is profile-gated, so `WireRequest.unverified` stays
        empty: `max_tokens` is a budget and is emitted whatever `sampling_allowed`
        says.

        Args:
            req: The package to render.
            model: The model identifier to put on the wire, as the caller spelled it.
            profile: What franca believes about that model.
            stream: Whether to shape the request for server-sent events.

        Returns:
            The request for the `Connector` to send.

        Raises:
            ModelError: If an item is outside what this subset can express; see
                `_messages`.
        """
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": _max_tokens(req, profile),
            "messages": _messages(req, profile.provider),
        }
        system = req.system_text()
        if system:
            body["system"] = system
        if stream:
            body["stream"] = True
        body.update(req.provider_options.get(ANTHROPIC_MESSAGES, {}))
        return WireRequest(body=body, stream=stream)

    def from_response(
        self,
        raw: Mapping[str, Any],
        req: PromptPackage,
        model: str,
        provider: Provider,
    ) -> ModelResponse:
        """Parse a Messages 200 body into a `ModelResponse`.

        Only `text` blocks become items in this subset; a `thinking` or `tool_use`
        block is passed over rather than mistranslated, and survives on `raw` until
        the full IR knows what to do with it. Every usage counter defaults to `0`,
        which covers both an absent key and an explicit `null`, so a reply that
        reports nothing still yields a usable record.

        `trace` is left `None` deliberately. The leaf stamps it once the call is over,
        because latency, attempts and the selection route are facts about the call and
        not about the body -- an adapter that guessed at them would be overwritten.

        Args:
            raw: The decoded JSON body of the 200.
            req: The package that produced it; unread here, since Anthropic answers
                with a self-contained message.
            model: The model identifier that was sent.
            provider: The provider that answered.

        Returns:
            The parsed response, with `trace` still `None`.
        """
        content: Sequence[Any] = raw.get("content") or ()
        usage: Mapping[str, Any] = raw.get("usage") or {}
        return ModelResponse(
            model=model,
            provider=provider,
            dialect=ANTHROPIC_MESSAGES,
            items=tuple(
                Item(role="assistant", kind="text", text=block.get("text"))
                for block in content
                if block.get("type") == "text"
            ),
            stop_reason=raw.get("stop_reason"),
            usage=Usage(
                input_tokens=usage.get("input_tokens") or 0,
                output_tokens=usage.get("output_tokens") or 0,
                cache_read_tokens=usage.get("cache_read_input_tokens") or 0,
                cache_write_tokens=usage.get("cache_creation_input_tokens") or 0,
            ),
            served_model=raw.get("model"),
            raw=dict(raw),
        )
