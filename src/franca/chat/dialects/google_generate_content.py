"""The Google `generateContent` wire: a `PromptPackage` onto Gemini's REST surface, and back.

The same M0 subset as the other two adapters -- text turns, an output budget, the system
prompt, streaming, and the token meters coming back -- so the three are directly
comparable. What makes this a third *dialect* rather than a flag on either of them is
that almost nothing about the envelope is shared.

**The model is in the path, never in the body.** `POST /v1beta/models/{model}:generateContent`,
and a top-level `model` key in the body is rejected. This is the only built-in adapter
that sets `WireRequest.path`, which is precisely why `path` exists on that type: the
endpoint row cannot know the model.

**Streaming swaps the method, it does not add a key.** There is no `stream: true` here.
The path becomes `:streamGenerateContent?alt=sse`, and without `alt=sse` the response is
a JSON array rather than server-sent events -- so the query parameter is load-bearing,
not decoration.

**Every key is camelCase and the roles are Google's own.** `systemInstruction`,
`generationConfig.maxOutputTokens`, `usageMetadata.promptTokenCount`; the assistant is
called `model`. A snake_case key sent here is not a synonym, it is an unknown field.

**The token meters overlap rather than partition.** `promptTokenCount` is the whole
prompt *including* whatever `cachedContentTokenCount` reports, while Anthropic's
`input_tokens` and `cache_read_input_tokens` are disjoint. Each counter is mapped to the
meter of the same name and nothing is reconciled: subtracting one from the other would
be a claim about which convention `Usage` means, and no measurement here supports one.
The divergence is recorded rather than smoothed over.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final
from urllib.parse import quote

from franca.chat.ir import Item, ModelResponse, PromptPackage
from franca.chat.profile import ChatProfile
from franca.core.adapter import Adapter, WireRequest
from franca.core.errors import ModelError
from franca.core.ids import CHAT, GOOGLE_GENERATE_CONTENT, Provider
from franca.core.profile import BaseProfile
from franca.core.types import Usage

DEFAULT_MAX_TOKENS: Final = 1024
"""The output budget sent when neither the package nor the profile names one.

`maxOutputTokens` is optional on this wire, and omitting it lets a thinking model spend
its own default budget on reasoning before it writes a word. A small explicit ceiling is
the safer library default, as it is on the OpenAI wire.
"""

_MODEL_PREFIX: Final = "models/"
"""Gemini names a model either bare or fully qualified; the path carries the prefix itself."""

_WIRE_ROLE: Final[Mapping[str, str]] = {
    "user": "user",
    "assistant": "model",
    "model": "model",
}
"""IR roles this wire accepts as `contents[].role`; Google spells the assistant `model`."""

_API_KEY_REASONS: Final = frozenset({"API_KEY_INVALID", "ACCESS_TOKEN_TYPE_UNSUPPORTED"})
"""`google.rpc.ErrorInfo` reasons that mean the credential is wrong, whatever the status."""


def _max_tokens(pkg: PromptPackage, profile: BaseProfile) -> int:
    """Resolve the output budget: the package's, then the profile's, then the floor.

    Args:
        pkg: The request being rendered.
        profile: What franca believes about the model.

    Returns:
        The `generationConfig.maxOutputTokens` value to put on the wire.
    """
    fallback = profile.default_max_output_tokens if isinstance(profile, ChatProfile) else None
    return pkg.max_output_tokens or fallback or DEFAULT_MAX_TOKENS


def path_for(model: str, *, stream: bool) -> str:
    """Build the request path for one model, since this wire carries it there.

    A fully qualified `models/gemini-...` is accepted alongside the bare name: the
    prefix is stripped so it is never doubled. What remains is percent-encoded with
    `/` left intact, because a tuned model is legitimately `tunedModels/{id}` while a
    `?` or a `#` in a model name would otherwise rewrite the query string.

    Args:
        model: The model identifier as the caller spelled it.
        stream: Whether to address the streaming method.

    Returns:
        The path, from the root, including `?alt=sse` when streaming.
    """
    name = quote(model.removeprefix(_MODEL_PREFIX), safe="/")
    method = "streamGenerateContent?alt=sse" if stream else "generateContent"
    return f"/v1beta/models/{name}:{method}"


def _contents(pkg: PromptPackage, provider: Provider) -> list[dict[str, Any]]:
    """Render the IR items as `contents`, merging same-role runs into one entry.

    Gemini expects alternating roles and takes several `parts` per entry, so adjacent
    items of the same wire role become one entry with several parts -- the same shape
    the Anthropic adapter produces, for the same reason.

    Args:
        pkg: The request whose `items` are being rendered.
        provider: The provider to name on a `ModelError`.

    Returns:
        The `contents` array, oldest turn first.

    Raises:
        ModelError: If an item is outside this M0 subset -- any kind other than
            `text`, or a role with no wire spelling. Raising rather than skipping is
            deliberate: a silently dropped turn changes the prompt, and a changed
            prompt is a measurement bug that only ever shows up as a worse answer.
    """
    contents: list[dict[str, Any]] = []
    for index, item in enumerate(pkg.items):
        role = _WIRE_ROLE.get(item.role)
        if role is None or item.kind != "text":
            raise ModelError(
                f"google_generate_content cannot send items[{index}]:"
                f" role={item.role!r} kind={item.kind!r};"
                " this adapter carries text items of role user, assistant or model."
                " Tool results, images and reasoning blocks land with the full IR",
                status=None,
                provider=provider,
                retryable=False,
                failure_class="unsupported",
            )
        part = {"text": item.text or ""}
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append(part)
        else:
            contents.append({"role": role, "parts": [part]})
    return contents


def _text_parts(raw: Mapping[str, Any]) -> tuple[str, ...]:
    """Pull the answer's prose out of the first candidate's parts.

    A part counts only when its `text` is a string and it is not flagged `thought`.
    Gemini returns its reasoning as text parts marked that way, and folding those into
    the answer would put the model's private deliberation into `ModelResponse.text` --
    the same mistranslation the other two adapters avoid by passing over `thinking`
    and `tool_use` blocks. A `functionCall` part is skipped for that reason too, and
    every skipped part survives on `raw`.

    Args:
        raw: The decoded JSON body of the 200.

    Returns:
        The text of each qualifying part, in wire order.
    """
    candidates: Sequence[Any] = raw.get("candidates") or ()
    if not candidates:
        return ()
    content: Mapping[str, Any] = candidates[0].get("content") or {}
    parts: Sequence[Any] = content.get("parts") or ()
    return tuple(
        part["text"]
        for part in parts
        if isinstance(part, Mapping)
        and isinstance(part.get("text"), str)
        and part.get("thought") is not True
    )


def _finish_reason(raw: Mapping[str, Any]) -> str | None:
    """Read `candidates[0].finishReason`, or `None` when there is no candidate."""
    candidates: Sequence[Any] = raw.get("candidates") or ()
    if not candidates:
        return None
    reason: object = candidates[0].get("finishReason")
    return reason if isinstance(reason, str) else None


def _has_api_key_reason(raw: Mapping[str, Any] | None) -> bool:
    """Whether Google's error details blame the credential rather than the request.

    Reads the structured `google.rpc.ErrorInfo` reason rather than matching on the
    message text, so the check is a field lookup and not a guess about wording that a
    localisation or a rewrite could invalidate.

    Args:
        raw: The decoded error body, if it decoded at all.

    Returns:
        Whether any detail carries a reason that means the key is the problem.
    """
    if raw is None:
        return False
    error: object = raw.get("error")
    if not isinstance(error, Mapping):
        return False
    details: object = error.get("details")
    if not isinstance(details, Sequence) or isinstance(details, str | bytes):
        return False
    return any(
        isinstance(detail, Mapping) and detail.get("reason") in _API_KEY_REASONS
        for detail in details
    )


class GoogleGenerateContentAdapter(Adapter[PromptPackage, ModelResponse]):
    """Translate the chat IR into Gemini's `generateContent` wire, and its reply back.

    Stateless and cheap to construct, as the `Adapter` contract requires.

    What this adapter renders today is text turns, the output budget and the system
    instruction; what it reads back is the first candidate's text parts, the three
    token meters this wire reports, `finishReason` and `modelVersion`. Tools, thinking
    budgets, thought signatures, images and structured output are all part of the
    dialect and none of them is here yet; so is the Interactions surface, which is a
    separate dialect rather than a variant of this one.
    """

    capability = CHAT
    dialect = GOOGLE_GENERATE_CONTENT
    status = "stable"

    def to_request(
        self,
        req: PromptPackage,
        model: str,
        profile: BaseProfile,
        *,
        stream: bool = False,
    ) -> WireRequest:
        """Render a package as one `POST` to a model's `generateContent` method.

        Unlike the other two adapters this one *sets* `path`, because the model lives
        there. It also emits no streaming key: the method name in the path is what
        makes the call a stream, and `WireRequest.stream` tells the connector to read
        the reply as events.

        `provider_options[google_generate_content]` is merged last and validated
        after, so a caller's override wins over anything computed here. The keys it
        carries are wire keys, which on this dialect means camelCase.

        Args:
            req: The package to render.
            model: The model identifier to address, as the caller spelled it.
            profile: What franca believes about that model.
            stream: Whether to shape the request for server-sent events.

        Returns:
            The request for the `Connector` to send, carrying its own path.

        Raises:
            ModelError: If an item is outside what this subset can express.
        """
        body: dict[str, Any] = {
            "contents": _contents(req, profile.provider),
            "generationConfig": {"maxOutputTokens": _max_tokens(req, profile)},
        }
        system = req.system_text()
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        body.update(req.provider_options.get(GOOGLE_GENERATE_CONTENT, {}))
        return WireRequest(body=body, stream=stream, path=path_for(model, stream=stream))

    def from_response(
        self,
        raw: Mapping[str, Any],
        req: PromptPackage,
        model: str,
        provider: Provider,
    ) -> ModelResponse:
        """Parse a `generateContent` 200 body into a `ModelResponse`.

        Only the first candidate is read: `candidateCount` is not in the M0 IR, so a
        reply carrying more keeps the rest on `raw` rather than being flattened into
        something the IR cannot represent faithfully. Each qualifying part becomes its
        own item, which is how a multi-part answer stays a transcript rather than a
        concatenation.

        `served_model` comes from `modelVersion`, the field this wire uses to say which
        snapshot actually answered. It is absent from some compatibility surfaces, and
        `None` is then the honest answer.

        `trace` is left `None`: the leaf stamps it when the call is over.

        Args:
            raw: The decoded JSON body of the 200.
            req: The package that produced it; unread, since the reply is self-contained.
            model: The model identifier that was addressed.
            provider: The provider that answered.

        Returns:
            The parsed response, with `trace` still `None`.
        """
        usage: Mapping[str, Any] = raw.get("usageMetadata") or {}
        served: object = raw.get("modelVersion")
        return ModelResponse(
            model=model,
            provider=provider,
            dialect=GOOGLE_GENERATE_CONTENT,
            items=tuple(Item(role="model", kind="text", text=text) for text in _text_parts(raw)),
            stop_reason=_finish_reason(raw),
            usage=Usage(
                input_tokens=usage.get("promptTokenCount") or 0,
                output_tokens=usage.get("candidatesTokenCount") or 0,
                cache_read_tokens=usage.get("cachedContentTokenCount") or 0,
            ),
            served_model=served if isinstance(served, str) else None,
            raw=dict(raw),
        )

    def error_from(
        self,
        status: int,
        raw: Mapping[str, Any] | None,
        provider: Provider,
    ) -> ModelError | None:
        """Reclassify a rejected credential that this wire reports as a bad request.

        Google answers an invalid API key with **400 INVALID_ARGUMENT**, not 401.
        Captured by sending a deliberately wrong key to
        `generativelanguage.googleapis.com` on 2026-09-16:

            {"error": {"code": 400, "message": "API key not valid...",
                       "status": "INVALID_ARGUMENT",
                       "details": [{"@type": ".../google.rpc.ErrorInfo",
                                    "reason": "API_KEY_INVALID"}]}}

        The connector's status map has no way to see that: a 400 is an ordinary,
        non-retryable provider error, and left alone it would tell a caller their
        request was malformed when their key is the problem. Since everything above
        the connector reasons in `failure_class`, that misdirection would reach every
        log line and every dashboard. The reason code makes it a field lookup rather
        than a guess, so the refinement is safe to make.

        A 401 or 403 is already `auth` and is left alone; every other status belongs
        to the connector's map, which handles it.

        Args:
            status: The HTTP status that came back.
            raw: The decoded error body, if it decoded at all.
            provider: The provider that answered.

        Returns:
            A replacement `auth` error when the body blames the credential, else `None`.
        """
        if status in {401, 403} or not _has_api_key_reason(raw):
            return None
        return ModelError(
            "Google rejected the API key; the request itself was not the problem."
            " Check the key named by GEMINI_API_KEY",
            status=status,
            provider=provider,
            retryable=False,
            failure_class="auth",
            raw=raw,
        )
