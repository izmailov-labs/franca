"""Mechanics every chat dialect shares, which no single wire owns.

Three adapters render one `OutputFormat` into three unrelated wire shapes, but the
decisions around that rendering are the same on every wire: *which* strategy to use,
what to call the schema, and how to read a JSON answer back. Made three times, they
would drift apart, and the adapters would disagree about what one package means --
the one thing an IR exists to prevent. So they are made once, here, and each adapter
keeps only the part that is genuinely its own: where the bytes go.

`merge_provider_options` lives here for the same reason, and it merges *deeply*. The
escape hatch most often carries a key that sits inside an object the adapter also
writes: Gemini's `temperature` lives in `generationConfig` beside `maxOutputTokens` and
the response schema, and Anthropic's `effort` lives in `output_config` beside `format`.
A top-level replace would let an override of one silently delete the others, which is
exactly the failure this layer refuses everywhere else.
"""

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, Final

from franca.chat.ir import OutputFormat
from franca.chat.profile import ChatProfile
from franca.core.enums import Strategy
from franca.core.errors import ModelError
from franca.core.ids import Dialect
from franca.core.profile import BaseProfile

STRATEGY_ORDER: Final = (Strategy.native, Strategy.tool_calling, Strategy.json_mode)
"""The fallback order when neither the caller nor the profile picked a strategy.

Strongest guarantee first. A native schema field is enforced by the provider, a forced
tool call carries the schema as the tool's parameters, and JSON mode promises only that
the answer parses -- the schema is not sent at all.
"""

DEFAULT_SCHEMA_NAME: Final = "response"
"""The name a schema or a forced tool is sent under when its `title` cannot be."""

_PORTABLE_NAME: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")
"""A name every chat wire accepts for both a response schema and a tool.

The intersection of the three rules rather than any one of them: OpenAI and Anthropic
allow letters, digits, `_` and `-` up to 64 characters, and Gemini additionally wants
the first character to be a letter or an underscore. A pydantic model's schema carries
its class name as `title`, which passes, so a model class names its own answer.
"""


def _unsupported(message: str, profile: BaseProfile) -> ModelError:
    """Build the error for a request this wire cannot express, raised before any I/O."""
    return ModelError(
        message,
        status=None,
        provider=profile.provider,
        retryable=False,
        failure_class="unsupported",
    )


def _usable(strategy: Strategy, fmt: OutputFormat) -> bool:
    """Whether `strategy` can serve `fmt` at all, whatever the wire supports."""
    if strategy is Strategy.json_mode:
        return fmt.strict is not True
    return fmt.json_schema is not None


def pick_strategy(
    fmt: OutputFormat,
    profile: BaseProfile,
    supports: frozenset[Strategy],
    dialect: Dialect,
    unverified: set[str],
) -> Strategy | None:
    """Decide how this wire obtains the structured answer `fmt` asks for.

    An explicit `fmt.strategy` is honoured or refused, never substituted: a caller who
    named `tool_calling` and silently got JSON mode would be measuring something they
    did not ask for. Without one, the profile's `structured_output` is tried first --
    it is a measurement of what this model does well -- and then `STRATEGY_ORDER`,
    skipping anything this wire does not support or this format cannot use.

    Two profile facts are tri-state, and both follow the one rule every profile gate
    does. `structured_output=None` means nobody has measured how this model does
    structured output, so whatever is chosen is recorded as `"structured_output"` in
    `unverified`. `strict_enforced` matters only when the caller asked for
    `strict=True`: `False` is a measured "this model treats a strict schema as a
    hint", which cannot meet the request and raises; `None` is recorded as
    `"strict_enforced"`.

    Args:
        fmt: The package's output format.
        profile: What franca believes about the model; a non-chat profile claims
            nothing.
        supports: The strategies this adapter can render.
        dialect: The dialect asking, to name in an error.
        unverified: The adapter's set of assumed profile flags; added to in place.

    Returns:
        The strategy to render, or `None` when `fmt` asks for no JSON answer.

    Raises:
        ModelError: With `failure_class="unsupported"` when the explicit strategy is
            not one this wire supports, when no supported strategy can serve `fmt`,
            or when `strict=True` meets a profile that says strictness is not
            enforced. Raised before any I/O.
    """
    if fmt.kind != "json":
        return None
    chat = profile if isinstance(profile, ChatProfile) else None
    preferred = chat.structured_output if chat is not None else None
    offered = ", ".join(sorted(strategy.value for strategy in supports)) or "nothing"

    if fmt.strategy is not None:
        if fmt.strategy not in supports:
            raise _unsupported(
                f"{dialect} cannot use output_format.strategy={fmt.strategy.value};"
                f" this wire supports {offered}",
                profile,
            )
        chosen = fmt.strategy
    else:
        candidates = STRATEGY_ORDER if preferred is None else (preferred, *STRATEGY_ORDER)
        found = next((s for s in candidates if s in supports and _usable(s, fmt)), None)
        if found is None:
            if not supports:
                reason = "it cannot render structured output"
            elif fmt.json_schema is None:
                reason = "without output_format.json_schema only json_mode applies"
            else:
                reason = "output_format.strict=True rules out json_mode"
            raise _unsupported(f"{dialect} supports {offered}; {reason}", profile)
        chosen = found

    if preferred is None:
        unverified.add("structured_output")
    if fmt.strict is True:
        enforced = chat.strict_enforced if chat is not None else None
        if enforced is False:
            raise _unsupported(
                f"output_format.strict=True cannot be met: profile row {profile.name!r}"
                " says this model treats a strict schema as a hint",
                profile,
            )
        if enforced is None:
            unverified.add("strict_enforced")
    return chosen


def schema_name(fmt: OutputFormat) -> str:
    """Name the schema, or the forced tool, the same way on every wire.

    The schema's own `title` when it is a name every wire accepts, otherwise
    `DEFAULT_SCHEMA_NAME`. Deterministic in the format alone, which is what lets
    `read_structured` find the forced tool's call again from the request.

    Args:
        fmt: The package's output format.

    Returns:
        A name valid on every chat wire.
    """
    title: object = (fmt.json_schema or {}).get("title")
    if isinstance(title, str) and _PORTABLE_NAME.fullmatch(title):
        return title
    return DEFAULT_SCHEMA_NAME


def decode_object(text: str | None) -> dict[str, Any] | None:
    """Parse `text` as a JSON object, or say it is not one.

    Lenient on purpose: this reads a reply that has already arrived and been paid for,
    and an exception would throw the reply away with the parse. A truncated answer, a
    refusal written as prose, or JSON whose root is not an object all come back as
    `None`, and the response's `text` and `stop_reason` still say what happened.
    Markdown fences are not stripped -- every strategy asks the wire for bare JSON, so
    a fenced answer is the model not complying, and guessing it back into shape would
    hide that.

    Args:
        text: The candidate JSON.

    Returns:
        The decoded object, or `None`.
    """
    if text is None:
        return None
    try:
        value: object = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def read_structured(
    fmt: OutputFormat,
    text: str,
    calls: Iterable[tuple[object, object]],
) -> dict[str, Any] | None:
    """Recover the structured answer from a reply, wherever this wire put it.

    `from_response` is not told which strategy `to_request` picked -- that depended on
    the profile, which the response path never sees -- and it does not need to be. A
    call to the tool named `schema_name(fmt)` can only be the forced tool, because the
    adapters render no caller tools yet, so its arguments are the answer when one is
    present. Otherwise the answer is the reply's text. When caller tools land, a tool
    of the same name becomes possible and the request will need to carry the choice.

    Args:
        fmt: The output format of the package that produced the reply.
        text: The reply's prose, joined.
        calls: The reply's tool calls as `(name, arguments)` pairs, in wire order. The
            arguments may be an object already or a JSON-encoded string, as OpenAI
            sends them.

    Returns:
        The decoded answer, or `None` when `fmt` asked for none or the reply holds no
        JSON object.
    """
    if fmt.kind != "json":
        return None
    name = schema_name(fmt)
    for called, arguments in calls:
        if called != name:
            continue
        if isinstance(arguments, str):
            return decode_object(arguments)
        return arguments if isinstance(arguments, dict) else None
    return decode_object(text)


def merge_provider_options(body: Mapping[str, Any], options: Mapping[str, Any]) -> dict[str, Any]:
    """Merge a caller's `provider_options` over a computed body, deeply; theirs wins.

    Where both sides hold an object under the same key, the two are merged key by key,
    recursively; anything else -- a scalar, a list, a type mismatch -- is replaced by
    the caller's value. So `{"generationConfig": {"temperature": 0.2}}` adds a key
    beside the computed budget and schema instead of deleting them, while a caller who
    sets `max_tokens` still overrides the computed one.

    Neither argument is modified, which matters because `options` belongs to a frozen
    package: the next attempt of a retry must see what the caller wrote.

    Args:
        body: The body the adapter computed.
        options: The caller's overrides for this dialect, in wire case.

    Returns:
        A new merged body.
    """
    merged = dict(body)
    for key, value in options.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_provider_options(current, value)
        else:
            merged[key] = value
    return merged
