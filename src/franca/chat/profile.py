"""The chat capability's model profile: every belief an adapter is allowed to act on.

`ChatProfile` is `BaseProfile` plus the chat-specific facts. It is data, not code: a
new model is a row in a `ProfileTable`, never a branch in an adapter. That only holds
if every fact an adapter needs has a field here, so the class is wide on purpose and
each field answers exactly one question.

Tri-state throughout. A `bool | None` flag is `True` verified present, `False`
verified absent, `None` nobody checked -- so a fresh row's `sampling_allowed` is
`None`, which is not `False` and must never be read as it. Every profile-gated helper
follows one rule for all three: `True` apply, `False` drop, `None` apply *and* record
the flag name in `WireRequest.unverified`, so an assumption reaches `CallTrace`
instead of being silently baked into a request.

The `None` default is also what makes the `ProfileTable` overlay work. Resolution is
`default.model_copy(update=row.model_dump(exclude_unset=True))`, so a row that omits
a field inherits the provider default while a row that writes `thinking=None`
explicitly means "verified absent, do not inherit". The distinction is *set-ness*,
not value, which is why nothing here carries a non-empty default.

Four fields -- `verification`, `format_bias`, `verbosity_bias` and
`emphasis_sensitive` -- are schema only. Nothing in the communication layer reads
them; they are facts the lint and evaluation layers will want, reserved now so that a
hand-maintained table does not have to be rewritten when those layers land.
"""

from franca.core.enums import Strategy, ThinkingMode
from franca.core.ids import Dialect
from franca.core.profile import BaseProfile


# `frozen=True` is restated rather than inherited: pydantic carries it through
# `model_config` at runtime, but under PEP 681 `dataclass_transform` a type checker
# reads a bare subclass as a non-frozen dataclass inheriting from a frozen one and
# rejects it. `extra="forbid"` needs no such restatement.
class ChatProfile(BaseProfile, frozen=True):
    """What franca believes about a chat model, as one row of a profile table.

    Frozen, and `extra="forbid"` inherited from `BaseProfile`: a misspelled flag in a
    hand-written table raises instead of being accepted, ignored, and read as
    "unverified" forever.

    Every field below is optional and every default means unverified, so
    `ChatProfile(provider=ANTHROPIC, model_prefix="")` is a valid provider-wide
    fallback row that claims nothing at all.

    Attributes:
        thinking: How this model expresses a reasoning budget on the wire. It is the
            only thing an adapter branches on when building its thinking config.
        effort_levels: The effort names the model accepts, in the provider's own
            spelling, lowest first.
        default_effort: The effort level to send when the caller named none.
        sampling_allowed: Whether `temperature` and its neighbours are honoured.
            `False` drops the sampling parameters together with `ignored_params`; an
            output token budget is never gated by this and is always emitted.
        structured_output: The strategy to prefer for structured output when the
            caller did not pick one. An explicit caller strategy is never silently
            substituted -- an unsupported one raises instead.
        tools_require_dialect: A dialect this model can only use tools through, so
            selection can pin it when the package carries tools.
        default_max_output_tokens: The output budget to send when the package names
            none.
        ignored_params: Request parameters this model accepts and silently ignores.
            Dropped alongside the sampling parameters when sampling is disallowed.
        retired: Whether the model is gone. `ChatModel.prepare` raises `ProfileError`
            on `True`, before any I/O.
        token_factor: This model's tokens per reference-tokeniser token, for the lint
            layer's estimates. Never used for cost.
        budget_tokens_allowed: Whether an explicit reasoning token budget is
            accepted; `False` drops the budget and falls back to effort or adaptive.
        prefill_allowed: Whether an assistant turn may be prefilled to steer the
            opening of the reply -- how `json_mode` is obtained where it is.
        strict_enforced: Whether the provider actually enforces a strict schema
            rather than treating it as a hint.
        tool_calling: Whether the model can call tools at all.
        parallel_tool_calls: Whether it may emit more than one tool call per turn.
        streaming: Whether the model streams.
        image_input: Whether image parts are accepted in a request.
        image_output: Whether the model can return images.
        history_append_only: Whether earlier turns must be resent unchanged, which
            forbids editing or dropping history mid-conversation.
        mid_conversation_effort: Whether the effort level may change between turns of
            one conversation.
        verification: How this row's claims were established, in prose. Schema only.
            Distinct from `verified`, which is the date of the last check.
        format_bias: The output shape this model drifts towards absent instruction,
            e.g. `"markdown"`. Schema only, for the lint layer.
        verbosity_bias: Which way its length drifts, e.g. `"verbose"`. Schema only.
        emphasis_sensitive: Whether emphasis markers in a prompt measurably change
            its behaviour. Schema only.
    """

    thinking: ThinkingMode | None = None
    effort_levels: tuple[str, ...] = ()
    default_effort: str | None = None
    sampling_allowed: bool | None = None
    structured_output: Strategy | None = None
    tools_require_dialect: Dialect | None = None
    default_max_output_tokens: int | None = None
    ignored_params: frozenset[str] = frozenset()
    retired: bool | None = None
    token_factor: float | None = None

    budget_tokens_allowed: bool | None = None
    prefill_allowed: bool | None = None
    strict_enforced: bool | None = None
    tool_calling: bool | None = None
    parallel_tool_calls: bool | None = None
    streaming: bool | None = None
    image_input: bool | None = None
    image_output: bool | None = None
    history_append_only: bool | None = None
    mid_conversation_effort: bool | None = None

    verification: str | None = None
    format_bias: str | None = None
    verbosity_bias: str | None = None
    emphasis_sensitive: bool | None = None
