"""The chat intermediate representation: one request shape, one response shape.

Every chat adapter reads a `PromptPackage` and writes a `ModelResponse`, whichever
wire it speaks, so this module is the vocabulary the whole capability shares. It is
deliberately a *superset* of what any single dialect supports: a field an adapter
cannot express is dropped by that adapter, never by the caller, which is what makes
the same package runnable against Anthropic, OpenAI and Google without a rewrite.

Three properties are load-bearing and none of them is obvious from the field list.

`Item.artifact` carries a provider's own opaque reasoning payload -- an Anthropic
thinking block with its signature, a Responses reasoning item, a Gemini thought
signature -- and it is hidden from `repr()` but **dumped**. A transcript persisted
without it is rejected on the next turn, so `exclude=True` here would be a bug, not a
tidiness win. `ModelResponse.raw` is the opposite case: a vendor envelope kept for
debugging and excluded from `model_dump()`, so a persisted response stays
franca-shaped.

`PromptPackage` is `strict=True`. A request is the one place where a silent coercion
is expensive: `max_output_tokens="1"` becoming `1` hides a bug that only shows up as
a bill. Sequences relax that to accept a list -- writing `items=[...]` is how everyone
writes it -- but their *elements* stay `InstanceOf`, so a bare dict is not quietly
promoted to an `Item`. That costs one thing, deliberately: a package dumped in Python
mode round-trips through `model_validate_json`, not `model_validate`.

Everything here is frozen and `extra="forbid"`. Frozen because middleware must not be
able to edit a request between a retry's attempts; `extra="forbid"` because a
misspelled field on a request would otherwise reach no wire and raise nothing.

This module is version-bump-protected: renaming or removing a field here is a major
version change, since adapters, cassettes and persisted transcripts are all keyed on
these names.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, InstanceOf

from franca.core.enums import StateMode, Strategy
from franca.core.ids import Dialect, Provider
from franca.core.types import Asset, Traced, Usage


class Item(BaseModel, frozen=True, extra="forbid"):
    """One turn, or one part of one turn, in a conversation.

    An item is flat rather than a union so that a transcript is one homogeneous
    sequence: `kind` says what the payload is and which of the optional fields the
    adapter should read. `role` spans every vocabulary in use -- `"model"` is
    Google's spelling of `"assistant"`, `"developer"` is OpenAI's second instruction
    channel -- and an adapter maps it onto whatever its own wire calls that.

    Attributes:
        role: Who the turn belongs to.
        kind: What the payload is; `"other"` is the escape hatch for a block franca
            has no vocabulary for, which then survives only in `artifact`.
        text: The text of a `"text"` or `"reasoning"` item.
        name: A tool's name on a `"tool_call"` or `"tool_result"`.
        call_id: The provider's identifier tying a `"tool_result"` back to its
            `"tool_call"`.
        args: A tool call's arguments, already decoded.
        asset: The binary payload of an `"image"` item.
        server_side: Whether the provider executed this itself -- a server-side tool
            such as web search -- so the caller is not expected to run it.
        artifact: The provider's own opaque payload for this item, passed back
            verbatim on the next turn: a thinking block with its signature, a
            Responses reasoning item, a Gemini thought signature. Hidden from
            `repr()` because it is noisy and unreadable, but **not** excluded from
            `model_dump()`: a persisted transcript that loses a thinking signature is
            rejected by the provider on the next turn.
    """

    role: Literal["system", "developer", "user", "assistant", "model", "tool"]
    kind: Literal["text", "image", "tool_call", "tool_result", "reasoning", "other"]
    text: str | None = None
    name: str | None = None
    call_id: str | None = None
    args: dict[str, Any] | None = None
    asset: Asset | None = None
    server_side: bool = False
    artifact: dict[str, Any] | None = Field(default=None, repr=False)


class SystemBlock(BaseModel, frozen=True, extra="forbid"):
    """One block of system instruction, with a hint about where it belongs.

    Blocks rather than a single string because caching is per block: a dialect that
    supports prompt caching marks a stable preamble and leaves a volatile tail
    uncached, which is impossible once the two have been concatenated.

    Attributes:
        text: The instruction itself.
        position: Sort key within the system prompt. Equal positions keep declaration
            order, so the default of `0` means "in the order I wrote them".
    """

    text: str
    position: int = 0


class ToolDef(BaseModel, frozen=True, extra="forbid"):
    """A tool the model may call, described in JSON Schema.

    Attributes:
        name: The tool's name, as the model will spell it in a call.
        parameters: The JSON Schema of the tool's arguments.
        description: What the tool does, in the model's own context window.
        strict: Whether the provider must enforce the schema exactly; `None` leaves
            the choice to the adapter and the profile.
        server_side: Whether the provider runs this tool itself, in which case the
            caller never sees a call to execute.
    """

    name: str
    parameters: dict[str, Any]
    description: str = ""
    strict: bool | None = None
    server_side: bool = False


class Reasoning(BaseModel, frozen=True, extra="forbid"):
    """How much thinking is asked for, in franca's terms rather than a vendor's.

    The four ways providers express a reasoning budget do not map onto each other --
    a token budget is not an effort level -- so the request carries the caller's
    intent and the adapter renders it in whatever its wire understands, guided by the
    model's `ThinkingMode`.

    Attributes:
        mode: The shape of the request: `"adaptive"` lets the model decide,
            `"enabled"`/`"disabled"` turn thinking on or off, `"level"` selects a
            named tier. `None` means "say nothing", which is not the same as
            `"disabled"`.
        effort: The named tier when `mode` is `"level"`, e.g. `"high"`; the vocabulary
            is per model and lives on the profile.
        budget_tokens: A token budget for thinking, for the wires that take one.
        artifact_present: Whether the items being sent carry provider reasoning
            artifacts, so an adapter can tell an empty history from a stripped one.
    """

    mode: Literal["adaptive", "enabled", "disabled", "level"] | None = None
    effort: str | None = None
    budget_tokens: int | None = None
    artifact_present: bool = False


class OutputFormat(BaseModel, frozen=True, extra="forbid"):
    """What shape the answer must take, and how that is to be obtained.

    `strategy` is the interesting field: the same JSON Schema is enforced natively by
    one model, through a forced tool call by another, and by a JSON mode plus a
    prefill by a third. The caller states the schema once; the profile decides which
    route a given model can actually take.

    Attributes:
        kind: `"text"` for prose, `"json"` for a structured answer. `None` leaves the
            provider's default alone.
        json_schema: The schema the answer must satisfy when `kind` is `"json"`.
        strict: Whether the schema is to be enforced rather than merely suggested;
            `None` defers to the profile.
        strategy: How the schema is enforced. `None` lets the adapter pick the best
            route the profile allows.
    """

    kind: Literal["text", "json"] | None = None
    json_schema: dict[str, Any] | None = None
    strict: bool | None = None
    strategy: Strategy | None = None


class PackageMeta(BaseModel, frozen=True, extra="forbid"):
    """Caller intent that changes how a request is built but is not itself content.

    These are the facts a lint rule or an adapter needs and cannot infer from the
    items: what this call is for, and whether it is a real turn at all.

    Attributes:
        route: The caller's own name for this call site, carried through to traces.
        intended_output_style: What the answer is meant to read like, for rules that
            check the instructions against it.
        is_code_route: Whether the answer is code, which several rules treat
            differently from prose.
        configuration_update: Whether this turn only changes settings rather than
            asking for anything; a stateful dialect can send it as such.
        store: Whether the provider may retain the exchange; `None` leaves the
            provider's default alone.
    """

    route: str | None = None
    intended_output_style: str | None = None
    is_code_route: bool = False
    configuration_update: bool = False
    store: bool | None = None


class PromptPackage(BaseModel, frozen=True, extra="forbid", strict=True):
    """One chat request, before any dialect has had an opinion about it.

    This is the object a caller builds and every adapter consumes. `provider`,
    `dialect` and `model` are hints, not routing: leaving them `None` is normal and
    means the registry decides. Everything else is the request itself.

    Validation is strict, so `max_output_tokens="1"` raises instead of quietly
    becoming `1`. The three sequence fields carry `strict=False` so that a plain list
    is accepted and stored as a tuple -- nobody writes a one-element tuple by choice
    -- while their elements are `InstanceOf`, so a bare dict is *not* promoted to an
    `Item`. The consequence is worth stating: a dumped package is re-read with
    `model_validate_json`, since `model_validate` of a Python dict would be handing
    dicts to those fields.

    Attributes:
        provider: Which provider to authenticate with, when the caller has pinned one.
        dialect: Which wire dialect to speak, when the caller has pinned one.
        model: The model identifier, as the caller spells it.
        system: The system instruction, in blocks; see `system_text`.
        items: The conversation, oldest first.
        tools: The tools the model may call.
        reasoning: How much thinking to ask for.
        output_format: What shape the answer must take.
        max_output_tokens: A ceiling on the answer's length. A budget, never a
            sampling parameter, so a model that ignores `temperature` still honours
            this.
        sampling: Sampling parameters by their franca name, e.g. `{"temperature":
            0.2}`. A dict rather than fields because which ones a model honours is a
            profile fact, and an unsupported one is dropped rather than rejected.
        cache_hints: What the caller believes is worth caching, for the dialects that
            can express it.
        state_mode: Whether the conversation is replayed on every turn or held by the
            provider.
        continuation: The server-side handle from a previous `ModelResponse`, sent as
            `previous_response_id` or `previousInteractionId`. Ignored by stateless
            dialects.
        meta: Caller intent that is not content.
        provider_options: Raw per-dialect overrides, keyed by dialect and merged into
            that dialect's body verbatim -- so the keys are spelled in the wire's own
            case, `camelCase` for Google. The escape hatch for anything franca has no
            field for.
    """

    provider: Provider | None = None
    dialect: Dialect | None = None
    model: str | None = None
    system: tuple[InstanceOf[SystemBlock], ...] = Field(default=(), strict=False)
    items: tuple[InstanceOf[Item], ...] = Field(default=(), strict=False)
    tools: tuple[InstanceOf[ToolDef], ...] = Field(default=(), strict=False)
    reasoning: Reasoning = Reasoning()
    output_format: OutputFormat = OutputFormat()
    max_output_tokens: int | None = None
    sampling: dict[str, float | int] = Field(default_factory=dict)
    cache_hints: dict[str, Any] = Field(default_factory=dict)
    state_mode: StateMode = StateMode.stateless
    continuation: str | None = None
    meta: PackageMeta = PackageMeta()
    provider_options: dict[Dialect, dict[str, Any]] = Field(default_factory=dict)

    def system_text(self) -> str:
        """Render the system blocks as the single string a one-slot dialect wants.

        Blocks are ordered by `position` and, within one position, by the order they
        were declared; the sort is stable, so the common case of every block at the
        default `0` is simply declaration order. Non-empty texts are joined by a
        blank line, and an empty block contributes nothing -- no separator, no
        leading whitespace.

        Returns:
            The joined system instruction, or `""` when there are no blocks.
        """
        blocks = sorted(self.system, key=lambda block: block.position)
        return "\n\n".join(block.text for block in blocks if block.text)

    def instruction_text(self) -> str:
        """Render everything that instructs the model, as opposed to conversing with it.

        Defined precisely as `system_text()`, followed by the `text` of every item
        whose `role` is `"system"` or `"developer"` and whose `kind` is `"text"`, in
        the order those items appear. Parts are joined by a blank line and empty
        parts are skipped, so this equals `system_text()` for the usual package that
        keeps its instructions in `system`.

        This is the string a lint rule reads, and what a dialect with a single
        `instructions` slot sends. Instruction-shaped items are included because some
        callers ingest a transcript that already spells its system prompt as a turn,
        and a rule that looked only at `system` would then see nothing.

        Returns:
            The joined instruction surface, or `""` when there is none.
        """
        instructions = ("system", "developer")
        parts = [
            self.system_text(),
            *(
                item.text
                for item in self.items
                if item.role in instructions and item.kind == "text" and item.text
            ),
        ]
        return "\n\n".join(part for part in parts if part)

    def last_item(self) -> Item | None:
        """Return the most recent item, which is what most lint rules look at.

        Returns:
            The final item, or `None` when the package carries no items.
        """
        return self.items[-1] if self.items else None


class Refusal(BaseModel, frozen=True, extra="forbid"):
    """A policy decline reported in-band: HTTP 200 with a stop reason of `refusal`.

    A refusal is a response, not an error. Current Anthropic models answer 200 with
    `stop_reason="refusal"` rather than a 4xx, and an evaluation runner must be able
    to tell "the model declined" from "the call failed" off one field, because
    scoring a decline as a capability failure is a measurement bug.

    Attributes:
        category: The provider's own label for why it declined. An **open** set,
            never a `Literal`: providers add categories without notice, and franca
            has no branch on the value -- an unknown one must reach the caller
            intact, not raise a `ValidationError` on a response that already arrived.
        explanation: The provider's prose, when it gave any.
    """

    category: str | None = None
    explanation: str | None = None


class ModelResponse(Traced, frozen=True):
    """One chat answer, translated out of whatever wire produced it.

    The items are the answer in the same vocabulary the request used, so a response
    can be appended to the next request's `items` unchanged -- artifacts, signatures
    and all -- which is what makes a multi-turn tool loop portable across dialects.

    Attributes:
        model: The model identifier the call was made with.
        provider: Who served it.
        dialect: Which wire dialect answered.
        items: The answer, as text, reasoning and tool-call items.
        stop_reason: Why generation stopped, in the wire's own words. An open
            vocabulary that is never parsed into an enum.
        usage: Token counts for this call.
        refusal: Set when, and only when, `stop_reason` says the model declined.
        served_model: The model id the wire itself reported, which is how a silent
            alias or a snapshot swap becomes visible.
        continuation: The provider-side handle for this exchange, to be sent as the
            next request's `continuation`.
        structured: The decoded structured answer, when one was asked for: the parsed
            text for a native or JSON-mode strategy, the forced tool's arguments for
            the tool-calling one.
        raw: The decoded provider payload. Kept for debugging and excluded from both
            `model_dump()` and `repr()`, so a persisted transcript never silently
            grows a vendor envelope.
    """

    model: str
    provider: Provider
    dialect: Dialect
    items: tuple[Item, ...]
    stop_reason: str | None = None
    usage: Usage = Usage()
    refusal: Refusal | None = None
    served_model: str | None = None
    continuation: str | None = None
    structured: dict[str, Any] | None = None
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    @property
    def text(self) -> str:
        """The model's prose: the text of every assistant or model item of kind `text`.

        Concatenated with no separator, because consecutive text items are the wire's
        own blocks of one continuous answer. Reasoning items are excluded -- thinking
        is not the answer -- as are tool results and any text the caller supplied.
        """
        return "".join(
            item.text or ""
            for item in self.items
            if item.kind == "text" and item.role in ("assistant", "model")
        )

    @property
    def tool_calls(self) -> tuple[Item, ...]:
        """Every item that asks the caller to run a tool, in the order they arrived."""
        return tuple(item for item in self.items if item.kind == "tool_call")
