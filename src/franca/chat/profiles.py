"""Built-in chat profile rows: what franca has *measured* about specific models.

A profile row is the answer to "one vendor, one contract?" -- which is no. Every fact
below was obtained by sending the parameter to the live API and recording whether the
request was legal, not by reading documentation, which is why `verified` carries a
date and `source` is left unset: the source is the wire.

The measurement that motivates the whole table, re-taken on 2026-09-16 by sending
`max_tokens=1` with each parameter in turn:

    parameter                haiku-4-5  sonnet-4-5  opus-4-5  sonnet-5  opus-5  fable-5-1
    temperature / top_p      accepted   accepted    accepted  400       400     400
    thinking.enabled+budget  accepted   accepted    accepted  400       400     400
    thinking.adaptive        400        400         400       accepted  accepted accepted
    output_config.effort     400        400         accepted  accepted  accepted accepted
    assistant prefill        accepted   accepted    accepted  400       400     400

Three groups, not two, and the boundaries do not line up: `effort` splits
`opus-4-5` away from its own generation, while sampling and thinking split it the
other way. No single "anthropic" branch can be right for all six, which is the
argument for per-model rows resolved by longest prefix rather than `if provider ==`.

**The prefill row changed, and that is the point of dating a row.** On 2026-09-07
every model accepted an assistant prefill, and this table said so. On 2026-09-16 the
5 generation answers it `400: This model does not support assistant message prefill.
The conversation must end with a user message.` Nothing about franca changed; the
provider did. A row is a claim about a day, which is why `verified` exists and why a
re-probe outranks both the old row and any documentation.

Tri-state matters here. `False` means *measured absent* -- a request carrying that
parameter is a 400, so franca can refuse it locally with a field path instead of
paying a round trip. `None` means nobody has checked, which is not the same claim and
must not be silently treated as `False`.

Rows are keyed by `model_prefix` and matched after `normalize_model_id`, so a dated
snapshot (`claude-haiku-4-5-20251001`), an aliased id and a `[1m]` context suffix all
land on the same row.
"""

from datetime import date
from typing import Final

from franca.chat.profile import ChatProfile
from franca.core.enums import ThinkingMode
from franca.core.ids import ANTHROPIC, DEEPSEEK, GOOGLE, OPENAI, XAI
from franca.core.profile import ProfileTable

_MEASURED: Final = date(2026, 9, 7)
"""The first sweep. Still correct for the 4-5 generation, which has not moved."""

_REMEASURED: Final = date(2026, 9, 16)
"""The second sweep, which corrected prefill on the 5 generation and added fable-5-1."""

_ANTHROPIC_LEGACY_SAMPLING: Final = ChatProfile(
    provider=ANTHROPIC,
    model_prefix="claude-haiku-4-5",
    verified=_MEASURED,
    sampling_allowed=True,
    thinking=ThinkingMode.budget,
    budget_tokens_allowed=True,
    prefill_allowed=True,
    effort_levels=(),
    streaming=True,
    notes="effort rejected: 'This model does not support the effort parameter.'",
)

_ANTHROPIC_SONNET_45: Final = _ANTHROPIC_LEGACY_SAMPLING.model_copy(
    update={"model_prefix": "claude-sonnet-4-5"}
)

_ANTHROPIC_OPUS_45: Final = ChatProfile(
    provider=ANTHROPIC,
    model_prefix="claude-opus-4-5",
    verified=_MEASURED,
    sampling_allowed=True,
    thinking=ThinkingMode.budget,
    budget_tokens_allowed=True,
    prefill_allowed=True,
    effort_levels=("high",),
    streaming=True,
    notes="the odd one out: takes effort AND legacy sampling. Only 'high' was measured.",
)

_ANTHROPIC_ADAPTIVE: Final = ChatProfile(
    provider=ANTHROPIC,
    model_prefix="claude-sonnet-5",
    verified=_REMEASURED,
    sampling_allowed=False,
    thinking=ThinkingMode.adaptive,
    budget_tokens_allowed=False,
    prefill_allowed=False,
    effort_levels=("high",),
    streaming=True,
    notes=(
        "temperature and top_p are 400 ('deprecated for this model');"
        " thinking.type=enabled is 400, adaptive is required;"
        " assistant prefill is 400, corrected from the 2026-09-07 sweep."
    ),
)

_ANTHROPIC_OPUS_5: Final = _ANTHROPIC_ADAPTIVE.model_copy(update={"model_prefix": "claude-opus-5"})

_ANTHROPIC_FABLE_5_1: Final = _ANTHROPIC_ADAPTIVE.model_copy(
    update={"model_prefix": "claude-fable-5-1"}
)
"""The top of the lineup, measured to behave exactly like the rest of its generation.

Worth its own row rather than leaving it to the `claude-sonnet-5` prefix, which would
never match it, or to the provider default, which claims nothing. A plain request with
no `thinking` key is accepted, so nothing here gates on the model always thinking.
"""

_ANTHROPIC_DEFAULT: Final = ChatProfile(
    provider=ANTHROPIC,
    model_prefix="",
    streaming=True,
    notes="fallback for an unmeasured Anthropic model: every capability stays None.",
)
"""Everything unknown is `None`, never `False`.

A default row exists so an unrecognised model resolves at all, but it must not
*claim* anything. Guessing `False` here would make franca refuse a request the model
would have accepted; guessing `True` would send a parameter that 400s. `None` is the
honest third answer, and the unverified-feature policy decides what to do with it.
"""

_OPENAI_DEFAULT: Final = ChatProfile(
    provider=OPENAI,
    model_prefix="",
    streaming=True,
    notes=(
        "unmeasured: the account carried no credits when the Anthropic rows were taken,"
        " and OpenAI checks quota before it validates parameters, so every probe"
        " returned 429 rather than a usable 400/200 verdict."
    ),
)
"""OpenAI rows are deliberately empty of capability claims.

This is a measurement gap, not a finding. The known divergences on that side --
`max_tokens` versus `max_completion_tokens`, reasoning models rejecting sampling
parameters -- are real but were not reproducible here, so recording them as verified
would be inventing evidence. They stay `None` until a funded account can answer.
"""

_GOOGLE_DEFAULT: Final = ChatProfile(
    provider=GOOGLE,
    model_prefix="",
    streaming=True,
    notes=(
        "unmeasured: no working Gemini key was available when the Anthropic rows were"
        " taken. thinkingConfig, the topP/temperature pair and structured output are all"
        " real on this wire and none of them has been probed."
    ),
)
"""Google rows are empty of capability claims for the same reason OpenAI's are.

A measurement gap, not a finding. The divergences that matter here are known to exist --
`thinkingConfig` splits the 2.5 generation from what came before, and `maxOutputTokens`
is spent on reasoning as well as prose -- but knowing a divergence exists is not knowing
which side of it a given model falls on, and that is what a row is for.
"""

_XAI_DEFAULT: Final = ChatProfile(
    provider=XAI,
    model_prefix="",
    streaming=True,
    notes=(
        "unmeasured: no xAI key was available. Served through the OpenAI chat wire, so"
        " the adapter's own defaults apply until somebody probes a grok model."
    ),
)
"""xAI rows carry nothing, and the empty row is still worth having.

Resolution needs a default per provider or it raises, so this is what makes
`xai:grok-...` reachable at all. What it must not do is *borrow* the OpenAI rows: the
two providers share a wire, which is a fact about bytes, not about what any particular
model will accept. Sharing a dialect is not sharing a contract.
"""

_DEEPSEEK_DEFAULT: Final = ChatProfile(
    provider=DEEPSEEK,
    model_prefix="",
    streaming=True,
    notes=(
        "unmeasured: no DeepSeek key was available. deepseek-reasoner is documented to"
        " reject temperature and top_p, which is exactly the kind of claim this table"
        " only records once it has been sent."
    ),
)
"""DeepSeek rows are empty, deliberately, including one that documentation would fill.

`sampling_allowed=False` for the reasoner is in the vendor's own documentation, and it
is still `None` here. The table's rule is that a row states what was sent and what came
back, because documentation has been wrong about exactly this before -- and a wrong
`False` makes franca refuse a request the model would have accepted, locally and with a
field path, which is the most confusing failure it can produce.
"""

CHAT_PROFILES: Final = ProfileTable[ChatProfile](
    verified_on=_REMEASURED,
    profiles=(
        _ANTHROPIC_LEGACY_SAMPLING,
        _ANTHROPIC_SONNET_45,
        _ANTHROPIC_OPUS_45,
        _ANTHROPIC_ADAPTIVE,
        _ANTHROPIC_OPUS_5,
        _ANTHROPIC_FABLE_5_1,
    ),
    defaults={
        ANTHROPIC: _ANTHROPIC_DEFAULT,
        OPENAI: _OPENAI_DEFAULT,
        GOOGLE: _GOOGLE_DEFAULT,
        XAI: _XAI_DEFAULT,
        DEEPSEEK: _DEEPSEEK_DEFAULT,
    },
)
"""Every measured chat model, resolved by longest matching `model_prefix`.

`resolve(provider, model)` overlays the matched row onto the provider default with
`exclude_unset`, so a row states only what it measured and inherits the rest.
"""
