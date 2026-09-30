"""The chat capability's adapter type: what every chat dialect implements.

There is one adapter contract per capability, and this module is chat's. It adds no
behaviour to `franca.core.adapter.Adapter` -- it only pins the two type parameters to
the chat IR, so that a dialect module says `Adapter[PromptPackage, ModelResponse]`
once and everything above the adapter seam (the registry, `ChatModel`, the
conformance suite) can name that type without respelling the pair.

The alias is deliberately the non-streaming `Adapter`. Streaming is a separate base
class in core precisely so a wire that does not stream is not forced to pretend, and
the chat delta type does not exist yet; when it does, this becomes
`StreamingAdapter[PromptPackage, ModelResponse, Delta]` and the adapters that stream
widen to it. Nothing that consumes `DialectAdapter` today calls `from_stream`, so
that widening is additive.
"""

from franca.chat.ir import ModelResponse, PromptPackage
from franca.core.adapter import Adapter

type DialectAdapter = Adapter[PromptPackage, ModelResponse]
"""A chat adapter: a `PromptPackage` in, a `ModelResponse` out, one wire dialect.

Becomes `StreamingAdapter[PromptPackage, ModelResponse, Delta]` when the chat delta
type lands with streaming; the non-streaming half of the contract is unchanged by
that, so an adapter written against this alias keeps compiling.
"""
