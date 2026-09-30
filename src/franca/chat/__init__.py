"""The chat capability: the conversation IR, its profiles and the dialect adapters.

`franca.chat.ir` is the vocabulary every chat adapter reads and writes -- a
`PromptPackage` in, a `ModelResponse` out -- and it is the same object whichever wire
ends up carrying the call. `franca.chat.profile` says what a given model does with
that vocabulary, and the adapters that translate it live under
`franca.chat.dialects`, one module per wire dialect.

Nothing is re-exported here on purpose. Importing `franca` or `franca.core` must not
drag a capability package in, and a capability package must not drag every adapter
in, so reaching one is always an explicit `from franca.chat.ir import PromptPackage`.
"""
