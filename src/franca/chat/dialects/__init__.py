"""The chat dialect adapters: one module per wire, each translating the same IR.

A dialect is a wire shape, not a vendor, so every module under here owns exactly one
request/response format and knows nothing about any other. Three have landed --
`anthropic_messages`, `openai_chat` and `google_generate_content` -- and between them
they serve five providers, because xAI and DeepSeek reach `openai_chat` through their
own endpoint rows rather than through code of their own. The OpenAI Responses and
Google Interactions wires are separate dialects and arrive with their milestones.

Nothing is imported here on purpose. This package is the natural place for the
registry's `ADAPTERS` mapping, but building that mapping eagerly would make
`import franca.chat.dialects` pull in every adapter -- and, through them, every
dialect's dependencies -- for a caller who wanted one. The mapping lands with the
registry, populated from the adapters the registrar is actually given; until then a
caller reaches an adapter by importing its own module.
"""
