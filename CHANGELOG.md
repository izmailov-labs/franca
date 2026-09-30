# Changelog

All notable changes to this package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-29

### Added

- Extracted from the validia.dev workspace into a standalone repository; `whence`
  is now consumed from PyPI (`whence>=1.0,<2`).
- `core/settings.py`: `Settings`, `ProviderSettings`, `RoutePolicy`, `RetryPolicy`,
  `load_settings()` and `SettingsKeyProvider`. Loading is delegated to `whence`,
  so a bad setting reports the file it came from; API keys are read at call time
  and never stored in a model.
- `tests/mock/`: a `mock`-marked suite that drives the real `HttpxTransport` and SSE
  path against [aimock](https://aimock.copilotkit.dev), a local mock LLM server the
  suite spawns through `npx` (`make test-mock`; `make aimock` runs it by hand).
  Deselected by default like `contract`; `AIMOCK_BASE_URL` reuses a running server.
- `GoogleGenerateContentAdapter`: the Gemini REST wire, the third chat dialect. It is
  the first adapter to set `WireRequest.path`, because this wire carries the model
  there and signals streaming by swapping the method rather than setting a body key.
  Its `error_from` restores `failure_class="auth"` for the 400 `INVALID_ARGUMENT` that
  Google answers a bad key with, reading the structured `API_KEY_INVALID` reason rather
  than matching on message text.
- Endpoint rows for Google, xAI and DeepSeek, bringing the built-in table to five
  providers across three dialects: xAI and DeepSeek are served by `OpenAIChatAdapter`
  through rows of their own, so a compatibility surface costs data and no code. Each row
  records what a deliberately wrong key actually returns, measured 2026-09-16: DeepSeek
  answers `401`, which the connector's status map already classifies as `auth`, while xAI
  answers `400` with a bare-string `error` body, so franca reports a wrong xAI key as a
  non-retryable `provider` failure rather than `auth`. Refining that would mean matching
  on message text, which the row documents instead of guessing.
- Provider default profile rows for `google`, `xai` and `deepseek`. Every capability on
  them is `None`: no key for those providers was available, and an unmeasured claim is
  not recorded as a measured one.
- `tests/chat/test_endpoints.py`: table-level invariants the individual rows cannot
  state -- unique ids, a resolvable profile per provider, clean URL joins.
- Profile row for `claude-fable-5-1`, measured 2026-09-16 and identical to the rest of
  the 5 generation: no sampling, adaptive thinking only, `output_config.effort`.

### Fixed

- `prefill_allowed` is now `False` on `claude-sonnet-5` and `claude-opus-5`. A re-probe
  on 2026-09-16 found the 5 generation answering an assistant prefill with
  `400 This model does not support assistant message prefill`, where the 2026-09-07
  sweep recorded it as accepted. The rows and their test assertions were corrected to
  the wire. Nothing in franca changed between the sweeps; the provider did, which is
  the case `verified` dates exist for.

[Unreleased]: https://github.com/izmailov-labs/franca/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/izmailov-labs/franca/releases/tag/v0.1.0
