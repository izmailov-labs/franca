# franca — working notes

The model communication layer: transports, connectors, one IR per capability, an adapter per
wire dialect, middleware and a registry. One request shape for every LLM wire dialect.

A **single-package repository** (uv >= 0.12): the repository root *is* the package. It was
extracted from the `validia.dev` uv workspace, where it lived at `libs/franca`; `validia` stayed
behind and will depend on the published release. Its configuration layer, `whence`, was extracted
the same way and is consumed from PyPI.

## Commands

```bash
make install       # uv sync --group dev + pre-commit install
make lint          # ruff check + ruff format --check
make fmt           # ruff check --fix + ruff format
make typecheck     # mypy (strict)
make test          # pytest (offline; contract tests deselected)
make test-contract # live provider calls; needs real keys
make test-mock     # real HTTP against aimock, the local mock LLM server; needs node
make aimock        # run aimock in the foreground on :4010 for manual experiments
make cov           # pytest --cov (fail_under=90)
make encoding      # fail on any read missing an explicit encoding=
make build         # uv build + twine check
make all           # everything CI runs
```

Run one test: `uv run pytest tests/core/test_connector.py`

## Conventions

- **Layout is `src/`.** Tests import the *installed* package (`--import-mode=importlib`), not
  the working tree.
- **Tooling configuration lives once, in `pyproject.toml`.** Ruff, mypy, pytest and coverage are
  configured there alongside the packaging metadata; there is no second config file.
- **mypy is `strict = true` and covers `tests/` too.** New code lands annotated; do not add
  `disallow_untyped_defs = false` or blanket `ignore_missing_imports`. Suppressions must be
  specific: `# type: ignore[code]`, never bare.
- **Async is the default shape.** `pytest-asyncio` runs in auto mode, so `async def test_*` needs
  no decorator. Ruff's `ASYNC` rules are on — they catch blocking calls inside `async def`.
- **Runtime dependencies are inherited by every consumer.** Adding one is a real decision. There
  are exactly two:
  - **pydantic (`>=2.12,<3`)** — the stdlib route is `Any`-heavy code owned forever, and
    dataclasses validate nothing on the outbound request, so a bad request would carry no field
    path (see §11 of `docs/research/communication-layer-plan.md`).
  - **whence (`>=1.0,<2`)** — layered settings loading with provenance, so a bad setting reports
    the file and line it came from. It has no runtime dependencies of its own.

  httpx is an optional extra (`franca[http]`), imported lazily inside `HttpxTransport.__init__`
  so `pip install franca` works without it. The `no-extras` CI job enforces that.
- **Async is asyncio-declared but loop-neutral by construction.** Only four operations touch the
  loop — sleep, monotonic/wall time, HTTP I/O, async-generator close — and each goes through an
  injected `Clock` or `Transport`, or an explicit `aclose()`. So no `anyio`: a trio user passes
  their own `TrioClock` plus `HttpxTransport`. This is enforced, not merely intended — ruff
  `TID251` bans `import asyncio` everywhere in `src/` except `franca/core/clock.py`. Deadlines are
  `clock.monotonic()` arithmetic; a total deadline and cancellation belong to the caller
  (`asyncio.timeout` / `trio.move_on_after`), so core has no `asyncio.timeout`, task groups or
  locks.
- **Endpoint rows are data; profile rows are *measurements*.** `chat/endpoints.py` carries one
  row per reachable surface -- host, path, auth header, required version headers, tri-state
  `features` -- so adding a provider is adding a row, never editing code that sends requests.
  `chat/profiles.py` records only what was sent to a live API and what came back, which is why
  every row carries a `verified` date. Never fill either table from vendor documentation: `None`
  means nobody checked, and a guessed `False` makes franca refuse locally a request the model
  would have accepted. A compatibility surface (xAI, DeepSeek) shares a *dialect*, not a
  contract, so it gets its own rows and borrows no profile.
- **Configuration goes through `whence`, never `os.environ` directly.** `core/settings.py`
  declares the schema and makes one call into whence; `whence` is an implementation detail, and
  `ConfigError` is what franca promises.
- **Live provider calls are opt-in.** Tests under `tests/contract/` carry the `contract` marker,
  which `addopts` deselects. They read keys from the environment, falling back to the gitignored
  repo-root `.env`, and skip cleanly when a key is absent.
- **The mock model is aimock, run through npx.** Tests under `tests/mock/` carry the `mock`
  marker, deselected like `contract`. `make test-mock` spawns `@copilotkit/aimock` (version
  pinned in both `tests/mock/conftest.py` and the Makefile) on a free port with
  `tests/mock/fixtures/`, waits for `/health`, and skips cleanly without `npx`;
  `AIMOCK_BASE_URL` points the suite at a server you started yourself (`make aimock`). It sits
  between the cassettes and the contract suite: real `HttpxTransport` and real SSE, no key and
  no bill. One fixture file serves both wires -- aimock renders it in whichever dialect the path
  implies -- and matches on the *last user message only*. A request no fixture matches is a 404
  `No fixture matched`, which franca classifies as a non-retryable provider error: add a
  fixture, never loosen the classification. The spawned server enforces `AIMOCK_API_KEYS`, so a
  wrong key is a real 401.
- **Ruff is pinned exactly** (`ruff==0.16.6`). It has no 1.0 and does not follow semver below it,
  so a floating version would silently change the lint gate. The `select` list is explicit for
  the same reason — ruff 0.16 grew its *default* set from 59 to 413 rules.
- **Docstrings are load-bearing**: ruff's `D` rules (google convention) enforce them in `src/`.
- **Every read passes `encoding=` explicitly.** `make encoding` turns a missing one into an
  error, because otherwise it only fails on a Windows code page.
- **Public API** is whatever `src/franca/__init__.py` lists in `__all__`. Everything else is
  internal and may change without a major bump.

## Diagrams

`diagrams/franca-core.*` is the whole communication layer in one picture; `diagrams/panels/`
holds higher-detail per-area sources. The `.mmd` files are the source of truth, never the PNG:
edit those and re-render with the `/diagram` skill.

Three label rules, all learned by breaking them. Labels must stay **ASCII**, because the renderer
ships source into the page through `atob()`. Labels must contain **no bare `<` or `>`**, including
`->` and `<=`, because mermaid HTML-escapes them and you get `&gt;` in the output. And labels must
contain **no `<br/>`**: it works in SVG and PNG, but the excalidraw converter emits it as literal
`<br>` text and then word-wraps mid-word. Use single-line labels and let the renderer wrap.

## Release

One package, so tags are plain: `vX.Y.Z`.

Version is static in `pyproject.toml`; `__version__` reads it back at runtime via
`importlib.metadata`. To release: bump the version, write the `CHANGELOG.md` entry, tag `vX.Y.Z`,
push the tag. `release.yml` verifies the tag against `pyproject.toml`, builds, and publishes via
PyPI Trusted Publishing (OIDC — there is no API token in this repo).

Two constraints worth remembering: the **trusted publisher must be configured on PyPI before the
first tag** (owner `izmailov-labs`, repository `franca`, workflow `release.yml`, environment
`pypi`), and **PyPI rejects new files added to a release older than 14 days**, so everything for
a version ships in one run.
