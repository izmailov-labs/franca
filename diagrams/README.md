# franca diagrams

## The diagram

`franca-core` is the whole communication layer in one picture: eight numbered areas, each tagged
with the milestone that builds it, with the request spine running top to bottom.

| File | What it is |
| --- | --- |
| `franca-core.mmd` | mermaid source, the single source of truth |
| `franca-core.excalidraw` | editable scene, open at excalidraw.com with File then Open |
| `franca-core.svg` | vector, for docs |
| `franca-core.png` | raster, for chat and issues |

Read it down the spine: `PromptPackage` resolves to a `Selection`, `wrap()` builds the middleware
chain around the `ChatModel` leaf, the leaf calls an adapter which produces a `WireRequest`, the
`Connector` turns that into bytes for the `Transport`, and the response comes back as a
`ModelResponse` carrying `Usage` and `CallTrace`, or as a stream of `Delta`.

Colours carry meaning. Teal hexagons are the injectable seams (protocols you swap in tests).
Orange is the spine. Sand is intermediate representation. Cream rounded boxes are pure functions.

## Build order

The area numbers are the order to implement them, but the milestones are what actually gate the
work. Areas 4, 5, 6 and the leaf are all of M0: that subset is the first real call, and it is worth
getting working end to end before anything else exists.

| Area | Milestone |
| --- | --- |
| 5 Connector, 6 Transport, 4 Adapters, the leaf | M0, the first call |
| 6 Transport streaming half, 7 Response IR | M1 |
| 1 Config + DI, 2 Resolve, 3 Pipeline | M2 |
| 4 Adapters, the five dialects | M3, M4, M5 |
| 3 Pipeline, remaining middleware | M6 |
| 8 Jobs + media | M7, M8 |

## panels/

`panels/` holds fifteen per-area mermaid sources at higher detail: thirteen flowcharts, one per
area, plus three class-level reference sheets (`10a` the prompt/response IR, `10b` wire, selection
and profiles, `10c` the `FrancaError` hierarchy). They are **sources only, not rendered**: a `.mmd`
on its own is not a diagram. Render any of them with the `/diagram` skill when the detail is
actually wanted.

Two conversion facts, both checked against the bundled mermaid-to-excalidraw converter rather than
assumed. The three `classDiagram` sheets do convert, and their `note` blocks survive with the text
intact. `<br/>` does **not**: the converter emits it as literal `<br>` text and applies its own word
wrapping, in every panel that uses it. Line breaks are therefore a render-path affordance (SVG and
PNG honour them); do not rely on them in the editable scene.

## Regenerating

Edit the `.mmd` and re-render; the source is authoritative, never the PNG. Every identifier in a
label must appear verbatim in `docs/research/communication-layer-plan.md`, and every label must
stay ASCII, because the renderer ships source into the page through `atob()`.
