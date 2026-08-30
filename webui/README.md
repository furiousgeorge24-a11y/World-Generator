# Shared preview webui

A pipeline-agnostic control panel + pan/zoom viewer for any seed-driven
image generator in this repository. The server and front end contain no
generator-specific content: the app name, version, every slider, the
view list, and the report all come from a backend **adapter module**
that the generator provides about itself.

    py -3.14 serve.py --backend <module> [--root <dir>] [--port 5000]

`--root` is prepended to `sys.path` before the import, so a pipeline
runs it from its own directory as
`py -3.14 ..\webui\serve.py --backend <its_adapter_module> --root .`

Requires Flask (plus whatever the backend itself needs). The server
hot-reloads on edits to any imported module; `WEBUI_RELOAD=0` pins a
single process.

## Adapter contract

The adapter is a plain module. Required functions:

- `meta() -> dict` — `{"name": str, "version": str, "controls": [...]}`.
  Each control dict describes one UI input:
  - `name`: identifier (slider label)
  - `ctype`: `"float"` | `"int"` | `"bool"`
  - `default`, `lo`, `hi`: value and range (`lo`/`hi` unused for bool)
  - `tier`: `"primary"` (always visible) | `"advanced"` (collapsed)
  - `invalidates`: `"full"` | `"late"` | `"render"` (see caching below)
  - `promise`: one-line user-facing description of what the control does
  - `temp` (optional): truthy marks the control as a stub/placeholder
- `generate(seed: int, controls: dict, size: int) -> world` — run the
  generator; `controls` maps control names to values (possibly partial —
  the backend applies its own defaults). The returned `world` is opaque
  to the server; it is only handed back to the functions below.
- `views(world) -> list[str]` — names of the renderable views for this
  result. **A view named `hypsometric` is expected**: it is the base
  view every pipeline should provide, and the viewer selects it by
  default and draws the history thumbnails from it. See "Base view"
  below. Backends that omit it fall back to the first entry in this
  list.
- `render_png(world, view: str) -> bytes` — PNG bytes for one view.
  Rendering must be cheap relative to generation; the server renders
  every view of each result so the client can swap views instantly.
- `report(world) -> dict` — JSON-serializable run report shown in the
  UI. If it contains `"findings": [{"level": ...}, ...]`, entries with
  level `"warn"` are surfaced as a badge.

Optional functions (enable the caching tiers; omit them and every
control change is treated as a full regeneration):

- `invalidation_of(name: str) -> "full" | "late" | "render"` — what a
  change to this control invalidates. `"render"` re-renders a cached
  result without regenerating; `"late"` re-runs only a cheap tail;
  `"full"` regenerates.
- `generate_head(seed, controls, size) -> head` and
  `run_tail(head, controls) -> world` — the late-class fast path. The
  server caches `head` (keyed by seed, size, and all non-late,
  non-render controls) and calls `run_tail` per request; `run_tail`
  must not mutate `head` (clone inside it).
- `set_render_controls(world, controls) -> None` — apply the
  render-class subset of `controls` to a cached `world` before
  re-rendering.

Determinism is the backend's business, but the caching assumes it: the
server reuses a cached result whenever seed, size, and all non-render
controls match.

## Base view: `hypsometric`

Every pipeline in this repository should render its primary image as a
**hypsometric view** — a top-down render whose colour is a direct
function of the value being generated, read through an elevation-style
ramp (sea-level-referenced water tones below, land ramp above), rather
than a shaded, lit, or stylized picture. The author's aesthetic
reference images in `examples/` are hypsometric renders, so this is the
form results are judged against and compared in.

Name that view `hypsometric` in `views()`. Additional views (debug
fields, intermediate layers, alternate palettes) are welcome and appear
in the selector alongside it; the viewer just treats `hypsometric` as
the one to open with and to build the history rail from.
