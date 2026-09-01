# Shared preview WebUI

A pipeline-agnostic control panel and pan/zoom viewer for seed-driven image
generators in this repository. It also supports an optional, strictly
separate inspection laboratory for viewing development evidence while a
generator remains unavailable. The server and front end contain no
pipeline-specific content: names, versions, controls, cases, views, and
reports all come from a backend **adapter module**.

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
  A bootstrap adapter may additionally return `"ready": false` and a
  user-facing `"status": str`. The viewer then shows that status and does
  not request a generation; the generation endpoint independently rejects
  direct requests with a structured JSON `503`. Omitting `ready` preserves
  the normal ready-to-generate behavior.

  Optional top-level presentation metadata is:

  - `default_size`: the size selected initially. It defaults to `256` when
    omitted.
  - `supported_sizes`: a non-empty array of sizes the selector should offer.
    The legacy `128`, `256`, `384`, and `512` choices remain when omitted.

  Each control dict describes one UI input:

  - `name`: identifier (slider label)
  - `ctype`: `"float"` | `"int"` | `"bool"`
  - `default`, `lo`, `hi`: value and range (`lo`/`hi` unused for bool)
  - `tier`: `"primary"` (always visible) | `"advanced"` (collapsed)
  - `invalidates`: `"full"` | `"late"` | `"render"` (see caching below)
  - `promise`: one-line user-facing description of what the control does
  - `temp` (optional): truthy marks the control as a stub/placeholder
  - `enabled` (optional): false locks the control. This is presentation
    metadata only; the adapter must still reject invalid input itself.
  - `disabled_reason` (optional): user-facing explanation shown with a locked
    control.
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
  level `"warn"` or `"fail"` are surfaced as distinct badges.

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

## Optional inspection laboratory

Inspection is a capability distinct from generation. An adapter opts in by
returning all of the following from `meta()`:

- `"inspect_ready": true`;
- `"inspection_status": str`, which describes the available evidence;
- `"inspection_cases": [{"id": str, "title": str, "description": str}, ...]`;
- optional `"inspection_controls"` records whose `control_group` is
  `development` or `render_only`. These are distinct from promised author
  controls; a render-only control may be enabled without implying an engine
  stage exists.

The WebUI then offers an Inspection mode even when `ready` remains false. It
prominently displays the adapter's optional `inspection_banner` and then the
loaded packet's exact `display_label`, hides generator history and generator
inputs, and never calls `/api/generate`. This lets a fixture remain visibly a
fixture while a later real stage identifies its narrower evidence scope
without being mislabeled as generator output. If both capabilities are ready,
a mode selector lets the user choose between them. Inspection controls marked
`enabled: false` remain visible and show `disabled_reason`; a disabled UI
control is not a security or validation boundary.

Inspection uses a lazy, packet-oriented HTTP contract:

1. `POST /api/inspect` with `{"case_id": str}` and optional
   `"render_controls": object` and `"selection": object` returns
   `{"packet_id": str, "case_id": str, "views": [...], "report": object,
   "elapsed_s": number}`.
2. `POST /api/inspect/render` with `{"packet_id": str, "view_id": str,
   "role": str}` returns one `image/png`. Role IDs are opaque values declared
   by that view; the conventional IDs are described below.
   The response is `no-store` and includes `X-Content-SHA256` plus a strong
   ETag for the exact PNG bytes.
3. `POST /api/inspect/promote` explicitly promotes the packet after a human
   confirmation. Its request is `{"packet_id": str, "author": str,
   "question": str, "confirm": true}` plus optional
   `"expected_previous": str`. The response is a JSON receipt. Loading or
   rendering evidence never promotes it.

An inspection-case catalog entry may publish `members` and a
`default_member`. The client represents the chosen persisted member as
`selection: {"member_id": "..."}`. Selection is observation-only: it may
choose evidence already present in one immutable snapshot, but must not rerun
formation, change snapshot identity, or widen promotion scope. Partial/member
packets may report `can_promote: false` while the complete cohort remains the
only promotable scope.

Inspection and generation have separate server paths and caches. A direct
generation request continues to receive a structured `503` while
`ready:false`, regardless of `inspect_ready`. Unknown packets, cases, views,
or roles must be rejected rather than substituted.

### Structured inspection views

Each entry in the inspection response's `views` array has this shape:

```json
{
  "id": "stable.view.id",
  "title": "Readable title",
  "stage": "stage identifier",
  "purpose": "What this view lets the reviewer decide",
  "caption": "Provenance-aware caption",
  "provenance": {},
  "legend": [{"label": "increase", "color": "#4caf80"}],
  "role_order": ["current", "baseline", "delta"],
  "roles": {
    "current": {"label": "Current", "available": true},
    "baseline": {"label": "Accepted baseline", "available": false,
                 "reason": "No accepted baseline."},
    "delta": {"label": "Delta from accepted baseline", "available": false,
              "reason": "No accepted baseline."}
  }
}
```

Legacy views conventionally declare `current`, `baseline`, and `delta`.
Revision-aware views may additionally declare role IDs such as
`revision_reference` and `revision_delta`; a revision reference is not an
accepted baseline. `role_order` controls presentation order and `label`
provides exact user-facing semantics. The client falls back to the legacy
three-role order and labels when those optional fields are absent.

Every applicable role is declared, including unavailable roles, so absence is
never presented as an empty image. `reason` is required when `available` is
false. An unavailable semantic role may declare exactly two `fallback_roles`;
when both are available, the client lazily renders those explicitly labeled
subjects side by side. This is a presentation fallback, not a fabricated
delta. `legend` may be empty; colors, when provided, are CSS colors used only
for a swatch. `provenance` must be JSON-serializable. Extra version, schema,
units, comparison, alignment, and source-field properties are preserved in
the report and may also appear directly on the record.

For backward compatibility, an unavailable legacy `delta` falls back to
`baseline`/`current` only when both are available. When no accepted baseline
exists, neither a baseline image nor a baseline/current comparison is
fabricated. A separately declared non-accepted revision reference may still
be shown under its own honest role and label.

The report may provide `display_label`, `evidence_kind`, `review_question`,
`promotion_scope`, an immutable
`review_event_history`, `accepted_baseline_id`,
`promotion_chain_head_sha256`, `expected_previous`, and `can_promote`.
`promotion_chain_head_sha256` is the preferred compare-and-swap token;
`expected_previous` is its legacy alias. The client never substitutes an
accepted snapshot ID for an event-chain head. It uses this metadata only to
prefill the explicit promotion form; the server remains responsible for
validating scope, disposition, compatibility, stale tokens, and whether a
packet is eligible for its declared baseline namespace.

Every inspection packet must identify its exact evidence kind and limits in
its report and artifacts. Tooling fixtures remain visually unmistakable and
must never be represented as generated evidence; real intermediate-stage
evidence must likewise state what later claims it cannot support. The WebUI's
banner is a second line of defense, not a substitute for backend provenance.

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
the one to open with and to build the history rail from. This convention
applies only to generator output. Inspection fixtures do not invent a
`hypsometric` view.
