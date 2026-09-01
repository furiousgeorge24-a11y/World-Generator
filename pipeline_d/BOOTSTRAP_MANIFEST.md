# Pipeline C bootstrap manifest

Status: frozen lineage record for Run 1, 2026-09-01.

This file records the one-time bootstrap of the independent land-origin lab.
It is not a runtime dependency list and it does not authorize future code or
design lookup in the source pipeline. After Run 1, work in this directory is
based on Pipeline C's own contract and records.

## What Run 1 was allowed to inherit

The authorized inheritance was deliberately narrow:

- the repository's generator-neutral preview WebUI;
- documentation and control-registry conventions;
- evaluation hygiene such as blind keys, strict schemas, immutable artifacts,
  hashes, calibration arms, and duplicate reliability probes.

All Pipeline C prose, prompts, schemas, registry code, and adapter code were
written for the new land-only contract. No formation model was ported.

## Source record

Hashes are SHA-256 over the exact working-tree bytes consulted at bootstrap.
For the three shared WebUI files edited in place, both the pre-Run-1 and
post-Run-1 hashes are recorded.

| Source | Bootstrap hash | Disposition |
|---|---|---|
| `../webui/README.md` | pre `ad06077cd364dc4e130ef63d9fa1237ac84ffe0906ced259185a41a8c3dd55d0`; post `616224e0d3031e2c69af27ad5d79a3ce0934ede9f25e3ff51f7b00c11ef67e1f` | Shared in place. The optional fail-closed readiness contract and failure badge behavior were documented. |
| `../webui/serve.py` | pre `733b2ba422ceebef2c685e4c0f891ed0058221fddac54fab11c3d449ebf7b668`; post `e2caa68246a1c5a4f72048be4c1988c82bea93bd7224e157dd1c7810b7523744` | Shared in place. An adapter reporting `ready: false` now receives a structured `503` without invoking generation. Existing adapters remain ready by default. |
| `../webui/web/index.html` | pre `fdf805e4517af9ae8d6ea866842ea41c2cde9e3b4beaf157aeb6f963d8e086b0`; post `74f29f2f996ddec726ca848a49a34146da69593bd97120ba51bf6bf98c295e59` | Shared in place. Unready adapters no longer auto-run, errors are readable, and `fail` and `warn` findings are distinct. |
| `../pipeline_b/CONTRACT.md` | `6d36653d3e78a0a55a80434d47ba9b9c1e248d74f5595ac0298d804d48622048` | Behavioral-document structure was consulted. Pipeline C's land-only guarantees were rewritten independently. |
| `../pipeline_b/DESIGN.md` | `e679d80b3eaec706f751be0bd0d8694ef20d9e1c586b55416f6d1a453bbc7e78` | Design-record structure was consulted. No mechanism or parameter value was adopted. |
| `../pipeline_b/EVAL.md` | `b2a477aaf1a3e82bbfbb0394ba2cd3536fa3d333f008877de2e1fed564797831` | Evaluation separation and evidence rules were consulted. C has new land-only instruments. |
| `../pipeline_b/MILESTONES.md` | `c4777c1e30012a0e5b0b569d06b86c8f5fff37cd4414c9dee9ab724b5e8fb10c` | Milestone format only. C starts a new milestone sequence. |
| `../pipeline_b/VALUE_LEDGER.md` | `f2156435f262e270daa45220a1a6be4521acff1206515619bf0d9f7740b2033f` | Ledger discipline only. No feature result was carried forward. |
| `../pipeline_b/ATTEMPT_REGISTER.md` | `ac700e6f00903a654266916263f22a8ccf4f226a19c4a2ff874c8c0669141500` | Register discipline only. No attempt, failure, seed, or conclusion was carried forward. |
| `../pipeline_b/engine/registry.py` | `91d1a2d64789c77a32fe19e6b39e6738b71fa9075acdfe4020fe987e6c31260e` | The generic control-metadata shape was adapted. No old control or engine import was retained. |
| `../pipeline_b/webui_adapter.py` | `7899731528530a7c6f4e79cf2e4495bfd8b9333f8345996a06910be052349b35` | The shared adapter boundary was checked. C's adapter is a new fail-closed implementation with no world path. |
| `../pipeline_b/engine/report.py` | `404aff76eec83f3c4f9db4774965c8f8de7d6c006ff3f4e1f40f0fccf290ab5a` | Reporting conventions were inspected; the terrain-specific report was excluded. |
| `../pipeline_b/eval/build_m3_run.py` | `84b98d8df8e904139025d6daea0975d3b57b5b690bfba485978c40231770dc20` | Atomic publication and hashing concepts were retained; the builder, inputs, and assumptions were excluded. |
| `../pipeline_b/eval/score_run.py` | `e8789ea9504c7c74d65500ba2a9d4ee7ddb579b179df8f9df94063df7c4929c0` | Strict validation and append-only scoring concepts were retained; fixed filenames, trial counts, and verdicts were excluded. |
| `../pipeline_b/tests/eval_checks.py` | `390154896c7d9c3112633774214a912a8f3d1fbf900eaecc80e1aeddc5885e46` | Tamper/no-overwrite test ideas were retained. C tests target only its new schemas and utilities. |

## Explicit exclusions

Run 1 did not port or create any of the following:

- land, plate, crust, elevation, erosion, hydrology, bathymetry, rendering, or
  crop-selection implementation;
- a generated world, placeholder raster, fake report, gallery, or benchmark;
- source-pipeline seeds, output images, prompt IDs, answer keys, thresholds,
  reference identifiers, milestone verdicts, or attempt history;
- a component-count target, border fade, edge mask, target-driven crop search,
  seed retry policy, or any other corrective land-mask operation;
- cache stages or speculative controls for stages that do not exist.

## Pipeline C outputs from this bootstrap

- Self-contained contract, design, evaluation, milestone, value, and attempt
  records.
- A two-control registry and an adapter that advertises `ready: false` and
  raises if called directly.
- New provider-neutral land-origin evaluation prompts, JSON schemas, immutable
  artifact utilities, and their tests.
- A launcher that uses the shared WebUI without copying it.

These outputs establish interfaces and evidence rules only. They are not
evidence that any land-origin requirement is feasible or satisfied.

## Closed boundary after Run 1

Pipeline C may depend at runtime only on its own files and the declared shared
WebUI. Root-level `examples/` may later be used as external perceptual
references only when each chosen asset and hash is frozen into an evaluation
bundle. No Pipeline C Python file may import another pipeline. Future design,
implementation, attempts, and conclusions must be recorded here rather than
looked up in `pipeline_b`.
