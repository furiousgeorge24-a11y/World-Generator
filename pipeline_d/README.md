# Pipeline C — land-origin laboratory

Pipeline C is a focused, self-contained experiment for one unresolved
problem: generating plausibly natural land origins while giving an author
reliable control over how much land is delivered and how strongly that land
tends to fragment.

It is not a complete terrain pipeline. Mountains, detailed bathymetry,
erosion, rivers, lakes, climate, biomes, settlements, names, and finished
cartographic styling are outside this module's scope. A successful module is
intended to expose a small, documented interface that could later be ported
into a larger system after a separate author decision.

## Current state

Run 1 completed the **M0 bootstrap**. It establishes the contract, architecture
obligations, evaluation protocol, evidence bookkeeping, shared-WebUI adapter
boundary, and quarantine record. Run 1 contains **no land-generation model,
generated map, candidate result, or empirical success claim**.

The first model-building work belongs to M1. M0's completion is an interface
and laboratory result, not evidence that a land model is feasible.

## Bootstrap usage

From the repository root, `pipeline_c\run.bat` starts the shared WebUI on
port `5002`. During Run 1 it intentionally reports that the engine is
unavailable: generation fails closed and produces no map or placeholder
output.

Run the bootstrap checks with:

```powershell
python pipeline_c/tests/bootstrap_checks.py
```

Run the evaluation-infrastructure checks with:

```powershell
python pipeline_c/tests/eval_checks.py
```

## Author controls to be solved

- `target_land_percent`: `0` through `70`, default `35`. Every delivered map
  must land within 10 percentage points of the request.
- `landmass_fragmentation`: continuous `0` through `1`, default `0.5`. At
  `0`, where enough land exists for the idea to be meaningful, the process
  should strongly tend toward one dominant macro-landmass. Small boundary,
  barrier, coastal, and volcanic islands remain valid. Higher values shift
  the same approximate land budget toward more separated major bodies; the
  control never promises an island count.

The same seed and map geometry use the same latent randomness and the same
delivered window across both control sweeps. Neither control may obtain its
result by rerolling seeds, moving the crop, editing a finished mask, or using
the delivered frame to shape terrain.

Both controls are advertised now to freeze the eventual interface, but the
adapter remains `ready: false`. It may not become ready while either control
is unimplemented or silently ignored. In particular, target-only M1 work does
not make the shared UI a working generator; readiness can first be considered
after M2 implements and evaluates fragmentation.

## Documentation authority

1. [`CONTRACT.md`](CONTRACT.md) defines what every conforming result must
   satisfy and which causal shortcuts are prohibited.
2. [`EVAL.md`](EVAL.md) defines how those promises are tested. On conflict,
   the contract wins.
3. [`DESIGN.md`](DESIGN.md) records architecture obligations and hypotheses;
   it does not ratify a land-formation mechanism during M0.
4. [`MILESTONES.md`](MILESTONES.md) defines sequencing and exit criteria.
5. [`VALUE_LEDGER.md`](VALUE_LEDGER.md) will track the cost and demonstrated
   yield of mechanisms once any exist.
6. [`ATTEMPT_REGISTER.md`](ATTEMPT_REGISTER.md) will preserve every executed,
   rejected, abandoned, or superseded model attempt.
7. [`BOOTSTRAP_MANIFEST.md`](BOOTSTRAP_MANIFEST.md) records the one-time
   inheritance boundary and shared dependencies.

## Isolation rule

The bootstrap is the sole authorized inheritance event. After it closes,
Pipeline C is developed from its own documentation, code, evidence, and
author rulings. Its only runtime shared repository dependency is the root
WebUI shell. Root reference images may be used only as external perceptual
evidence after the chosen assets and hashes are frozen into an evaluation
bundle, as recorded in the bootstrap manifest. Pipeline C may not import from
or consult `pipeline_b` during later development.
