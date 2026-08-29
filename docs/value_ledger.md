# Value ledger

Rule (see CLAUDE.md): the project is deliberately over-engineered; trimming
comes later, from evidence. Every feature gets a row. **Predicted yield is
written at implementation time, before any gallery exists.** Features
predicted marginal get same-seed on/off ablation pairs in the next gallery
(ablation = the feature's knob at zero). Observed yield is filled in at
review. The ledger nominates; the author decides. A milestone is not done
until its rows are current.

Verdicts: `keep` | `demote-to-advanced` | `trim-candidate`

| Feature | Milestone | Cost paid | Predicted yield | Observed yield | Verdict |
|---|---|---|---|---|---|
| Weighted-Voronoi plate sizes | M1/C1 | tiny | high — kills uniform-cell look | *(pending author review)* | |
| Plate boundary domain-warp (`plate_raggedness`) | M1/C1 | small | high at boundaries; invisible until C2 paints them | *(pending)* | |
| Euler-pole velocities | M1/C1 | small | none yet — pays off entirely in C2 classification | *(pending)* | |
| Crust kernel domain-warp | M1/C1 | small | high — first pass produced blob continents, warp visibly fixed silhouettes | *(pending author confirm)* | |
| Fine-octave shape noise (0.28 r) | M1/C1 | tiny | medium — suspect coast detail is redundant once C2 elevation noise exists; **ablation pair at M1 review** | *(pending)* | |
| Border-margin warp (`border_irregularity`) | M1/C1 | tiny | medium-high — prevents frame-parallel margins; **ablation pair at M1 review** | *(pending)* | |
| Plate motion arrows (plates view) | M1/C1 | tiny | debug-only; never in author-facing renders | *(pending)* | |
| Proto-elevation | M1/C1 | tiny | scaffolding — dies in C2 by design | trimmed in C2 as scheduled | trimmed |
| FFT source-splat profile painter | M1/C2 | medium | high — the whole boundary grammar rides on it | *(pending)* | |
| Arc–trench gap (offset sources) | M1/C2 | tiny | high — mountains behind a coastal strip, the couplet signature | *(pending)* | |
| Collision plateau term | M1/C2 | tiny | medium — needs continent-continent convergence to occur; **ablation pair at M1 review** | *(pending)* | |
| Rift graben + shoulders | M1/C2 | small | high — graben cuts visible lake-chain lines; shoulders subtle; **shoulder ablation pair** | *(pending)* | |
| Ridge swell (age-proxy blur) | M1/C2 | small | medium — broad basin structure; honest √age deferred to M2 hydro? **ablation pair** | *(pending)* | |
| Slow-spread axial valley | M1/C2 | tiny | predicted marginal below 1024² — σ22km needs fine cells; **ablation pair** | *(pending)* | |
| Along-strike modulation | M1/C2 | tiny | high for ocean arcs (walls → chains), low for cordilleras | *(pending)* | |
| Tanh feature compression | M1/C2 | tiny | high — killed white-out stacking; invisible when working (that's the point) | *(pending)* | |
| Uplift→coastline feedback | M1/R1 | small | high — arc islands gain shelves, ranges push peninsulas; **ablation pair** | *(pending)* | |
| Margin-typed shelf breadth | M1/R1 | small | high — the active/passive asymmetry the reference shows | *(pending)* | |
| Age-law ocean depth (blur proxy) | M1/R1 | small | medium-high — basins gain identity; known limit: rounded young-halos, not ridge-elongated bands (R2 sharpens) | *(pending)* | |
| Tight uplift falloff | M1/R1 | tiny | medium — geography near margin seas restored | *(pending)* | |
| margins debug view | M1/R1 | tiny | debug-only audit tool | *(pending)* | |
| Anisotropic multi-site plates | M1/R2 | small | high — killed disk microplates (visible immediately in plates view) | *(pending author confirm)* | |
| Convex arc bulges (`arc_curvature`) | M1/R2 | small | high — signature oceanward bow; watch for 'horn' peninsulas where bowed couplets meet coasts | *(pending)* | |
| Smooth along-strike modulation | M1/R2 | tiny | high — complete couplets, bounded island-chain gaps; replaced iid randomness | *(pending)* | |
| Ridge segmentation + offsets | M1/R2 | small | medium-high — age bands elongate; staircase subtle below 512²; **ablation pair** | *(pending)* | |
| Fracture-zone scars | M1/R2 | tiny | predicted marginal below 512², faint by design; **ablation pair** | *(pending)* | |
| Crest spines (`crest_sharpness`) | M1/R3 | tiny | medium-high — ranges gain a ridge line; **ablation pair** | *(pending)* | |
| Edged plateaus (smoothstep shaping) | M1/R3 | tiny | medium — flat-top + defined edge vs soft mound | *(pending)* | |
| Outer rise (`outer_rise`) | M1/R3 | tiny | pre-registered marginal below 1024²; **ablation pair** (standing suspicion now testable) | *(pending)* | |
| Compensated arc kernels | M1/R3 | tiny | medium — kills glow-halo around island arcs; net arc height re-tuned +11% | *(pending)* | |
| Era belts (`era_count`) | M1/R3 | medium (2nd partition + pair pass, ~1.5s at 1024²) | high — interiors gain worn ancient ranges; risk: stamped-bar look on unlucky geometry; **ablation pair era_count 2 vs 1** | *(pending)* | |
| Interior provinces (`province_relief`) | M1/R3 | small | high — basins/shields/raised interiors; watch olive-shift of land palette; **ablation pair** | *(pending)* | |
| Tectonic grain (`tectonic_grain`) | M1/R3 | small | medium — subtle at 256, real at 512+; swirl artifacts possible at orientation seams; **ablation pair** | *(pending)* | |
| Coastal complexity (`coast_complexity`) | M1/R3 | tiny | medium-high — coast character variety; weak below 512² by octave budget | *(pending)* | |
| Audit bug fixes 1–4 | M1/C3 | small | correctness, not looks: km-based classification, orientation-uniform amplitudes, fixed height reference, class smoothing | n/a | keep |
| Massif decomposition (width/amp split + jitter) | M1/C3 | small | high — ranges become clumped chains, not pipes; **judge at 512+ in review** | *(pending)* | |
| Abyssal-hill fabric (`seafloor_fabric`) | M1/C3 | small | resolves standing pre-registration; predicted subtle below 512; **ablation shipped** | *(pending)* | |
| Ocean noise raise (90–350 m) | M1/C3 | tiny | medium — deep floor stops reading flat | *(pending)* | |
| Hotspot chains (`hotspot_count`) | M1/C3 | small | high — dotted island trails, visible in ablation | *(pending)* | |
| Arc/rift volcanic marks + `volcanic` layer/view | M1/C3 | tiny | export ingredient + audit view; no direct look | *(pending)* | |
| Jigsaw seas (`rift_maturity`) | M1/C3 | small | high — flooded rift strips with correlated coasts, strongest single tile in the ablation sheet | *(pending author)* | |
| Failed rifts (`failed_rifts`) | M1/C3 | tiny | invisible until M2 rivers find the scars — predicted sleeper | *(pending)* | |
| Back-arc basins (`backarc_basins`) | M1/C3 | tiny | subtle at preview; **ablation shipped** | *(pending)* | |
| PD sweep depression fill | M2/W1 | medium | foundation — no visual yield alone; everything in M2 stands on it | numpy-only, 2.7s @2048², zero sinks | keep |
| Flow accumulation (list-loop) | M2/W1 | small | foundation; the dendritic skeleton | 1.5s @2048² | keep |
| Lakes from fill-to-spill | M2/W1 | small | high — rift lake chains emerge exactly as designed | visible in w1 galleries; author confirm pending | |
| `lake_min_depth_m` control | M2/W1 | tiny | medium — lake abundance is taste | *(pending)* | |
| drainage view | M2/W1 | tiny | debug + review vehicle for all of M2 | *(pending)* | |
| Implicit stream-power carve | M2/W2 | large | the milestone's reason — dissected flanks, drowned valleys, coast detail | combed-flank texture visible at 512+; built-in ablation `erosion_strength=0`; pair shipped in out/w2_carve/ | |
| Channel-initiation threshold | M2/W2 | tiny | high — the difference between uniform lowering (invisible) and differential dissection (visible); learned the hard way this run | *(pending)* | |
| Hillslope diffusion (`hillslope_smoothing`) | M2/W2 | tiny | medium; **ablation pair at review** | *(pending)* | |
| Volcano age split (`volcano_youth`) | M2/W2 | small | medium — fresh cones vs dissected chain tails; judge at 512+; **ablation pair** | *(pending)* | |
| Roughness-by-zone finding (median) | M2/W2 | tiny | audit only — mean-metric false-positived on trench walls, median separates texture from features | deep/land 0.67, info-level | keep |
| Shelf/rise burial (`sediment_softening`) | M2/W3 | small | medium — mutes shelf, builds rise apron; subtle below 512²; **ablation shipped** | *(pending)* | |
| Submarine canyons (`canyon_depth`) | M2/W3 | small | pre-registered scale-dependent: ~1px below 8km cells; judge at 1024+; **ablation shipped** | *(pending)* | |
| Deep-sea fans (`fan_size`) | M2/W3 | tiny | pre-registered subtle: lives in near-black depth band; palette may need a stop; **ablation shipped** | *(pending)* | |
| Head/tail split + late-class cache | M2/W3 | medium | UX, not looks: sea-level drag 0.21s at 512² vs ~4s full | measured | keep |
| Crust-plate affinity (`crust_plate_affinity`) | A1 (post-M2) | small | high — composition-level: continents anchor to plate interiors, so cordillera-coast and collision configurations become common instead of lucky; feeds forward through margin typing/shelves/couplets. Measured: crust-on-continental-plates 0.35→0.89 across the sweep, 0 fallbacks. Watch: at 1.0 landmasses consolidate toward supercontinents (variety drops); default 0.65. **Ablation = knob at 0; sweep shipped in out/a1_affinity/** | *(pending)* | |
| Plate interiority layer + plates-view shading | A1 (post-M2) | tiny | audit/debug — shows the anchor field; analytic sampling doubles as the resolution-independence guarantee | *(pending)* | |
| Canon palette family (`render_palette`, 4 variants) | KR | small | high — carries canon quality 7 (sea-level-dense stops, dark summits, sparse snow, calm abyss); deep-band compression quiets abyssal texture for free. Stops calibrated to measured hypsometry. Ablation = palette 0 (classic) in the same sheet | author picked **canon-soft** as default (KR review, 2026-08-28) | keep |
| Sqrt-space quantize bands | KR | tiny | medium-high — uniform metre bands wasted the budget on deep+highland; sqrt-space matches the ramp philosophy and revives the platform register + banded lowlands at q>0 | *(pending)* | |
| Lake depth-color floor (90 m stop) | KR | tiny | low-medium — lakes stop rendering paler than the shelf; legible blue at every size | *(pending)* | |
| Drowned datum (`flood_rise_m`) | K1 | medium | high, **partially gated on K3**: the carve grades to the lowstand coast, so the shelf is dissected land that drowned. Measured: 3850 km³ shelf incision on the 256² smoke world; platform seas gain interior banks/deeps; canyons now continue real rivers from the lowstand coast. Honest gallery verdict: ria/valley intricacy at map scale is capped by smooth lowlands (the a_c cliff) — full coastal payoff lands when K3 adds worked plains. Ablation flood=0 shipped | *(pending author review)* | |
| Wave planation (`wave_planation`) | K1 | small | medium — lowstand bench + cleaner platform edges (fossil shelf break); cut-only, provably never moves today's coastline (smoke-tested). Subtle at 512²; judge at 1024+. Ablation shipped | *(pending)* | |
| Incise-loop working-set filter | K1 | tiny | perf only: skip never-incising deep floor; erosion 4.3→3.6 s at 1024² (offsets K1's larger active set) | measured | keep |

| Tapered channel initiation (`lowland_dissection`) | K3 | tiny | medium-high — kills the smooth-vs-carved texture cliff; plains gain fine valley marking. Replaces the process-false hard gate. **Ablation = 0 (old gate), sweep shipped** | *(pending)* | |
| Deposition mass balance (`deposition`) | K3 | medium (+~3 s at 1024²) | the process half M2 lacked: carved mass settles in valley floors, closed basins, coastal wedges; conserves and reports (deposited/basin/exported km³). Honest: conservative at defaults — deposited ~4% of eroded, floors subtle at 512²; settling constants (_S_REF, _CAP) are the tuning lever if the author wants pronounced floodplains. **Sweep shipped** | *(pending)* | |
| Plains grain (`plains_grain`) | K3 | small | high — the canon's worked-lowland mottling: tonal interleave on plains, island fields + banks on the flooded platform (K1's deferred payoff arriving), intricate coasts. Side effect: small-lake speckle multiplies (honest fill-to-spill; tune late-class `lake_min_depth_m` to taste). Fixed-km octaves floor at 2.2·cell, so very coarse cells carry less grain by octave budget (documented pattern). **Sweep shipped** | *(pending)* | |
| Diffusion no-flux coast (fix) | K3 | tiny | closes the M2-audit defect: smoothing 0→1 land-fraction drift 0.022 → 0.0053 (smoke-guarded < 0.008). Diffusion can soften a coastal plain, never drag a coastline down | measured | keep |
| Planation ravinement cap (fix) | K1 (amended in K3) | tiny | fixes km-deep crater artifact: bump-vs-blur near cliffs read structure as "interfluve" and shaved thousands of metres (the suspicious deep pockets in K1's lagoon zooms). Cut now capped at 45 m — a physical ravinement thickness | measured (max grain-toggle divergence 7397→709 m) | keep |

| Profile model (saturating height + intensity widening) | K2 | large | the keystone: convergent belts get physical anatomy — saturating crest height (isostatic), width breathing with convergence, apron flanks, rim-enclosed plateau floors past saturation. Anatomy zooms show nested bands, high belt lakes, foreland lake chains — canon quality 1+2 grammar emergent. Replaces prescribed symmetric Gaussians | *(pending author review)* | |
| `plateau_tendency` | K2 | tiny (rides the model) | high — ref14's headline feature: rim-enclosed plateaus with interior lakes; 0 = peaks only. Gates plateau mass ~4× (smoke-verified). **Ablation sweep shipped** | *(pending)* | |
| Foreland basin + apron rows (`outer_rise` re-grounded as flexure) | K2 | small | medium-high — retro-side asymmetry: wide aprons, flexural basin (visible as foreland lake chains after K3 fill). First cut had basins flooding into parallel moat-seas; halved to −0.05·H | *(pending)* | |
| Retired at K2 | K2 | — | whole-stack tanh (`_POS_CAP`) → linear stack + isostatic-ceiling knee (4800 m, 0.22 slope); R3 edged-plateau smoothstep → superseded by fill-row construction; C2 collision-plateau blob (σ190) → superseded. Feedback coefficient retuned 0.75→0.55 to keep land_fraction on target | n/a | trimmed |
| Tier-1 default retune (canon comparison) | post-K (0.7.1) | tiny (defaults only) | The 2026-08-28 k_review-vs-examples comparison found ours over-plated (mean plate footprint ~1,450 km at review extent vs the refs' 2–4 plates per frame) and airbrushed against the refs' stepped chunk. Author authorized tier 1; picks made from out/tier1_defaults/ sweeps: `plate_count` 10→6 (long continuous belts, consolidated masses; 4-seed variety de-risk clean), `render_quantize` 0→12 (canon chunk; bonus: kills summit white-glare, steps the shelf like ref1), `lake_min_depth_m` 0.8→6 (speckle dies, structural belt/rift lakes survive), `deposition` 0.6→0.8 (honest: 0.6→1.0 barely visible at 512² — settling constants stay the tier-2 lever), `plains_grain` 0.5→0.7 (subtle at 512², reads at zoom; no artifacts). Smoke 24/24 green; k_review pack rebuilt at 0.7.1 | picked from sweeps, author-authorized 2026-08-28 | keep |

Perf watch (K1 finding, updated at K3; K2 adds ~+0.5 s in boundaries —
1024² ≈ 24.2 s): 1024² default generation measures
~23.7 s against the 15 s working budget (erosion 7.0 s — the deposit loop
roughly doubles it; sediment 0.8 s). K1 added ~+0.8 s, K3 ~+4 s, the rest
accumulated earlier (relief 4.5 s, crust 3.2 s, boundaries 2.6 s, eras
2.0 s). Nominate a dedicated perf pass after the K-series — the incise +
deposit Python loops are the prime candidates; do not tune stages
mid-rework.
| float32 noise + hoisted sorts | M2/W3 | small | perf only: 1024² 16.4→14.8s (in budget), 2048² 71→63s | measured | keep |

Watch item (W2): D8 45-degree river-segment bias on smooth terrain — expected
to dissolve once erosion entrenches valleys; if it survives the carve, it
becomes a finding + design candidate (D-infinity or flow-path jitter).

Standing pre-registrations (suspicions recorded before building):
- **Automated perceptual-diff impact scores** — meta-feature; possibly
  marginal itself. Build last, if at all.
- *(Outer rise and abyssal-hill fabric graduated to real rows above,
  with ablations shipped in out/m1_review/.)*
