"""Control registry (contract section 9).

A control is data, not a function argument. UIs generate themselves from
this table; adding a knob is one entry here. Out-of-range or unknown
inputs are clamped/ignored with a report finding — never an exception
(contract section 8: generation never fails).
"""

from dataclasses import dataclass, asdict

RENDER, LATE, FULL = "render", "late", "full"


@dataclass(frozen=True)
class Control:
    name: str
    ctype: str          # "float" | "int" | "bool"
    default: object
    lo: object          # None for bool
    hi: object
    stage: str
    invalidates: str    # render | late | full
    tier: str           # primary | advanced
    promise: str
    temp: bool = False  # stub-era control; dies with its stub


REGISTRY: list[Control] = [
    Control("cell_size_km", "float", 4.0, 0.5, 50.0, "grid", FULL, "primary",
            "Physical width of one cell. Sets what the map means: small = "
            "region, large = continent/world. Feature placement is in km, "
            "so changing resolution at fixed extent keeps the same world."),
    Control("plate_count", "int", 6, 4, 24, "plates", FULL, "primary",
            "Number of tectonic plates. Sizes vary naturally; boundary "
            "density (future mountain/trench density) scales with count."),
    Control("plate_raggedness", "float", 0.5, 0.0, 1.0, "plates", FULL,
            "advanced",
            "Small-scale irregularity of plate boundaries: 0 smooth arcs "
            "to 1 heavily wandering."),
    Control("continent_count", "int", 4, 1, 10, "crust", FULL, "primary",
            "Major continental nuclei. Landmasses may merge or fragment; "
            "this sets how many cores seed them."),
    Control("crust_plate_affinity", "float", 0.65, 0.0, 1.0, "crust", FULL,
            "primary",
            "How strongly continental cores anchor to the interiors of "
            "continent-carrying plates. 0 places continents blind to the "
            "plate mosaic; 1 pins every core inside a plate, so coasts "
            "organize against boundaries and cordilleras hug convergent "
            "margins. Landmasses still spill across boundaries at every "
            "value — correlation, never coincidence."),
    Control("land_fraction", "float", 0.35, 0.15, 0.65, "crust", FULL,
            "primary",
            "Approximate fraction of the map that is continental platform. "
            "The report states the achieved value; a finding fires beyond "
            "±0.08."),
    Control("continent_irregularity", "float", 0.5, 0.0, 1.0, "crust", FULL,
            "primary",
            "Shape complexity of landmasses: 0 smooth cores to 1 ragged, "
            "fragmented outlines."),
    Control("border_sea_width", "float", 0.08, 0.02, 0.25, "crust", FULL,
            "primary",
            "Typical distance land keeps from the frame, as a fraction of "
            "map size. The outermost ring is water at every value; the "
            "margin sea wanders rather than tracking the frame."),
    Control("border_irregularity", "float", 0.5, 0.0, 1.0, "crust", FULL,
            "advanced",
            "How much the border-sea width wanders around its typical "
            "value."),
    Control("plate_anisotropy", "float", 0.6, 0.0, 1.0, "plates", FULL,
            "advanced",
            "Elongation of plate shapes: 0 rounder cells, 1 slivers and "
            "wedges. Kills circular microplates."),
    Control("arc_curvature", "float", 0.6, 0.0, 1.0, "boundaries", FULL,
            "advanced",
            "How strongly subduction couplets bow toward the subducting "
            "plate — the signature oceanward convexity of island arcs."),
    Control("ridge_segmentation", "float", 0.7, 0.0, 1.0, "boundaries", FULL,
            "advanced",
            "Mid-ocean ridges break into offset spreading segments with "
            "fracture-zone scars running into the old floor; 0 disables."),
    Control("orogeny_strength", "float", 1.0, 0.3, 2.0, "boundaries", FULL,
            "primary",
            "Height of boundary mountain systems — cordilleras, collision "
            "belts, island arcs scale together."),
    Control("trench_depth", "float", 1.0, 0.3, 2.0, "boundaries", FULL,
            "advanced",
            "Depth multiplier for subduction trenches."),
    Control("plateau_tendency", "float", 0.5, 0.0, 1.0, "boundaries", FULL,
            "primary",
            "How readily strong convergence matures into a rim-enclosed "
            "orogenic plateau: crustal thickening saturates at the "
            "isostatic ceiling and spreads — flat high floor, crests "
            "concentrated at the rims, foreland basin beyond. 0 = peaks "
            "only (ablation); 1 = Tibets form at moderate convergence."),
    Control("arc_gap_km", "float", 120.0, 40.0, 300.0, "boundaries", FULL,
            "advanced",
            "Distance from trench to the volcanic arc / cordillera crest "
            "(mountains stand behind a coastal strip, not on the trench "
            "lip)."),
    Control("ridge_swell", "float", 1.0, 0.0, 2.0, "boundaries", FULL,
            "advanced",
            "Depth contrast between young ridge floor and old abyss; "
            "0 flattens ocean basins."),
    Control("shelf_width", "float", 0.5, 0.0, 1.0, "relief", FULL, "primary",
            "Breadth of passive-margin continental shelves. Active "
            "(trench-adjacent) margins stay narrow at every value."),
    Control("province_relief", "float", 0.5, 0.0, 1.0, "relief", FULL,
            "primary",
            "Interior structure of continents: basins, shields, raised "
            "interiors. 0 leaves interiors as plain platform."),
    Control("coast_complexity", "float", 0.55, 0.0, 1.0, "relief", FULL,
            "primary",
            "Coastline raggedness, varying along the map — some stretches "
            "bold and smooth, others intricate. 0 keeps the raw contour."),
    Control("tectonic_grain", "float", 0.6, 0.0, 1.0, "relief", FULL,
            "advanced",
            "Relief texture elongates along nearby range/rift direction "
            "instead of being uniform in all directions."),
    Control("crest_sharpness", "float", 0.6, 0.0, 1.0, "boundaries", FULL,
            "advanced",
            "Sharp ridge spine atop broad mountain bases; 0 leaves ranges "
            "soft-topped."),
    Control("outer_rise", "float", 0.6, 0.0, 1.0, "boundaries", FULL,
            "advanced",
            "Plate flexure under loads: the outer-rise bulge oceanward of "
            "trenches and the foreland basin behind mountain belts. "
            "0 disables both."),
    Control("hotspot_count", "int", 3, 0, 8, "boundaries", FULL, "primary",
            "Volcanic hotspot chains: island/seamount trails aligned with "
            "plate motion, youngest edifice at the head."),
    Control("seafloor_fabric", "float", 0.55, 0.0, 1.0, "relief", FULL,
            "advanced",
            "Abyssal-hill corduroy aligned to the ridge that made the "
            "crust; the deep floor's fine grain."),
    Control("rift_maturity", "float", 0.5, 0.0, 1.0, "boundaries", FULL,
            "advanced",
            "Biases continental rifts along the ladder: valley, lake "
            "chain, narrow flooded sea with correlated facing coasts."),
    Control("failed_rifts", "float", 0.6, 0.0, 1.0, "boundaries", FULL,
            "advanced",
            "Faint linear lowland scars where continental boundaries "
            "sheared without opening; future rivers will find them."),
    Control("backarc_basins", "float", 0.55, 0.0, 1.0, "boundaries", FULL,
            "advanced",
            "Marginal seas opening behind island arcs (arc offshore, "
            "shallow young basin behind, then the mainland)."),
    Control("era_count", "int", 2, 1, 3, "eras", FULL, "advanced",
            "Pseudo-history: ancient plate configurations imprint worn, "
            "blunted orogenic belts inside today's continents. 1 = "
            "current era only."),
    Control("relief_roughness", "float", 0.5, 0.0, 1.0, "relief", FULL,
            "primary",
            "Amplitude of fractal relief detail. Mountains roughen far "
            "more than plains (noise is tectonically modulated)."),
    Control("sea_level_m", "float", 0.0, -120.0, 120.0, "relief", LATE,
            "primary",
            "Fine trim of the waterline across the shelf band — floods "
            "valleys into rias or exposes shelf. Never re-decides where "
            "continents are."),
    Control("flood_rise_m", "float", 250.0, 0.0, 400.0, "erosion", FULL,
            "primary",
            "Post-glacial sea-level rise in metres: how far below today's "
            "sea the rivers graded and the coast formed before drowning. "
            "Higher floods more carved landscape — snakier coasts, rias, "
            "island fields, wider dissected platform seas; 0 = no "
            "lowstand ever existed (built-in ablation). Default 250 is "
            "the author's K1 gallery pick (Earth's value is ~120)."),
    Control("wave_planation", "float", 0.6, 0.0, 1.0, "sediment", FULL,
            "advanced",
            "Wave-base planation of the lowstand shoreline and the "
            "transgressed shelf: shaves interfluve highs into a bench so "
            "the shelf break reads as a crisp fossil coastline. Cut-only; "
            "drowned valleys survive. 0 = off."),
    Control("plains_grain", "float", 0.7, 0.0, 1.0, "relief", FULL,
            "primary",
            "Sub-grid relief on plains and the continental platform — the "
            "worked, mottled texture of processes below grid scale. "
            "Erosional country keeps it; deposition flattens valley "
            "floors back out of it. 0 = billiard-smooth lowlands."),
    Control("deposition", "float", 0.8, 0.0, 1.0, "erosion", FULL,
            "primary",
            "Fraction of carved mass that settles where rivers lose "
            "carrying capacity: valley-floor flats, filled closed basins, "
            "coastal wedges below the lowstand coast. 0 exports "
            "everything to the sea (pre-K3 ablation)."),
    Control("lowland_dissection", "float", 0.5, 0.0, 1.0, "erosion", FULL,
            "advanced",
            "How much weak incision continues below the channel-initiation "
            "catchment: plains gain fine valley marking instead of a hard "
            "smooth-vs-carved cliff. 0 restores the hard threshold."),
    Control("erosion_strength", "float", 0.7, 0.0, 1.5, "erosion", FULL,
            "primary",
            "Depth of fluvial dissection. 0 = the uncarved skeleton "
            "(built-in ablation); around 1 = fully developed valley "
            "networks."),
    Control("erosion_steps", "int", 6, 0, 10, "erosion", FULL, "advanced",
            "Erosion iterations — networks entrench further with more "
            "steps; generation time scales roughly linearly."),
    Control("hillslope_smoothing", "float", 0.4, 0.0, 1.0, "erosion", FULL,
            "advanced",
            "Hillslope diffusion between river cuts: 0 keeps walls raw "
            "and angular, 1 softens valley sides."),
    Control("volcano_youth", "float", 0.4, 0.0, 1.0, "erosion", FULL,
            "advanced",
            "Fraction of each hotspot chain placed after the carve: "
            "fresh sharp cones vs. dissected old massifs."),
    Control("sediment_softening", "float", 0.6, 0.0, 1.0, "sediment", FULL,
            "advanced",
            "How buried the sea floor looks: mutes shelf texture (drowned "
            "valleys stay as ghosts) and builds the continental-rise "
            "apron. 0 leaves the raw floor."),
    Control("fan_size", "float", 1.0, 0.0, 2.0, "sediment", FULL, "advanced",
            "Deep-sea fan mounds where major rivers reach deep water; "
            "0 removes them. Deposition never breaches the surface."),
    Control("canyon_depth", "float", 1.0, 0.0, 2.0, "sediment", FULL,
            "advanced",
            "Submarine canyons notching the slope opposite major river "
            "mouths; 0 removes them."),
    Control("lake_min_depth_m", "float", 6.0, 0.2, 20.0, "hydrology", LATE,
            "advanced",
            "Filled depressions shallower than this render as dry ground; "
            "deeper ones surface as lakes. Never changes drainage, only "
            "what counts as visible water."),
    Control("render_quantize", "int", 12, 0, 24, "render", RENDER, "advanced",
            "0 = smooth hypsometric ramp; N = N discrete bands each for "
            "land and water. Render-only: never re-simulates."),
    Control("render_palette", "int", 2, 0, 3, "render", RENDER, "primary",
            "Hypsometric palette: 0 classic (v0.4 ramp), 1 canon (color "
            "budget dense near sea level, summits darken to near-black "
            "with sparse snow caps, calm compressed abyss), 2 canon-soft "
            "(gentler darks, lighter deeps — author default, KR review), "
            "3 canon-crisp (hard shelf-break knee, contrastier lowland "
            "bands). Render-only: never re-simulates."),
]

_BY_NAME = {c.name: c for c in REGISTRY}


def as_dicts() -> list[dict]:
    return [asdict(c) for c in REGISTRY]


def resolve(overrides: dict | None) -> tuple[dict, list[dict]]:
    """Full control dict from partial overrides. Returns (values, findings)."""
    values = {c.name: c.default for c in REGISTRY}
    findings: list[dict] = []
    for key, raw in (overrides or {}).items():
        c = _BY_NAME.get(key)
        if c is None:
            findings.append({"check": "controls", "level": "warn",
                             "msg": f"unknown control {key!r} ignored"})
            continue
        try:
            val = {"float": float, "int": int, "bool": bool}[c.ctype](raw)
        except (TypeError, ValueError):
            findings.append({"check": "controls", "level": "warn",
                             "msg": f"{key}={raw!r} unparseable; default kept"})
            continue
        if c.ctype != "bool" and c.lo is not None:
            clamped = min(max(val, c.lo), c.hi)
            if clamped != val:
                findings.append({"check": "controls", "level": "warn",
                                 "msg": f"{key}={val} clamped to {clamped}"})
                val = clamped
        values[key] = val
    return values, findings


def invalidation_of(name: str) -> str:
    return _BY_NAME[name].invalidates if name in _BY_NAME else FULL
