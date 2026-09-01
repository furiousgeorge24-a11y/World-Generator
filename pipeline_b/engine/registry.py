"""Controls as data (§13). Promises are worded in process terms; every
promise must hold across its whole range. All M1 controls invalidate
the full generation (no cached tiers yet).
"""

from . import VERSION
from .tectonics import Config

CONTROLS = [
    dict(name="plates", ctype="int", default=7, lo=4, hi=10,
         tier="primary", invalidates="full",
         promise="number of rigid plates the lithosphere breaks into"),
    dict(name="continental_budget", ctype="float", default=0.30,
         lo=0.15, hi=0.45, tier="primary", invalidates="full",
         promise="fraction of the frame's crust area seeded as "
                 "continental nuclei (upper bound of eventual land)"),
    dict(name="nuclei", ctype="int", default=3, lo=2, hi=6,
         tier="advanced", invalidates="full",
         promise="number of cratonic nuclei sharing the continental "
                 "budget"),
    dict(name="plate_speed", ctype="float", default=45.0, lo=10.0,
         hi=100.0, tier="primary", invalidates="full",
         promise="mean plate drift in km per 25 Myr era; scales all "
                 "kinematic vigor (0 would freeze the world)"),
    dict(name="eras", ctype="int", default=20, lo=8, hi=32,
         tier="advanced", invalidates="full",
         promise="tectonic eras simulated; more eras = longer history, "
                 "more collision/rift record"),
    dict(name="wander", ctype="float", default=0.08, lo=0.0, hi=0.2,
         tier="advanced", invalidates="full",
         promise="per-era drift-direction wander in radians; 0 keeps "
                 "plate courses fixed"),
    dict(name="world_margin", ctype="float", default=0.45, lo=0.2,
         hi=0.6, tier="advanced", invalidates="full",
         promise="world simulated beyond the delivered frame on each "
                 "side, as a fraction of frame width (must exceed total "
                 "plate drift or rim crust nears the frame)"),
    # --- M2 surface stage (never reshuffles structure: §4)
    dict(name="hydrosphere_depth", ctype="float", default=4930.0,
         lo=4500.0, hi=5400.0, tier="primary", invalidates="full",
         promise="planetary water inventory as mean depth (m) if spread "
                 "over the whole world; more water raises the eustatic "
                 "stand and floods margins further inland"),
    dict(name="orogeny_height", ctype="float", default=4000.0,
         lo=0.0, hi=6000.0, tier="primary", invalidates="full",
         promise="elevation scale (m) of young orogenic crust "
                 "thickening; saturates with accumulated shortening, "
                 "decays with belt age; 0 leaves belts flat"),
    dict(name="detail_amplitude", ctype="float", default=1.0,
         lo=0.0, hi=2.0, tier="primary", invalidates="full",
         promise="scales sub-grid relief texture everywhere; per-zone "
                 "amplitude is still set by local process state (rough "
                 "land, muted shelf, quiet deep); 0 turns texture off"),
    dict(name="passive_shelf_km", ctype="float", default=320.0,
         lo=140.0, hi=480.0, tier="advanced", invalidates="full",
         promise="width (km) of the stretched-crust subsidence zone on "
                 "passive margins; sets how broad trailing-coast "
                 "shelves flood (active margins stay narrow)"),
    # --- M3 surface-process stage ("late": reruns only erosion + the
    # output sampling, never tectonics — §4 staged controls)
    dict(name="erosion_time", ctype="float", default=20.0,
         lo=0.0, hi=40.0, tier="primary", invalidates="late",
         promise="duration (Myr) of the recent fluvial-erosion window: "
                 "valley carving, floodplains, and sediment delivery "
                 "all scale with it; 0 turns the surface-process stage "
                 "off"),
    dict(name="erodibility", ctype="float", default=1.0,
         lo=0.3, hi=3.0, tier="primary", invalidates="late",
         promise="global rock-erodibility scale for stream-power "
                 "incision (belt rock stays relatively harder; local "
                 "heterogeneity rides on top)"),
    dict(name="soil_creep", ctype="float", default=1.0,
         lo=0.0, hi=2.0, tier="advanced", invalidates="late",
         promise="multiplier on the effective hillslope soil-creep "
                 "diffusivity (1 = calibrated 8.8 km^2/Myr); keeps "
                 "graded exposed slopes curved between channels; 0 "
                 "turns creep off"),
    dict(name="lowstand_drop", ctype="float", default=80.0,
         lo=0.0, hi=150.0, tier="advanced", invalidates="late",
         promise="sea-level lowstand (m below present) that rivers cut "
                 "against before the flood-back — deeper lowstand "
                 "means stronger drowned-valley and shelf-channel "
                 "record; 0 disables the lowstand"),
    dict(name="deposition_length", ctype="float", default=180.0,
         lo=60.0, hi=400.0, tier="advanced", invalidates="late",
         promise="e-folding travel (km) of river sediment across the "
                 "seafloor: short piles thick nearshore wedges, long "
                 "spreads far-traveled fans"),
    dict(name="river_density", ctype="float", default=0.6,
         lo=0.0, hi=1.0, tier="primary", invalidates="render",
         promise="how far down the computed discharge ranking rivers "
                 "are drawn on the map (render-only; regenerates "
                 "nothing)"),
]


def meta():
    return {"name": "pipeline_b", "version": VERSION,
            "controls": CONTROLS}


def make_config(controls):
    cfg = Config()
    for c in CONTROLS:
        v = controls.get(c["name"], c["default"])
        v = int(v) if c["ctype"] == "int" else float(v)
        lo, hi = c["lo"], c["hi"]
        setattr(cfg, c["name"], min(max(v, lo), hi))
    return cfg


def effective_controls(controls=None):
    """Full, type-normalized, range-clamped control echo for provenance."""
    cfg = make_config(controls or {})
    return {c["name"]: getattr(cfg, c["name"]) for c in CONTROLS}
