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
    Control("stub_feature_scale_km", "float", 600.0, 100.0, 3000.0,
            "stub_relief", FULL, "primary",
            "STUB (dies in C1): wavelength of placeholder terrain.",
            temp=True),
    Control("stub_relief_amp_m", "float", 2600.0, 500.0, 8000.0,
            "stub_relief", FULL, "primary",
            "STUB (dies in C1): amplitude of placeholder terrain.",
            temp=True),
    Control("stub_land_bias", "float", 0.0, -1.0, 1.0,
            "stub_relief", FULL, "primary",
            "STUB (dies in C1): shifts placeholder land fraction.",
            temp=True),
    Control("render_quantize", "int", 0, 0, 24, "render", RENDER, "advanced",
            "0 = smooth hypsometric ramp; N = N discrete bands each for "
            "land and water. Render-only: never re-simulates."),
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
