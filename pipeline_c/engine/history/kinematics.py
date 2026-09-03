"""The history loop: drive, solve, strain, damage, heal, advect, repeat.

Nothing in here draws a plate or a boundary. The loop moves a strength field
around and lets it fail where the velocity it produces pulls hardest; what
survives is read off afterwards by `plates.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain import prolong_bilinear, sample_bilinear_periodic
from ..geometry import WorldGeometry
from ..noise import periodic_noise
from ..sampler import StageSampler
from .constants import (
    DAMAGE_RATE,
    INTACT_SPREAD_CLIP,
    DRIVE_ROT_RATIO,
    DRIVE_WAVELENGTH_KM,
    EPOCH_FRACTIONS,
    HEAL_RATE,
    HISTORY_MYR,
    HOMOG_LENGTH_FRACTION,
    MG_MAX_CYCLES,
    STAGE_ID,
    STAGE_VERSION,
    SOLVE_GRID_DIVISOR,
    STEP_MYR,
    STRENGTH_EXPONENT,
    STRENGTH_INIT_MEAN,
    STRENGTH_INIT_SPREAD,
    STRENGTH_MIN,
    STRENGTH_OCTAVES,
    STRENGTH_WAVELENGTH_KM,
    SUTURE_STRENGTH,
    WEAK_THRESHOLD,
)
from . import _c04_5 as c04_5_seams
from . import markers as marker_seams
from . import rigid
from .drive import Drive, build_drive
from .seams import (
    advect_nearest,
    damage_excess,
    intact_strength_field,
    nucleate,
    seam_mask,
    tip_pass,
    tip_pass_continuous,
    tips,
)
from .solver import (
    effective_gradients,
    kappa0_for,
    prolong,
    restrict,
    restrict_kappa,
    solve,
)

STRENGTH_PROCESS_ID = "strength-initial"

#: Steps whose end-of-step strength is kept for the build report's
#: time-lapse. Not views, not a field-shaping constant: nothing in the
#: loop reads them.
EARLY_SNAPSHOT_STEPS = (2, 4, 8, 12, 16)


@dataclass(frozen=True, slots=True)
class HistoryParams:
    """One run's settings, as a record. `constants.py` holds the defaults.

    Production passes no params and therefore runs on the constants. The
    exploration lab on port 5003 builds one of these from its dials; the
    dials are development instruments and whatever they find is frozen back
    into `constants.py` rather than shipped as a control.

    `seams` swaps the damage rule and, at 2, what a piece is. At 0 the sheet
    damages every cell whose load exceeds the yield, which is what the engine
    has always done and what production runs. At 1 the seam formulation of
    `DESIGN.md` §3.6 runs instead: damage only on a seam, at a seam's tip, or
    at a nucleation site, on the sheet's own velocity solve. At 2 the block
    model of that section's last paragraph runs: pieces are rigid bodies
    solved three unknowns at a time and coupled through seam tractions, the
    stress the seam rules read is the sheet solve of the drag a piece fails
    to match rather than the sheet's own, and the seam network is carried on
    markers that cannot duplicate. The seam rules themselves — tips, the
    Griffith threshold, nucleation, slip-rate damage, healing — are the same
    at 1 and 2; what changes is the velocity they act on, the stress they
    read, how a seam moves, and — since `WORK_ORDER_C04_5.md` §1 — how a tip
    picks its direction. At 1 a tip scores its eight neighbours and steps
    into the best; at 2 it reads a continuous direction off the stress and
    walks from its markers' own position, so a crack loaded between two
    lattice directions can run between them. See `seams.tip_pass` and
    `seams.tip_pass_continuous`.

    `work_damage` is consulted under all three rules and means the same thing
    — which excess a damaging cell damages by — but under `seams` it picks
    between two laws that differ in kind on a seam: at `0`, the default, the
    strain-rate excess, under which a slipping seam stays weak and heals only
    when it stops; at `1` the work excess, under which an open seam
    dissipates almost nothing and heals shut. See `seams.damage_excess`. At
    `seams = 2` the strain rate a seam cell reads is its own slip rate — the
    rigid jump between the pieces it links plus the elastic strain rate of the
    internal-stress solution at that cell, so a crack inside a piece slips
    too — and the power is `kappa(S) * slip ** 2`; an intact cell is rigid,
    has no strain rate of its own and never damages.

    `crack_speed_km_per_myr`, `nucleations_per_step` and
    `toughness_fraction` are read only under `seams = 1` and `2`, and are
    recorded whatever the rule so a run's record says what every dial was set
    to whether or not the rule read it. `toughness_fraction` is the fracture
    toughness as a fraction of the intact strength: a tip propagates at
    `toughness_fraction * sigma_c / sqrt(L)`, so at the default `1.0` a crack
    one cell long propagates at exactly the stress it takes to nucleate one,
    which is where `WORK_ORDER_C04.md` §2.4 left it, and below 1.0 it
    propagates at less. Nucleation reads the full intact strength whatever
    this is set to.
    """

    stiffness_fraction: float = HOMOG_LENGTH_FRACTION
    yield_percentile: float = 12.0
    heal_time_myr: float = 1.0 / HEAL_RATE
    damage_time_myr: float = 1.0 / DAMAGE_RATE
    work_damage: int = 0
    seams: int = 0
    crack_speed_km_per_myr: float = 40.0
    nucleations_per_step: int = 2
    toughness_fraction: float = 1.0
    strength_exponent: int = STRENGTH_EXPONENT
    strength_spread: float = STRENGTH_INIT_SPREAD
    drive_wavelength_km: float = DRIVE_WAVELENGTH_KM
    drive_shear: float = DRIVE_ROT_RATIO
    history_myr: float = HISTORY_MYR
    max_cycles: int = MG_MAX_CYCLES
    solve_divisor: int = SOLVE_GRID_DIVISOR

    def __post_init__(self) -> None:
        for name, low, high in (
            ("stiffness_fraction", 0.02, 4.0),
            ("yield_percentile", 0.5, 50.0),
            ("heal_time_myr", 5.0, 2000.0),
            ("damage_time_myr", 0.5, 200.0),
            ("work_damage", 0, 1),
            ("seams", 0, 2),
            ("crack_speed_km_per_myr", 0.0, 400.0),
            ("nucleations_per_step", 0, 20),
            ("toughness_fraction", 0.05, 1.0),
            ("strength_exponent", 1, 8),
            ("strength_spread", 0.0, 0.3),
            ("drive_wavelength_km", 100.0, 100000.0),
            ("drive_shear", 0.0, 2.0),
            ("history_myr", 50.0, 1000.0),
            ("max_cycles", 5, 200),
            ("solve_divisor", 1, 2),
        ):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a number, not a bool")
            if name in ("work_damage", "seams", "nucleations_per_step",
                        "strength_exponent", "max_cycles", "solve_divisor"):
                if not isinstance(value, int):
                    raise ValueError(f"{name} must be an integer")
            elif not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")

    @property
    def heal_rate(self) -> float:
        """1/Myr. A fault seals in `heal_time_myr` once it stops moving."""
        return 1.0 / self.heal_time_myr

    @property
    def damage_rate(self) -> float:
        """1/Myr at twice yield, where the excess-squared factor is one."""
        return 1.0 / self.damage_time_myr

    @property
    def steps(self) -> int:
        """Steps of `STEP_MYR` that cover the history."""
        return round(self.history_myr / STEP_MYR)

    def to_record(self) -> dict[str, float | int]:
        return {
            "stiffness_fraction": float(self.stiffness_fraction),
            "yield_percentile": float(self.yield_percentile),
            "heal_time_myr": float(self.heal_time_myr),
            "damage_time_myr": float(self.damage_time_myr),
            "work_damage": int(self.work_damage),
            "seams": int(self.seams),
            "crack_speed_km_per_myr": float(self.crack_speed_km_per_myr),
            "nucleations_per_step": int(self.nucleations_per_step),
            "toughness_fraction": float(self.toughness_fraction),
            "strength_exponent": int(self.strength_exponent),
            "strength_spread": float(self.strength_spread),
            "drive_wavelength_km": float(self.drive_wavelength_km),
            "drive_shear": float(self.drive_shear),
            "history_myr": float(self.history_myr),
            "max_cycles": int(self.max_cycles),
            "solve_divisor": int(self.solve_divisor),
        }


DEFAULT_PARAMS = HistoryParams()


@dataclass(slots=True)
class Epoch:
    """One kept step. The fields are the step's, not the end state's.

    `strength` is the end of the step, after damage, cracking and motion;
    everything else is the step's own solve, which is the convention the
    velocity and the strain rate have always followed here.

    The last four exist only under `seams = 2`. `mismatch` is the magnitude
    of the drag a piece failed to match, which is what the internal-stress
    solve is forced with. `piece_labels`, `piece_centroid` and
    `piece_velocity` are the rigid bodies the step's solve moved and how each
    one moved, which is what the `pieces_motion` view draws. Under the sheet
    and under `seams = 1` there are no rigid pieces: `mismatch` is a flat
    zero field and the other three are `None`.
    """

    t_myr: float
    strength: np.ndarray       # (n, n)
    velocity: np.ndarray       # (2, n, n) km/Myr
    strain_rate: np.ndarray    # (n, n) 1/Myr
    power: np.ndarray          # (n, n) stiffness x strain rate squared
    divergence: np.ndarray     # (n, n) 1/Myr
    stress: np.ndarray         # (n, n) stiffness x strain rate, bilinear
    mismatch: np.ndarray | None = None        # (n, n) |D - u| on intact cells
    piece_labels: np.ndarray | None = None    # (n, n) int32, -1 on seams
    piece_centroid: np.ndarray | None = None  # (2, N) cell units
    piece_velocity: np.ndarray | None = None  # (2, N) km/Myr


@dataclass(slots=True)
class History:
    geometry: WorldGeometry
    params: HistoryParams
    drive: Drive
    strength_initial: np.ndarray
    epochs: list[Epoch]
    early: list[tuple[int, float, np.ndarray]]
    weak_fraction: list[float]
    seam_fraction: list[float]
    tip_count: list[int]
    nucleation_count: list[int]
    advance_count: list[int]
    #: The block model's record, one entry per step, and zero at every step
    #: under the sheet and under `seams = 1`, which have no rigid pieces.
    #: `piece_count`, `largest_piece_share`, `wrapping_pieces` and the two
    #: residuals are the pieces the step's own rigid solve moved, read off
    #: the raster the step started from; `marker_count` and `gaps_closed`
    #: are the end of the step.
    piece_count: list[int]
    largest_piece_share: list[float]
    second_piece_cells: list[int]
    force_residual_max: list[float]
    torque_residual_max: list[float]
    wrapping_pieces: list[int]
    marker_count: list[int]
    gaps_closed: list[int]
    #: What the curve of `WORK_ORDER_C04_6.md` §1 did, per step and summed
    #: over the step's passes, and zero at every step under the sheet and
    #: under `seams = 1`. `meetings` counts the advances that landed within
    #: `MEETING_RADIUS_CELLS` of another chain and linked to it, which is a
    #: crack stopping where it met another; `subdivisions` counts the edges
    #: the move stretched past `SEGMENT_MAX_CELLS` and the step split at
    #: their midpoints; `suture_markers` is how many markers ended the step
    #: at or above `WEAK_THRESHOLD` and below `SUTURE_STRENGTH`, remembered
    #: vertices on intact cells; `reactivations` is how many crossed back
    #: below `WEAK_THRESHOLD`, a remembered curve reopening rather than a new
    #: crack; `sample_count` is how many points the end-of-step raster drew.
    #: `degenerate_tips` counts the tips whose averaged stress tensor had its
    #: two absolute eigenvalues within one per cent of each other, where the
    #: direction is whatever the eigensolver returns.
    meetings: list[int]
    subdivisions: list[int]
    suture_markers: list[int]
    reactivations: list[int]
    sample_count: list[int]
    degenerate_tips: list[int]
    #: Which part of the slip carried the damage, one entry per step and zero
    #: at every step under the sheet and under `seams = 1`.
    #: `seam_yield_share` is the share of the step's markers standing on a
    #: cell whose slip rate exceeded the yield; `elastic_slip_mean` and
    #: `rigid_slip_mean` are the two parts of the slip averaged over the seam
    #: cells of the raster the step damaged.
    seam_yield_share: list[float]
    elastic_slip_mean: list[float]
    rigid_slip_mean: list[float]
    exceed_fraction: list[float]
    strength_mean: list[float]
    strength_min: list[float]
    strain_rate_mean: list[float]
    strain_rate_max: list[float]
    solver_cycles: list[int]
    solver_residual: list[float]
    steps: int
    step_myr: float
    history_myr: float
    yield_strain_per_myr: float
    yield_power: float
    sigma_c: float
    sigma_c_field: np.ndarray

    @property
    def power_yield(self) -> float:
        """The C03.8 power yield, under the name `DESIGN.md` §3.6 uses.

        One number, read at step 1 whatever the damage rule, so there is one
        field and not two that could disagree.
        """
        return self.yield_power


def strength_noise(geometry: WorldGeometry) -> np.ndarray:
    """The one heterogeneity field of the lithosphere, at unit amplitude.

    The noise band is `STRENGTH_WAVELENGTH_KM` in kilometres and there is no
    dial for it; `strength_spread` is its only control and it scales this
    field. The sheet adds it to the initial strength; the seam formulation
    starts from an intact sheet and multiplies the intact yield stress by it
    instead, so the same pattern means the same thing under both rules — this
    is where a crack finds it easier to start and easier to run.
    """
    sampler = StageSampler(geometry.world_id, STAGE_ID, STAGE_VERSION,
                           STRENGTH_PROCESS_ID)
    return periodic_noise(sampler, geometry, channel=0,
                          wavelength_km=STRENGTH_WAVELENGTH_KM,
                          octaves=STRENGTH_OCTAVES)


def initial_strength(geometry: WorldGeometry,
                     spread: float = STRENGTH_INIT_SPREAD) -> np.ndarray:
    """High everywhere, with the mild heterogeneity the design asks for.

    `spread` is the standard deviation of the heterogeneity about
    `STRENGTH_INIT_MEAN`; the noise field itself is unchanged by it, so a run
    at spread zero starts from a uniform sheet and a run at any spread has the
    same pattern with a different amplitude. The default is the constant, so
    production is unaffected.

    The noise band is `STRENGTH_WAVELENGTH_KM` in kilometres and there is no
    dial for it: `spread` is the strength noise's only control.
    """
    return np.clip(STRENGTH_INIT_MEAN + float(spread) * strength_noise(geometry),
                   STRENGTH_MIN, 1.0)


def solve_n_for(history_n: int,
                divisor: int = SOLVE_GRID_DIVISOR) -> int:
    """Cells per axis of the grid the velocity is solved on.

    `divisor` is `HistoryParams.solve_divisor`; `SOLVE_GRID_DIVISOR` is its
    default and the only value production uses. At 1 the solve grid is the
    kinematic grid and this is the identity.
    """
    solve_n = history_n // divisor
    if solve_n * divisor != history_n:
        raise ValueError("the history grid must divide by the solve divisor")
    return solve_n


def to_solve_grid(kappa: np.ndarray, traction: np.ndarray,
                  solve_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Coarsen the coefficients and the forcing onto the solve grid.

    `kappa` goes down by the solver's own harmonic 2 x 2 mean with the quarter
    factor, so a thin weak line stays a barrier and the diffusivity keeps its
    meaning in the wider cell. The forcing goes down by a plain 2 x 2 mean.

    The loop is on the shapes, so at `solve_divisor = 1` the solve grid is the
    kinematic grid, no coarsening happens, and this is the identity.
    """
    while kappa.shape[-1] > solve_n:
        kappa = restrict_kappa(kappa)
        traction = restrict(traction)
    return kappa, traction


def to_kinematic_grid(velocity: np.ndarray, history_n: int) -> np.ndarray:
    """Interpolate a solved velocity back onto the kinematic grid.

    The loop is on the shapes, so at `solve_divisor = 1` this is the identity.
    """
    while velocity.shape[-1] < history_n:
        velocity = prolong_bilinear(velocity)
    return velocity


def to_kinematic_blocks(field: np.ndarray, history_n: int) -> np.ndarray:
    """Lift a solve-grid field onto the kinematic grid, block by block.

    The solver's piecewise-constant `prolong`, applied as many times as
    `to_kinematic_grid` interpolates. It must stay piecewise constant: a
    bilinear lift would put intermediate strain into the strong cells beside a
    failed one, which is the halo this discretization removes.

    The block factor comes from the two shapes and not from the constant, so
    at `solve_divisor = 1` the field is already on the kinematic grid and this
    is the identity.
    """
    while field.shape[-1] < history_n:
        field = prolong(field)
    return field


def to_kinematic_smooth(field: np.ndarray, history_n: int) -> np.ndarray:
    """Lift a solve-grid scalar onto the kinematic grid by interpolation.

    **Why there are two lifts of the same stress field.**
    `to_kinematic_blocks` is the lift damage rides: the strain a cell carries
    is the stress it carries divided by its own stiffness, resolved on the
    solve grid, and a bilinear lift would put intermediate strain into the
    strong cells beside a failed one, which is the halo the block lift
    removes. But the seam rule does not integrate a rate, it *chooses*: which
    of eight directions a tip runs in, and which intact cells are the
    highest-stressed in the world. Read off a block lift, both choices see a
    field that is constant over 2 x 2 kinematic cells, so a tip cannot tell
    its neighbour from its neighbour's neighbour and nucleation ranks four
    cells equal. This lift is for the choosing; the block lift is for the
    rate; each is used where it belongs and neither replaces the other.

    At `solve_divisor = 1` the two grids are the same one and this is the
    identity, as `to_kinematic_blocks` is.
    """
    while field.shape[-1] < history_n:
        field = prolong_bilinear(field)
    return field


def epoch_steps(steps: int) -> list[int]:
    """The 1-based step indices whose end state is kept, sorted and unique."""
    return sorted({round(fraction * steps) for fraction in EPOCH_FRACTIONS})


def default_steps(history_myr: float = HISTORY_MYR) -> int:
    return round(history_myr / STEP_MYR)


def run_history(geometry: WorldGeometry, *, params: HistoryParams | None = None,
                steps: int | None = None,
                _trace: list | None = None,
                _c04_5: bool = False) -> History:
    """Run the kinematic history and keep the epochs the views need.

    `params` is one `HistoryParams`; omitting it runs on `constants.py`, which
    is what production does. `steps` overrides the step count the params imply
    and exists so a test can run a short history over the same drive schedule.

    `_trace` is bookkeeping and changes no field: given a list, the block
    model appends one record per step — the seam mask the step ended with,
    the cells the raster drew into at all, how many markers each cell held
    before healing, how many of those healing removed, the step's slip rate,
    and every marker's position on either side of the move — which is what
    `WORK_ORDER_C04_5.md` §2 needs to say whether a cut piece rejoined through
    a cell that healed or through one its markers left, and to trace one such
    cell marker by marker. It is off by default and costs nothing then.

    `_c04_5` runs the C04.5 seam set instead of the curve of
    `WORK_ORDER_C04_6.md` §1: the raster is the markers as points with no
    segments drawn, the tip rule is `seams.tip_pass_continuous` on that
    raster, and a marker leaves at `WEAK_THRESHOLD` rather than at
    `SUTURE_STRENGTH`. It exists so §3.1's comparison row and §3.4's cost can
    be taken on both engines in one session and it is removed with
    `_c04_5.py` before the build report.

    **The yield is a percentile of the first step's strain field.** That is a
    calibration convenience, not a physical law: it makes the yield dial mean
    the same thing at every stiffness, because a stiffer sheet has smaller
    strains everywhere. It is a statistical self-reference and it is not to
    survive into production. Once a setting is chosen, the equivalent physical
    fraction of the drive's characteristic strain is frozen as a constant and
    the percentile goes.

    **Under `params.seams = 1` the damage rule is the seam formulation of
    `DESIGN.md` §3.6** and not the sheet's: the sheet starts intact at
    strength 1, damage happens only on a seam, at a seam's tip, or at a
    nucleation site, the strength noise becomes the heterogeneity of the
    intact yield stress, and strength is advected by whole cells rather than
    bilinearly. The drive, the solve, the stiffness, the strain, the epochs,
    the views and the metrics are the same either way.

    **Under `params.seams = 2` the block model runs**, the last paragraph of
    §3.6. Three things move and nothing else does:

    - *The velocity* is not solved on the sheet. Pieces are rigid bodies,
      three unknowns each, coupled through seam tractions; `rigid.py` says
      how, and the equations are the sheet's own summed over a piece. The
      velocity every view shows and markers move with is that rigid field.
    - *The stress* the tip and nucleation rules read is a second solve, of
      the sheet's operator forced by the drag a piece failed to match rather
      than by the drive. At step 1 there are no seams, the one piece stands
      still because the drive has zero mean, that forcing is the drive
      itself, and the solve is bit for bit the sheet's step-1 solve — which
      is why `sigma_c`, `yield_strain_per_myr` and `yield_power` are read
      from it exactly as the sheet reads them.
    - *The seam network* is a marker set, `markers.py`. There is no
      advection of strength at all: the raster is rebuilt from the markers
      every step and the markers move at the seam-cell velocity.

    The seam rules — tips, the Griffith threshold, nucleation, slip-rate
    damage, healing — are untouched, and so are the drive, the stiffness, the
    epochs, the views and the metrics. At `seams = 0`, which is the default
    and what production runs, every field is bit-for-bit what it was before
    the switch existed.
    """
    if params is None:
        params = DEFAULT_PARAMS
    elif not isinstance(params, HistoryParams):
        raise TypeError("params must be a HistoryParams")

    if steps is None:
        steps = default_steps(params.history_myr)
    else:
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if steps < 4 or steps % 4:
            raise ValueError("steps must be a positive multiple of four")

    history_myr = float(params.history_myr)
    step_myr = history_myr / steps
    heal_rate = params.heal_rate
    damage_rate = params.damage_rate
    strength_exponent = params.strength_exponent
    max_cycles = params.max_cycles
    cell_km = geometry.cell_km
    n = geometry.history_n
    solve_divisor = params.solve_divisor
    solve_n = solve_n_for(n, solve_divisor)
    kappa0 = kappa0_for(n, params.stiffness_fraction)
    # The yield is read off the first step's own strain field, at §1.2's
    # percentile, once the field exists. Nothing before then depends on it.
    yield_strain_per_myr = 0.0
    yield_power = 0.0
    keep = set(epoch_steps(steps))

    drive = build_drive(geometry, wavelength_km=params.drive_wavelength_km,
                        rot_ratio=params.drive_shear,
                        history_myr=history_myr)
    use_seams = params.seams >= 1
    use_rigid = params.seams >= 2
    if use_seams:
        # An intact sheet with no seams in it and no stagnant lid anywhere:
        # every boundary in the run is one this history opened. The noise the
        # sheet spent on the initial strength is spent on the intact yield
        # stress instead, at step 1, once there is a stress field to scale.
        strength_start = np.ones((n, n), dtype=np.float64)
        intact_noise = strength_noise(geometry)
    else:
        strength_start = initial_strength(geometry, params.strength_spread)
        intact_noise = None
    strength = strength_start.copy()
    sigma_c = 0.0
    sigma_c_field = np.zeros((n, n), dtype=np.float64)
    # Advances of a tip per step: a physical propagation rate turned into
    # cells by the cell size and the step length, so a crack crosses a piece
    # over several steps and the field it runs through is re-solved as it
    # goes. At the production step this is `STEP_MYR`.
    advances_per_step = max(0, round(
        params.crack_speed_km_per_myr * step_myr / cell_km))
    # The sub-cell remainder of the nearest-cell advection, carried from one
    # step to the next so a displacement below half a cell is spent rather
    # than rounded away. See `seams.advect_nearest`. Unused at `seams = 2`,
    # which advects nothing: the markers carry their own positions.
    advection_offset = np.zeros((2, n, n), dtype=np.float64)
    # The block model's carried state. `seam_set` is the seam network; the
    # two `previous_*` fields are the last step's velocity and seam mask, for
    # the one case §1.3 leaves to them — a seam cell with no intact
    # neighbour, which has no piece to take a velocity from. `pre_motion` is
    # the seam mask before the last step's markers moved, which is what a gap
    # is measured against.
    seam_set = marker_seams.empty()
    previous_u: np.ndarray | None = None
    previous_seam: np.ndarray | None = None
    pre_motion_seam: np.ndarray | None = None

    columns = np.arange(n, dtype=np.float64)[None, :]
    rows = np.arange(n, dtype=np.float64)[:, None]

    epochs: list[Epoch] = []
    early: list[tuple[int, float, np.ndarray]] = []
    weak_fraction: list[float] = []
    seam_fraction: list[float] = []
    tip_count: list[int] = []
    nucleation_count: list[int] = []
    advance_count: list[int] = []
    piece_count: list[int] = []
    largest_piece_share: list[float] = []
    second_piece_cells: list[int] = []
    force_residual_max: list[float] = []
    torque_residual_max: list[float] = []
    wrapping_pieces: list[int] = []
    marker_count: list[int] = []
    gaps_closed: list[int] = []
    meetings: list[int] = []
    subdivisions: list[int] = []
    suture_markers: list[int] = []
    reactivations: list[int] = []
    sample_count: list[int] = []
    degenerate_tips: list[int] = []
    seam_yield_share: list[float] = []
    elastic_slip_mean: list[float] = []
    rigid_slip_mean: list[float] = []
    exceed_fraction: list[float] = []
    strength_mean: list[float] = []
    strength_min: list[float] = []
    strain_rate_mean: list[float] = []
    strain_rate_max: list[float] = []
    solver_cycles: list[int] = []
    solver_residual: list[float] = []

    solved = None
    t_myr = 0.0
    for step in range(1, steps + 1):
        traction = drive.field(t_myr)
        kappa = kappa0 * strength**strength_exponent

        state = None
        gap_candidates = None
        if use_rigid:
            # The pieces of this step, the rigid motion that balances the
            # drag over each against the tractions its seams transmit, and
            # the velocity field that motion makes. `strength` here is the
            # raster the markers built at the end of the last step.
            if pre_motion_seam is not None:
                gap_candidates = marker_seams.gap_cells(
                    pre_motion_seam, strength < WEAK_THRESHOLD)
            state = rigid.piece_state(strength, traction, kappa0,
                                      strength_exponent, cell_km,
                                      previous_u, previous_seam)
            velocity = state["velocity"]
            # What the sheet solve is forced with is no longer the drive: it
            # is the drive a piece failed to match, which has no rigid motion
            # of the world in it and so gives the piece's own elastic
            # deformation, screened beyond the stiffness length. A piece
            # larger than that length therefore sees its stress partly
            # screened, and `stiffness_fraction` keeps exactly that meaning.
            forcing = state["mismatch"]
        else:
            forcing = traction

        # The velocity is smooth by construction, so it is solved on a grid
        # `solve_divisor` times coarser than the strength it is solved through
        # and interpolated back. At divisor 1 the two grids are the same one
        # and every transfer below is the identity.
        kappa_s, traction_s = to_solve_grid(kappa, forcing, solve_n)
        solved, cycles, residual = solve(traction_s, kappa_s, u0=solved,
                                         max_cycles=max_cycles)
        if not use_rigid:
            velocity = to_kinematic_grid(solved, n)
        solver_cycles.append(int(cycles))
        solver_residual.append(float(residual))

        # The strain rate a cell carries is the stress it carries divided by
        # its own stiffness, so it is built from the solver's edge fluxes, on
        # the grid those fluxes live on, and lifted to the kinematic grid
        # block by block. A central difference on the interpolated velocity
        # would instead reach across a failed cell and charge its strong
        # neighbour with a share of the velocity jump, which stress continuity
        # forbids.
        g_x, g_y = effective_gradients(solved, kappa_s)
        cell_s_km = cell_km * solve_divisor
        exx = g_x[0] / cell_s_km
        eyy = g_y[1] / cell_s_km
        exy = 0.5 * (g_y[0] + g_x[1]) / cell_s_km
        strain_rate_s = np.sqrt(exx * exx + eyy * eyy + 2.0 * exy * exy)
        divergence_s = exx + eyy
        # The power the sheet dissipates per cell. Stress here is stiffness
        # times strain rate, so this is stress times strain rate up to the
        # constant the yield calibration absorbs. It rides the same lift as
        # the strain rate, block by block, because it is resolved where the
        # strain is.
        # `stress_s` is the magnitude of the stress tensor whose components
        # are `kappa_s` times the strains: the same product the power is
        # built from, named because the seam rule reads it on its own.
        # `power_s` is `(kappa_s * strain_rate_s) * strain_rate_s`, which is
        # what the expression it replaces evaluated to, bit for bit.
        stress_s = kappa_s * strain_rate_s
        power_s = stress_s * strain_rate_s
        strain_rate = to_kinematic_blocks(strain_rate_s, n)
        power = to_kinematic_blocks(power_s, n)
        divergence = to_kinematic_blocks(divergence_s, n)

        if step == 1:
            # The yield: the strain the top `yield_percentile` per cent of the
            # first step's field exceeds. Read on the solve grid, where the
            # strain is resolved, and before any damage has been applied.
            yield_strain_per_myr = float(np.percentile(
                strain_rate_s, 100.0 - params.yield_percentile,
                method="linear"))
            if yield_strain_per_myr <= 0.0:
                raise ValueError("the first strain field has no yield to read")
            # The same percentile of the same step's dissipated power. Both
            # are read on every run whatever the law, so the record is
            # complete and the two thresholds can be compared.
            yield_power = float(np.percentile(
                power_s, 100.0 - params.yield_percentile, method="linear"))
            if yield_power <= 0.0:
                raise ValueError("the first power field has no yield to read")
            if use_seams:
                # The intact strength: the stress the top `yield_percentile`
                # per cent of the first step's field carries, read on the
                # solve grid where the stress is resolved, then scaled cell
                # by cell by the strength noise and clipped.
                sigma_c = float(np.percentile(
                    stress_s, 100.0 - params.yield_percentile,
                    method="linear"))
                if sigma_c <= 0.0:
                    raise ValueError(
                        "the first stress field has no strength to read")
                sigma_c_field = intact_strength_field(
                    sigma_c, intact_noise, params.strength_spread,
                    INTACT_SPREAD_CLIP)

        if use_seams:
            # The tensor the tip rule reads, and the magnitude the nucleation
            # rule ranks by, both bilinear: see `to_kinematic_smooth`. The
            # block lift of the same magnitude is what seam damage rides.
            sxx = to_kinematic_smooth(kappa_s * exx, n)
            syy = to_kinematic_smooth(kappa_s * eyy, n)
            sxy = to_kinematic_smooth(kappa_s * exy, n)
            smag = to_kinematic_smooth(stress_s, n)

        # Damage requires exceeding the yield. `work_damage` picks what is
        # compared with it: the strain rate against its percentile, or the
        # dissipated power against the same percentile of the power field.
        # Below yield there is no damage at all and healing is unopposed;
        # above it damage grows with the square of the excess. The integrator
        # is exact over the step: per cell the law is linear in S, so it is
        # stable at any step length and reduces to the explicit update as
        # dt -> 0.
        #
        # Under `seams` the only change to the comparison is *where* it is
        # allowed to act: on a seam and nowhere else. An intact cell has a
        # damage rate of zero however high its strain, so healing is unopposed
        # there and it stays intact. That is the unloading the sheet lacked,
        # and it is what holds a seam one cell wide. `work_damage` picks the
        # law on the cells that do damage, and on a seam the two differ in
        # kind: see `seams.damage_excess`. At 0, the default, a slipping seam
        # stays weak and heals when it stops; at 1 an open seam dissipates
        # almost nothing, damages almost not at all, and heals shut.
        #
        # At `seams = 2` the rate a seam cell reads is its own slip rate — the
        # velocity jump across it divided by the cell, plus the elastic strain
        # rate its own load imposes on its faces — and an intact cell is
        # rigid and has no strain rate of its own at all. The work law reads
        # the power that slip dissipates, `kappa(S) * slip ** 2`, against the
        # same step-1 percentile. Damage and healing then act on the markers
        # rather than on the raster, so two markers in one cell see one rate
        # and a marker that seals is removed.
        if use_rigid:
            # Under the block model the sheet solve is forced by the mismatch,
            # so `strain_rate` on these lines is the block lift of the
            # strain-rate invariant of `w`, the non-rigid part of the
            # velocity: the rate at which a crack's faces displace relative to
            # each other under the load its own piece carries. A seam cell's
            # slip is the rigid jump between the pieces it links plus that
            # elastic part, in the same units, so a crack inside a piece —
            # which links no pair and had no rigid jump at all — now slips.
            # See `rigid.seam_slip_rate`. `yield_strain_per_myr` is still read
            # at step 1 from the same invariant, where there are no seams and
            # nothing is added to it, so the calibration is unchanged.
            seam_cells = seam_mask(strength)
            rigid_slip = state["slip_rate"]
            elastic_slip = strain_rate * seam_cells
            slip = rigid.seam_slip_rate(rigid_slip, strain_rate, strength)
            seam_power = kappa * slip * slip
            excess = damage_excess(strength, seam_power, yield_power, slip,
                                   yield_strain_per_myr, params.work_damage)
            # What carried the damage, per step: the share of the markers
            # standing on a cell above the yield slip rate, and the two parts
            # of the slip averaged over the seam cells they act on.
            seam_total = int(seam_cells.sum())
            if seam_set.size:
                marker_rows, marker_columns = marker_seams.cells(seam_set, n)
                step_yield_share = float(np.mean(
                    slip[marker_rows, marker_columns] > yield_strain_per_myr))
            else:
                step_yield_share = 0.0
            step_elastic_mean = (float(elastic_slip[seam_cells].mean())
                                 if seam_total else 0.0)
            step_rigid_mean = (float(rigid_slip[seam_cells].mean())
                               if seam_total else 0.0)
        elif use_seams:
            excess = damage_excess(strength, power, yield_power, strain_rate,
                                   yield_strain_per_myr, params.work_damage)
        elif params.work_damage:
            excess = np.maximum(power / yield_power - 1.0, 0.0)
        else:
            excess = np.maximum(strain_rate / yield_strain_per_myr - 1.0, 0.0)
        if not use_rigid:
            step_yield_share = 0.0
            step_elastic_mean = 0.0
            step_rigid_mean = 0.0
        step_reactivated = 0
        if use_rigid:
            if _trace is not None:
                # Bookkeeping only: which cells held markers before healing
                # and how many of those healing is about to remove. Read
                # from `markers.healed_strength`, which is the law
                # `damage_and_heal` applies on the next line.
                trace_rows, trace_columns = marker_seams.cells(seam_set, n)
                trace_flat = trace_rows * n + trace_columns
                trace_held = np.bincount(
                    trace_flat, minlength=n * n).reshape(n, n)
                trace_sealed = ~(marker_seams.healed_strength(
                    seam_set, excess, heal_rate, damage_rate, step_myr, n)
                    < (WEAK_THRESHOLD if _c04_5 else SUTURE_STRENGTH))
                trace_healed = np.bincount(
                    trace_flat[trace_sealed], minlength=n * n).reshape(n, n)
            if _c04_5:
                seam_set, _sealed = c04_5_seams.damage_and_heal(
                    seam_set, excess, heal_rate, damage_rate, step_myr, n)
                strength = c04_5_seams.raster(seam_set, n)
            else:
                # A marker leaves at `SUTURE_STRENGTH`, not at the weak
                # threshold: between the two it is intact in the raster and
                # still a vertex, so the curve is remembered and reopens
                # where the slip returns. See `markers.damage_and_heal`.
                healed = marker_seams.damage_and_heal(
                    seam_set, excess, heal_rate, damage_rate,
                    step_myr, n)
                seam_set, _sealed, step_reactivated = healed
                strength = marker_seams.raster(seam_set, n)
        else:
            rate = damage_rate * excess * excess
            total = heal_rate + rate
            equilibrium = heal_rate / total
            strength = np.clip(
                equilibrium + (strength - equilibrium) * np.exp(-total * step_myr),
                STRENGTH_MIN, 1.0)

        step_tips = 0
        step_advances = 0
        step_nuclei = 0
        step_gaps_closed = 0
        step_meetings = 0
        step_subdivisions = 0
        step_degenerate = 0
        if use_rigid and not _c04_5:
            # `WORK_ORDER_C04_6.md` §1.3. The tips are the curve's own end
            # vertices, so a pass grows the marker set itself and the raster
            # is rebuilt from it; the raster's `tips` is not read at all. A
            # pass in which no tip advanced leaves the curve untouched, so
            # every later pass on the same field would be the same no-op and
            # the loop stops there rather than repeating it.
            before_passes = seam_mask(strength)
            for index in range(advances_per_step):
                (seam_set, pass_tips, pass_advances, pass_meetings,
                 pass_degenerate) = marker_seams.advance_tips(
                    seam_set, strength, sxx, syy, sxy, sigma_c_field,
                    params.toughness_fraction)
                if index == 0:
                    step_tips = pass_tips
                step_advances += pass_advances
                step_meetings += pass_meetings
                step_degenerate += pass_degenerate
                if pass_advances == 0:
                    break
                strength = marker_seams.raster(seam_set, n)
            if advances_per_step == 0:
                step_tips = int((marker_seams.degrees(seam_set) <= 1).sum())
            nucleated, step_nuclei = nucleate(
                strength, smag, sigma_c_field, params.nucleations_per_step)
            # A nucleus is a marker of degree 0 at its cell's centre, which
            # is what it has always been.
            seam_set = marker_seams.create(
                seam_set, marker_seams.opened_cells(strength, nucleated))
            strength = marker_seams.raster(seam_set, n)
            if gap_candidates is not None:
                step_gaps_closed = int(
                    (gap_candidates & seam_mask(strength)
                     & ~before_passes).sum())
        elif use_seams:
            # `advances_per_step` passes on the same stress field, the seam
            # mask and the crack lengths recomputed between them, so a tip
            # that advanced is the tip that advances next and a crack that
            # has grown longer runs on a lower threshold. A pass in which no
            # tip advanced leaves the strength field untouched, so every
            # later pass on the same field would be the same no-op: the loop
            # stops there rather than repeating it.
            #
            # At `seams = 2` the passes still run on the raster, because the
            # rules read a field of cells; every cell they open becomes one
            # marker at its centre afterwards. A cell is opened once however
            # many tips reached it, so no cell gains two markers in a step,
            # and the raster the markers then make is the raster the passes
            # left: a freshly opened cell held no marker before, since every
            # marker is below the weak threshold and the passes open only
            # intact cells.
            opened = np.zeros((n, n), dtype=bool)
            if use_rigid:
                # Where inside its cell every marker sits, so a tip starts
                # from its own position and not from a cell centre. A cell
                # this step's passes have already opened holds no marker
                # yet — the marker set gains them once the passes are done —
                # so its `p'` is carried here beside the ones on the record.
                marker_offsets, marker_holds = marker_seams.cell_offsets(
                    seam_set, n)
                open_offsets = np.zeros((2, n, n), dtype=np.float64)
            for index in range(advances_per_step):
                if use_rigid:
                    offsets = np.where(opened, open_offsets, marker_offsets)
                    holds = marker_holds + opened
                    (advanced, pass_tips, pass_advances, pass_offsets,
                     pass_multi, pass_degenerate) = tip_pass_continuous(
                        strength, sxx, syy, sxy, sigma_c_field, offsets,
                        holds, params.toughness_fraction)
                    del pass_multi
                    step_degenerate += pass_degenerate
                    newly = marker_seams.opened_cells(strength, advanced)
                    open_offsets = np.where(newly, pass_offsets, open_offsets)
                    opened |= newly
                else:
                    advanced, pass_tips, pass_advances = tip_pass(
                        strength, sxx, syy, sxy, sigma_c_field,
                        params.toughness_fraction)
                strength = advanced
                if index == 0:
                    step_tips = pass_tips
                step_advances += pass_advances
                if pass_advances == 0:
                    break
            if advances_per_step == 0:
                step_tips = int(tips(strength < WEAK_THRESHOLD).sum())
            nucleated, step_nuclei = nucleate(
                strength, smag, sigma_c_field, params.nucleations_per_step)
            if use_rigid:
                opened |= marker_seams.opened_cells(strength, nucleated)
            strength = nucleated
            if use_rigid:
                # A tip advance's marker goes at the point the advance
                # reached; a nucleus's offset is zero, which is its cell's
                # centre, as it has always been.
                seam_set = c04_5_seams.create(seam_set, opened,
                                              open_offsets)
                if gap_candidates is not None:
                    step_gaps_closed = int((gap_candidates & opened).sum())

        # Strength belongs to the lithosphere, so it travels with it. Without
        # this a weak zone would stay pinned to the mantle frame. The sheet
        # interpolates; the seam formulation cannot, because a seam is a
        # discontinuity and interpolating it is what turns one cell into two.
        # See `seams.advect_nearest` for the remainder it carries instead.
        if use_rigid:
            # Nothing is resampled: each marker steps by its own cell's
            # velocity and keeps its sub-cell position, so a seam cannot be
            # written into two cells however the velocity jumps across it,
            # and the cells between two linked markers are drawn wherever the
            # two ends go, so it cannot be vacated either.
            pre_motion_seam = strength < WEAK_THRESHOLD
            if _trace is not None:
                trace_before = (seam_set.x.copy(), seam_set.y.copy())
            seam_set = marker_seams.move(seam_set, velocity, step_myr,
                                         cell_km, n)
            if _trace is not None:
                trace_after = (seam_set.x.copy(), seam_set.y.copy())
            if _c04_5:
                strength = c04_5_seams.raster(seam_set, n)
            else:
                # The move is what stretches an edge, because its two ends
                # belong to two cells and take two velocities.
                seam_set, step_subdivisions = marker_seams.subdivide(
                    seam_set, n)
                strength = marker_seams.raster(seam_set, n)
        elif use_seams:
            strength, advection_offset = advect_nearest(
                strength,
                -velocity * step_myr / cell_km,
                advection_offset, columns, rows)
        else:
            strength = sample_bilinear_periodic(
                strength,
                columns - velocity[0] * step_myr / cell_km,
                rows - velocity[1] * step_myr / cell_km,
            )

        weak_fraction.append(float(np.mean(strength < WEAK_THRESHOLD)))
        seam_fraction.append(weak_fraction[-1])
        tip_count.append(step_tips)
        nucleation_count.append(step_nuclei)
        advance_count.append(step_advances)
        piece_count.append(0 if state is None else int(state["piece_count"]))
        largest_piece_share.append(
            0.0 if state is None else float(state["largest_piece_share"]))
        second_piece_cells.append(
            0 if state is None else int(state["second_piece_cells"]))
        force_residual_max.append(
            0.0 if state is None else float(state["force_residual"]))
        torque_residual_max.append(
            0.0 if state is None else float(state["torque_residual"]))
        wrapping_pieces.append(
            0 if state is None else int(state["wrapping_pieces"]))
        marker_count.append(seam_set.size)
        gaps_closed.append(step_gaps_closed)
        meetings.append(step_meetings)
        subdivisions.append(step_subdivisions)
        suture_markers.append(
            int((seam_set.s >= WEAK_THRESHOLD).sum()) if use_rigid else 0)
        reactivations.append(step_reactivated if use_rigid else 0)
        sample_count.append(
            marker_seams.sample_total(seam_set, n) if use_rigid else 0)
        degenerate_tips.append(step_degenerate)
        if _trace is not None:
            _trace.append({
                "step": step,
                "seam_end": strength < WEAK_THRESHOLD,
                "covered": (marker_seams.drawn_cells(seam_set, n)
                            if use_rigid and not _c04_5 else None),
                "held": trace_held if use_rigid else None,
                "healed": trace_healed if use_rigid else None,
                "slip": slip.copy() if use_rigid else None,
                # The same markers in the same order on either side of the
                # move, so a marker's displacement over the step is the
                # difference and no identity has to be carried.
                "before_move": trace_before if use_rigid else None,
                "after_move": trace_after if use_rigid else None,
            })
        seam_yield_share.append(step_yield_share)
        elastic_slip_mean.append(step_elastic_mean)
        rigid_slip_mean.append(step_rigid_mean)
        exceed_fraction.append(float(np.mean(excess > 0.0)))
        strength_mean.append(float(strength.mean()))
        strength_min.append(float(strength.min()))
        strain_rate_mean.append(float(strain_rate.mean()))
        strain_rate_max.append(float(strain_rate.max()))
        t_myr += step_myr
        if step in EARLY_SNAPSHOT_STEPS:
            early.append((step, t_myr, strength.copy()))
        if step in keep:
            epochs.append(Epoch(
                t_myr=t_myr,
                strength=strength.copy(),
                velocity=velocity.copy(),
                strain_rate=strain_rate,
                power=power,
                divergence=divergence,
                stress=(smag if use_seams
                        else to_kinematic_smooth(stress_s, n)),
                mismatch=(np.hypot(forcing[0], forcing[1]) if use_rigid
                          else np.zeros((n, n), dtype=np.float64)),
                piece_labels=(None if state is None
                              else state["labels"].copy()),
                piece_centroid=(None if state is None
                                else state["pieces"].centroid.copy()),
                piece_velocity=(None if state is None
                                else state["piece_velocity"].copy()),
            ))
        if use_rigid:
            previous_u = velocity
            previous_seam = state["labels"] < 0

    return History(
        geometry=geometry,
        params=params,
        drive=drive,
        strength_initial=strength_start,
        epochs=epochs,
        early=early,
        weak_fraction=weak_fraction,
        seam_fraction=seam_fraction,
        tip_count=tip_count,
        nucleation_count=nucleation_count,
        advance_count=advance_count,
        piece_count=piece_count,
        largest_piece_share=largest_piece_share,
        second_piece_cells=second_piece_cells,
        force_residual_max=force_residual_max,
        torque_residual_max=torque_residual_max,
        wrapping_pieces=wrapping_pieces,
        marker_count=marker_count,
        gaps_closed=gaps_closed,
        meetings=meetings,
        subdivisions=subdivisions,
        suture_markers=suture_markers,
        reactivations=reactivations,
        sample_count=sample_count,
        degenerate_tips=degenerate_tips,
        seam_yield_share=seam_yield_share,
        elastic_slip_mean=elastic_slip_mean,
        rigid_slip_mean=rigid_slip_mean,
        exceed_fraction=exceed_fraction,
        strength_mean=strength_mean,
        strength_min=strength_min,
        strain_rate_mean=strain_rate_mean,
        strain_rate_max=strain_rate_max,
        solver_cycles=solver_cycles,
        solver_residual=solver_residual,
        steps=steps,
        step_myr=step_myr,
        history_myr=history_myr,
        yield_strain_per_myr=yield_strain_per_myr,
        yield_power=yield_power,
        sigma_c=sigma_c,
        sigma_c_field=sigma_c_field,
    )


__all__ = [
    "DEFAULT_PARAMS",
    "EARLY_SNAPSHOT_STEPS",
    "STRENGTH_PROCESS_ID",
    "Epoch",
    "History",
    "HistoryParams",
    "default_steps",
    "epoch_steps",
    "initial_strength",
    "run_history",
    "solve_n_for",
    "strength_noise",
    "to_kinematic_blocks",
    "to_kinematic_grid",
    "to_kinematic_smooth",
    "to_solve_grid",
]
