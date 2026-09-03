"""Every number that shapes a field in the kinematic history.

Nothing else in the engine defines a field-shaping constant. `KAPPA0` is the
one derived quantity and lives in the solver, because it depends on the grid.
"""

from __future__ import annotations

STAGE_ID = "kinematic_history.v1"
STAGE_VERSION = "1"

SCALE_MIN = 5                    # km per delivered pixel
SCALE_MAX = 20
SCALE_DEFAULT = 5
SUPPORTED_SIZES = (128, 256, 512, 1024, 2048)
DEFAULT_SIZE = 1024
MIN_HISTORY_N = 128              # floor grid; parent never below 128 cells
CELL_PX = 8                      # one history cell is eight delivered pixels
PARENT_WINDOW_RATIO = 2          # the parent is twice the delivered window

HISTORY_MYR = 300.0
STEP_MYR = 4.0
EPOCH_FRACTIONS = (0.25, 0.5, 0.75, 1.0)

DRIVE_RMS_KM_PER_MYR = 40.0      # 4 cm/yr
DRIVE_ROT_RATIO = 0.5            # rotational part relative to curl-free part
DRIVE_KEYFRAMES = 3
DRIVE_WAVELENGTH_KM = 5120.0     # coarsest mantle wavelength; parent / 2 at 1024 px, 5 km/px
DRIVE_OCTAVES = 4

STRENGTH_MIN = 0.05
STRENGTH_INIT_MEAN = 0.9
STRENGTH_INIT_SPREAD = 0.1
STRENGTH_WAVELENGTH_KM = 1280.0  # coarsest strength-noise wavelength; parent / 8 there
STRENGTH_OCTAVES = 5
STRENGTH_EXPONENT = 4            # kappa = KAPPA0 * S**4
HOMOG_LENGTH_FRACTION = 0.125    # homogenization length = parent / 8

DAMAGE_RATE = 0.2                # 1/Myr at twice yield; 1 / 5 Myr to fail
HEAL_RATE = 0.01                 # 1/Myr; 1 / 100 Myr to seal
WEAK_THRESHOLD = 0.5
REGIME_RATIO = 0.3

SEAM_OPEN_STRENGTH = STRENGTH_MIN   # strength of a freshly cracked cell
SUTURE_STRENGTH = 0.9               # WORK_ORDER_C04_6.md §1.4: a marker leaves the
                                    # curve here, not at WEAK_THRESHOLD; between the
                                    # two it is a remembered vertex on an intact cell
SEGMENT_MAX_CELLS = 1.5             # WORK_ORDER_C04_6.md §1.4: an edge longer than
                                    # this is subdivided at its midpoint after a move
MEETING_RADIUS_CELLS = 1.5          # WORK_ORDER_C04_6.md §1.3: an advance that lands
                                    # this close to another chain links to it
SEGMENT_SAMPLE_CELLS = 0.5          # WORK_ORDER_C04_6.md §1.2: the raster samples a
                                    # segment at most this far apart, so the cells it
                                    # draws are an 8-connected path one cell wide
INTACT_SPREAD_CLIP = (0.2, 2.0)     # bounds on the intact-strength heterogeneity

SOLVE_GRID_DIVISOR = 2           # velocity is solved on half the kinematic grid

MG_COARSEST = 8                  # coarsest multigrid grid, solved exactly
MG_PRE = 3                       # red-black Gauss-Seidel sweeps before coarsening
MG_POST = 3                      # and after, in the reverse colour order
MG_TOL = 1e-3
MG_MAX_CYCLES = 20

__all__ = [name for name in globals() if name.isupper()]
