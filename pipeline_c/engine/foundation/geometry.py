"""Exact physical rectangles and cell-center grids for the C4 foundation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ._util import FoundationRecordError, require_int
from .constants import (
    ANALYSIS_HEIGHT_M,
    ANALYSIS_MIN_X_M,
    ANALYSIS_MIN_Y_M,
    ANALYSIS_WIDTH_M,
    CANONICAL_UNITS,
    COORDINATE_SYSTEM_ID,
    GRID_SCHEMA_ID,
    NUMERICAL_HEIGHT_M,
    NUMERICAL_MIN_X_M,
    NUMERICAL_MIN_Y_M,
    NUMERICAL_WIDTH_M,
    PARENT_HEIGHT_M,
    PARENT_MIN_X_M,
    PARENT_MIN_Y_M,
    PARENT_WIDTH_M,
    RECTANGLE_SEMANTICS,
    ROW_ORIENTATION,
    SAMPLE_LOCATION,
    SUPPORTED_SIZES,
)


SIGNED_64_MIN = -(2**63)
SIGNED_64_MAX = 2**63 - 1


@dataclass(frozen=True, slots=True)
class PhysicalRect:
    """A rectangle in integer metres with half-open membership semantics."""

    min_x_m: int
    min_y_m: int
    width_m: int
    height_m: int

    def __post_init__(self) -> None:
        for name in ("min_x_m", "min_y_m"):
            require_int(
                getattr(self, name), name, minimum=SIGNED_64_MIN, maximum=SIGNED_64_MAX
            )
        for name in ("width_m", "height_m"):
            require_int(getattr(self, name), name, minimum=1, maximum=SIGNED_64_MAX)
        if self.max_x_m > SIGNED_64_MAX or self.max_y_m > SIGNED_64_MAX:
            raise FoundationRecordError("rectangle maximum exceeds signed 64-bit metres")

    @property
    def max_x_m(self) -> int:
        return self.min_x_m + self.width_m

    @property
    def max_y_m(self) -> int:
        return self.min_y_m + self.height_m

    @property
    def area_m2(self) -> int:
        return self.width_m * self.height_m

    def contains_point(self, x_m: int, y_m: int) -> bool:
        require_int(x_m, "x_m", minimum=SIGNED_64_MIN, maximum=SIGNED_64_MAX)
        require_int(y_m, "y_m", minimum=SIGNED_64_MIN, maximum=SIGNED_64_MAX)
        return (
            self.min_x_m <= x_m < self.max_x_m
            and self.min_y_m <= y_m < self.max_y_m
        )

    def contains_rect(self, other: "PhysicalRect") -> bool:
        if not isinstance(other, PhysicalRect):
            raise TypeError("other must be PhysicalRect")
        return (
            self.min_x_m <= other.min_x_m
            and self.min_y_m <= other.min_y_m
            and other.max_x_m <= self.max_x_m
            and other.max_y_m <= self.max_y_m
        )

    def overlaps(self, other: "PhysicalRect") -> bool:
        if not isinstance(other, PhysicalRect):
            raise TypeError("other must be PhysicalRect")
        return (
            self.min_x_m < other.max_x_m
            and other.min_x_m < self.max_x_m
            and self.min_y_m < other.max_y_m
            and other.min_y_m < self.max_y_m
        )

    def translated(self, dx_m: int, dy_m: int) -> "PhysicalRect":
        require_int(dx_m, "dx_m", minimum=SIGNED_64_MIN, maximum=SIGNED_64_MAX)
        require_int(dy_m, "dy_m", minimum=SIGNED_64_MIN, maximum=SIGNED_64_MAX)
        return PhysicalRect(
            self.min_x_m + dx_m,
            self.min_y_m + dy_m,
            self.width_m,
            self.height_m,
        )

    def expanded(self, margin_m: int) -> "PhysicalRect":
        require_int(margin_m, "margin_m", minimum=0, maximum=SIGNED_64_MAX)
        return PhysicalRect(
            self.min_x_m - margin_m,
            self.min_y_m - margin_m,
            self.width_m + 2 * margin_m,
            self.height_m + 2 * margin_m,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "height_m": self.height_m,
            "max_x_m": self.max_x_m,
            "max_y_m": self.max_y_m,
            "min_x_m": self.min_x_m,
            "min_y_m": self.min_y_m,
            "rectangle_semantics": RECTANGLE_SEMANTICS,
            "units": CANONICAL_UNITS,
            "width_m": self.width_m,
        }

    @classmethod
    def from_record(cls, value: object) -> "PhysicalRect":
        keys = {
            "height_m", "max_x_m", "max_y_m", "min_x_m", "min_y_m",
            "rectangle_semantics", "units", "width_m",
        }
        if not isinstance(value, dict) or set(value) != keys:
            raise FoundationRecordError("physical rectangle record has unexpected keys")
        if value["rectangle_semantics"] != RECTANGLE_SEMANTICS:
            raise FoundationRecordError("unsupported rectangle semantics")
        if value["units"] != CANONICAL_UNITS:
            raise FoundationRecordError("physical rectangle must use integer metres")
        result = cls(
            min_x_m=value["min_x_m"],
            min_y_m=value["min_y_m"],
            width_m=value["width_m"],
            height_m=value["height_m"],
        )
        if value["max_x_m"] != result.max_x_m or value["max_y_m"] != result.max_y_m:
            raise FoundationRecordError("physical rectangle maxima are inconsistent")
        return result


@dataclass(frozen=True, slots=True)
class PhysicalGrid:
    """An exact area-cell grid; row zero is the greatest-y row."""

    rectangle: PhysicalRect
    width_px: int
    height_px: int
    coordinate_system_id: str = COORDINATE_SYSTEM_ID
    sample_location: str = SAMPLE_LOCATION
    row_orientation: str = ROW_ORIENTATION

    def __post_init__(self) -> None:
        if not isinstance(self.rectangle, PhysicalRect):
            raise FoundationRecordError("grid rectangle must be PhysicalRect")
        require_int(self.width_px, "width_px", minimum=1)
        require_int(self.height_px, "height_px", minimum=1)
        if self.coordinate_system_id != COORDINATE_SYSTEM_ID:
            raise FoundationRecordError("unsupported coordinate system")
        if self.sample_location != SAMPLE_LOCATION:
            raise FoundationRecordError("C4 grids must sample at cell centers")
        if self.row_orientation != ROW_ORIENTATION:
            raise FoundationRecordError("C4 row zero must be the greatest-y row")
        if self.rectangle.width_m % self.width_px:
            raise FoundationRecordError("grid cell width must be an integer number of metres")
        if self.rectangle.height_m % self.height_px:
            raise FoundationRecordError("grid cell height must be an integer number of metres")

    @property
    def cell_width_m(self) -> int:
        return self.rectangle.width_m // self.width_px

    @property
    def cell_height_m(self) -> int:
        return self.rectangle.height_m // self.height_px

    def _cell(self, column: int, row: int) -> tuple[int, int]:
        require_int(column, "column", minimum=0)
        require_int(row, "row", minimum=0)
        if column >= self.width_px or row >= self.height_px:
            raise FoundationRecordError("cell index lies outside the physical grid")
        return column, row

    def cell_center_m(self, column: int, row: int) -> tuple[Fraction, Fraction]:
        column, row = self._cell(column, row)
        x = Fraction(
            2 * self.rectangle.min_x_m + (2 * column + 1) * self.cell_width_m,
            2,
        )
        y = Fraction(
            2 * self.rectangle.max_y_m - (2 * row + 1) * self.cell_height_m,
            2,
        )
        return x, y

    def cell_bounds_m(self, column: int, row: int) -> PhysicalRect:
        column, row = self._cell(column, row)
        return PhysicalRect(
            self.rectangle.min_x_m + column * self.cell_width_m,
            self.rectangle.max_y_m - (row + 1) * self.cell_height_m,
            self.cell_width_m,
            self.cell_height_m,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "cell_height_m": self.cell_height_m,
            "cell_width_m": self.cell_width_m,
            "coordinate_system_id": self.coordinate_system_id,
            "height_px": self.height_px,
            "rectangle": self.rectangle.to_record(),
            "row_orientation": self.row_orientation,
            "sample_location": self.sample_location,
            "schema_id": GRID_SCHEMA_ID,
            "schema_version": 1,
            "width_px": self.width_px,
        }


PARENT_RECT = PhysicalRect(
    PARENT_MIN_X_M, PARENT_MIN_Y_M, PARENT_WIDTH_M, PARENT_HEIGHT_M
)
NUMERICAL_RECT = PhysicalRect(
    NUMERICAL_MIN_X_M,
    NUMERICAL_MIN_Y_M,
    NUMERICAL_WIDTH_M,
    NUMERICAL_HEIGHT_M,
)
DEVELOPMENT_ANALYSIS_RECT = PhysicalRect(
    ANALYSIS_MIN_X_M,
    ANALYSIS_MIN_Y_M,
    ANALYSIS_WIDTH_M,
    ANALYSIS_HEIGHT_M,
)


def analysis_grid(size: int) -> PhysicalGrid:
    require_int(size, "size", minimum=1)
    if size not in SUPPORTED_SIZES:
        raise FoundationRecordError(
            f"unsupported C4 size {size}; expected one of {SUPPORTED_SIZES}"
        )
    return PhysicalGrid(DEVELOPMENT_ANALYSIS_RECT, size, size)


def exact_nested_ratio(coarse: PhysicalGrid, fine: PhysicalGrid) -> tuple[int, int]:
    """Return the exact fine-cell count on each coarse axis or fail."""

    if not isinstance(coarse, PhysicalGrid) or not isinstance(fine, PhysicalGrid):
        raise TypeError("coarse and fine must be PhysicalGrid")
    if coarse.rectangle != fine.rectangle:
        raise FoundationRecordError("nested grids must cover the same physical rectangle")
    if fine.width_px % coarse.width_px or fine.height_px % coarse.height_px:
        raise FoundationRecordError("fine grid dimensions must divide by coarse dimensions")
    ratio_x = fine.width_px // coarse.width_px
    ratio_y = fine.height_px // coarse.height_px
    if ratio_x < 1 or ratio_y < 1:
        raise FoundationRecordError("fine grid cannot be coarser than the coarse grid")
    if (
        coarse.cell_width_m != fine.cell_width_m * ratio_x
        or coarse.cell_height_m != fine.cell_height_m * ratio_y
    ):
        raise FoundationRecordError("grid dimensions do not imply exact containment")
    return ratio_x, ratio_y


__all__ = [
    "DEVELOPMENT_ANALYSIS_RECT",
    "NUMERICAL_RECT",
    "PARENT_RECT",
    "PhysicalGrid",
    "PhysicalRect",
    "analysis_grid",
    "exact_nested_ratio",
]
