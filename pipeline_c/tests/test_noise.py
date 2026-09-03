"""Gates on the one sanctioned stochastic input."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

from engine.geometry import WorldGeometry  # noqa: E402
from engine.noise import band_cycles, periodic_noise  # noqa: E402
from engine.sampler import StageSampler  # noqa: E402
from engine.history.constants import (  # noqa: E402
    DRIVE_OCTAVES,
    DRIVE_WAVELENGTH_KM,
    STAGE_ID,
    STAGE_VERSION,
    STRENGTH_OCTAVES,
    STRENGTH_WAVELENGTH_KM,
)
from engine.history.drive import DRIVE_PROCESS_ID, build_drive  # noqa: E402
from tools.spectrum import (  # noqa: E402
    axis_to_diagonal,
    dominant_cycles,
    low_edge,
)

#: The audit seed, and the twelve development seeds of `STATUS.md`.
AUDIT_SEED = 4287772760
DEVELOPMENT_SEEDS = (
    2075014389, 2477733044, 476149591, 151640007, 2697441485, 1504571935,
    548870008, 2157195430, 4108373596, 4287772760, 287488203, 1833546021,
)


def sampler_for(geometry: WorldGeometry, process: str = "test") -> StageSampler:
    return StageSampler(geometry.world_id, STAGE_ID, STAGE_VERSION, process)


def field_for(geometry: WorldGeometry, *, channel: int = 0,
              process: str = "test") -> np.ndarray:
    return periodic_noise(sampler_for(geometry, process), geometry,
                          channel=channel, nodes_coarsest=4, octaves=3)


class Determinism(unittest.TestCase):
    def test_repeated_calls_are_byte_identical(self) -> None:
        geometry = WorldGeometry(11, 512, 5)
        first = field_for(geometry)
        second = field_for(geometry)
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_channel_and_process_separate_the_field(self) -> None:
        geometry = WorldGeometry(11, 512, 5)
        base = field_for(geometry)
        self.assertFalse(np.array_equal(base, field_for(geometry, channel=1)))
        self.assertFalse(
            np.array_equal(base, field_for(geometry, process="other")))

    def test_octave_count_and_lattice_change_the_field(self) -> None:
        geometry = WorldGeometry(11, 512, 5)
        sampler = sampler_for(geometry)
        base = periodic_noise(sampler, geometry, channel=0,
                              nodes_coarsest=4, octaves=3)
        more = periodic_noise(sampler, geometry, channel=0,
                              nodes_coarsest=4, octaves=4)
        coarser = periodic_noise(sampler, geometry, channel=0,
                                 nodes_coarsest=2, octaves=3)
        self.assertFalse(np.array_equal(base, more))
        self.assertFalse(np.array_equal(base, coarser))


class Seamlessness(unittest.TestCase):
    def test_the_wrap_is_not_a_seam(self) -> None:
        geometry = WorldGeometry(1, 512, 5)
        field = field_for(geometry)
        wrap = max(
            float(np.abs(field[:, 0] - field[:, -1]).max()),
            float(np.abs(field[0, :] - field[-1, :]).max()),
        )
        interior = max(
            float(np.abs(np.diff(field, axis=1)).max()),
            float(np.abs(np.diff(field, axis=0)).max()),
        )
        self.assertLessEqual(wrap, interior)

    def test_the_stage_lattices_are_seamless_too(self) -> None:
        geometry = WorldGeometry(1, 512, 5)
        sampler = sampler_for(geometry)
        for wavelength_km, octaves in (
            (DRIVE_WAVELENGTH_KM, DRIVE_OCTAVES),
            (STRENGTH_WAVELENGTH_KM, STRENGTH_OCTAVES),
        ):
            nodes = band_cycles(geometry, wavelength_km=wavelength_km)
            with self.subTest(nodes=nodes, octaves=octaves):
                field = periodic_noise(sampler, geometry, channel=0,
                                       nodes_coarsest=nodes, octaves=octaves)
                wrap = max(
                    float(np.abs(field[:, 0] - field[:, -1]).max()),
                    float(np.abs(field[0, :] - field[-1, :]).max()),
                )
                interior = max(
                    float(np.abs(np.diff(field, axis=1)).max()),
                    float(np.abs(np.diff(field, axis=0)).max()),
                )
                self.assertLessEqual(wrap, interior)


class Normalization(unittest.TestCase):
    def test_zero_mean_and_unit_deviation(self) -> None:
        for pixels in (128, 512, 1024):
            with self.subTest(pixels=pixels):
                field = field_for(WorldGeometry(3, pixels, 5))
                self.assertAlmostEqual(float(field.mean()), 0.0, delta=1e-9)
                self.assertAlmostEqual(float(field.std()), 1.0, delta=1e-9)

    def test_shape_follows_the_history_grid(self) -> None:
        geometry = WorldGeometry(3, 1024, 5)
        self.assertEqual(field_for(geometry).shape, (256, 256))


class WorldSeparation(unittest.TestCase):
    def test_resolutions_are_different_worlds(self) -> None:
        # parent_m differs, so the node addresses differ; the world_id differs
        # too. A lower resolution is a smaller world, never a coarser sample.
        big = WorldGeometry(5, 1024, 5)
        small = WorldGeometry(5, 512, 5)
        self.assertNotEqual(big.parent_m, small.parent_m)
        overlap = field_for(big)[:128, :128]
        self.assertFalse(np.array_equal(overlap, field_for(small)))

    def test_a_band_that_is_not_a_power_of_two_is_accepted(self) -> None:
        # The lattice-divisibility rule went with the lattice. A radial
        # envelope has no node spacing to divide the parent, so any positive
        # band is buildable and only a non-positive one is refused.
        geometry = WorldGeometry(5, 512, 5)
        field = periodic_noise(sampler_for(geometry), geometry, channel=0,
                               nodes_coarsest=3, octaves=1)
        self.assertEqual(field.shape, (geometry.history_n,) * 2)
        for nodes, octaves in ((0, 1), (1, 0), (-2, 3)):
            with self.subTest(nodes=nodes, octaves=octaves):
                with self.assertRaises(ValueError):
                    periodic_noise(sampler_for(geometry), geometry, channel=0,
                                   nodes_coarsest=nodes, octaves=octaves)


class Isotropy(unittest.TestCase):
    """The field must not prefer the world's axes.

    Mean power density in +-15 degree wedges about the k-axes over the same
    about the diagonals, `4 <= |k| <= n/2`. One is isotropic. The lattice
    noise this replaced measured 2.02 on the audit seed, because a lattice of
    `nodes` nodes puts its power on the axes at multiples of the node spacing
    and the smoothstep interpolation reinforces them.
    """

    def field(self, seed: int) -> np.ndarray:
        geometry = WorldGeometry(seed, 1024, 5)
        self.assertEqual(geometry.history_n, 256)
        return periodic_noise(sampler_for(geometry, "strength-initial"),
                              geometry, channel=0,
                              wavelength_km=STRENGTH_WAVELENGTH_KM,
                              octaves=STRENGTH_OCTAVES)

    def test_the_audit_seed_is_isotropic_at_256(self) -> None:
        statistic = axis_to_diagonal(self.field(AUDIT_SEED))
        print(f"\n  raw noise isotropy, seed {AUDIT_SEED} at n = 256: "
              f"{statistic:.4f}")
        self.assertGreaterEqual(statistic, 0.9)
        self.assertLessEqual(statistic, 1.1)

    def test_the_development_seeds_average_isotropic(self) -> None:
        # One draw of a random field fluctuates: the k**-1 envelope puts most
        # of the power in the few dozen lowest-|k| bins, so a single field's
        # statistic carries real sampling spread. The generator's claim is
        # about the ensemble, so the mean over the twelve is bounded too.
        values = [axis_to_diagonal(self.field(seed))
                  for seed in DEVELOPMENT_SEEDS]
        mean = float(np.mean(values))
        print(f"  twelve development seeds: min {min(values):.4f} "
              f"max {max(values):.4f} mean {mean:.4f}")
        self.assertGreaterEqual(mean, 0.9)
        self.assertLessEqual(mean, 1.1)
        self.assertLess(max(values), 1.15)
        self.assertGreater(min(values), 1.0 / 1.15)

    def test_the_drive_band_is_isotropic_too(self) -> None:
        geometry = WorldGeometry(AUDIT_SEED, 1024, 5)
        statistic = axis_to_diagonal(periodic_noise(
            sampler_for(geometry, "mantle-drive"), geometry, channel=0,
            wavelength_km=DRIVE_WAVELENGTH_KM, octaves=DRIVE_OCTAVES))
        print(f"  drive band ({DRIVE_WAVELENGTH_KM:g} km, "
              f"octaves {DRIVE_OCTAVES}): {statistic:.4f}")
        self.assertGreaterEqual(statistic, 0.9)
        self.assertLessEqual(statistic, 1.1)


class TheBandInKilometres(unittest.TestCase):
    """`WORK_ORDER_C03_10.md` §1.2: a length, not a fraction of the world."""

    def test_a_wavelength_reproduces_the_node_count_it_equals(self) -> None:
        # At 1024 px and 5 km/px the parent is 10,240 km, so half the parent
        # is the production drive's 5,120 km and the two spellings are the
        # same band. Byte-identical, not merely close: the cycle count is
        # exactly 2.0 in floating point and the envelope is the same array.
        geometry = WorldGeometry(AUDIT_SEED, 1024, 5)
        self.assertEqual(geometry.parent_km, 10240)
        sampler = sampler_for(geometry, "mantle-drive")
        by_nodes = periodic_noise(sampler, geometry, channel=0,
                                  nodes_coarsest=2, octaves=DRIVE_OCTAVES)
        by_length = periodic_noise(sampler, geometry, channel=0,
                                   wavelength_km=geometry.parent_km / 2,
                                   octaves=DRIVE_OCTAVES)
        self.assertEqual(by_nodes.tobytes(), by_length.tobytes())
        self.assertEqual(
            band_cycles(geometry, wavelength_km=DRIVE_WAVELENGTH_KM), 2.0)
        self.assertEqual(
            band_cycles(geometry, wavelength_km=STRENGTH_WAVELENGTH_KM), 8.0)

    def test_exactly_one_of_the_two_keywords(self) -> None:
        geometry = WorldGeometry(5, 512, 5)
        sampler = sampler_for(geometry)
        with self.assertRaises(ValueError):
            periodic_noise(sampler, geometry, channel=0, octaves=3)
        with self.assertRaises(ValueError):
            periodic_noise(sampler, geometry, channel=0, nodes_coarsest=4,
                           wavelength_km=1000.0, octaves=3)
        for wavelength in (0.0, -100.0):
            with self.subTest(wavelength=wavelength):
                with self.assertRaises(ValueError):
                    periodic_noise(sampler, geometry, channel=0,
                                   wavelength_km=wavelength, octaves=3)

    def test_a_non_integer_cycle_count_puts_the_low_edge_at_that_count(self) -> None:
        # 1,400 km across a 5,120 km parent is 3.657 cycles: no whole number
        # of node spacings, which the radial envelope does not need. The band
        # starts at the first mode the torus has at or above that radius and
        # every mode below it is empty.
        geometry = WorldGeometry(5, 512, 5)
        self.assertEqual(geometry.parent_km, 5120)
        wavelength_km = 1400.0
        nodes = band_cycles(geometry, wavelength_km=wavelength_km)
        self.assertNotAlmostEqual(nodes, round(nodes), places=6)
        field = periodic_noise(sampler_for(geometry), geometry, channel=0,
                               wavelength_km=wavelength_km, octaves=3)
        n = geometry.history_n
        cycles = np.fft.fftfreq(n, d=1.0 / n)
        radius = np.sqrt(cycles[:, None] ** 2 + cycles[None, :] ** 2)
        first_inside = float(radius[radius >= nodes].min())
        last_outside = float(radius[radius < nodes].max())
        edge = low_edge(field)
        print(f"\n  band at {wavelength_km:g} km on a "
              f"{geometry.parent_km} km parent: {nodes:.4f} cycles, low edge "
              f"{edge:.4f}, last empty mode {last_outside:.4f}")
        self.assertAlmostEqual(edge, first_inside, places=9)
        self.assertLess(last_outside, nodes)
        self.assertLessEqual(nodes, edge)

    def test_a_wavelength_longer_than_the_parent_is_allowed(self) -> None:
        # Below one cycle the world holds a piece of one mantle cell. The
        # field still exists and still has variation; it is simply the lowest
        # modes the torus has.
        geometry = WorldGeometry(5, 512, 5)
        field = periodic_noise(sampler_for(geometry), geometry, channel=0,
                               wavelength_km=4.0 * geometry.parent_km,
                               octaves=4)
        self.assertEqual(field.shape, (geometry.history_n,) * 2)
        self.assertAlmostEqual(float(field.std()), 1.0, delta=1e-9)
        self.assertEqual(dominant_cycles(field), 1)


class ScaleInvariance(unittest.TestCase):
    """`DESIGN.md` §2: the physics is in kilometres at every resolution."""

    def dominant_wavelength_km(self, pixels: int) -> tuple[float, int, int]:
        geometry = WorldGeometry(AUDIT_SEED, pixels, 5)
        drive = build_drive(geometry)
        cycles = dominant_cycles(drive.phi[0])
        return geometry.parent_km / cycles, cycles, geometry.parent_km

    def test_the_drive_peaks_at_the_same_wavelength_at_512_and_1024(self) -> None:
        small, small_k, small_parent = self.dominant_wavelength_km(512)
        big, big_k, big_parent = self.dominant_wavelength_km(1024)
        print(f"  drive potential, first keyframe: 512 px parent "
              f"{small_parent} km peaks at k = {small_k}, {small:.1f} km; "
              f"1024 px parent {big_parent} km peaks at k = {big_k}, "
              f"{big:.1f} km")
        # One spectral bin at the coarser reading: the neighbouring radius on
        # the 512-px grid is one cycle away, which is this many kilometres.
        bin_km = abs(small_parent / small_k - small_parent / (small_k + 1))
        self.assertLessEqual(abs(big - small), bin_km)
        self.assertAlmostEqual(small, DRIVE_WAVELENGTH_KM, delta=1e-9)
        self.assertAlmostEqual(big, DRIVE_WAVELENGTH_KM, delta=1e-9)

    def test_the_drive_process_id_is_the_one_the_engine_uses(self) -> None:
        self.assertEqual(DRIVE_PROCESS_ID, "mantle-drive")


if __name__ == "__main__":
    unittest.main()
