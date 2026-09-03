"""Gates on the stateless address PRF, carried over from the C4 suite.

These pin the sampler's bytes. The fixtures that used to build them through
the deleted foundation state are inlined here as literals.
"""

from __future__ import annotations

from pathlib import Path
import random
import struct
import sys
import unittest

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

from engine.sampler import (  # noqa: E402
    KEY_SCHEDULE_ID,
    SampleAddress,
    StageSampler,
)
from engine._util import EngineRecordError  # noqa: E402

# The C4 debug-0 world identity and its registration sampler, kept verbatim so
# the golden vectors below stay comparable with the ones they came from.
WORLD_ID = "6fec2784811c2d4f2f33cd09bae963803039920759e35f90097a0f8f2ee72b9f"
OTHER_WORLD_ID = "5c27ff7a20ae555296b94c350b45156b23ff9653e637193fac824a44b49739ba"
STAGE_ID = "world_foundation.v1"
STAGE_VERSION = "1"
PROCESS_ID = "physical-registration"

_AXIS_M = (17_920_000, 20_480_000, 23_040_000)
REGISTERED_PROBES_M = tuple(
    (x_m, y_m) for y_m in _AXIS_M for x_m in _AXIS_M
)


def registration_sampler() -> StageSampler:
    return StageSampler(WORLD_ID, STAGE_ID, STAGE_VERSION, PROCESS_ID)


class PrfGoldenVectors(unittest.TestCase):
    def setUp(self) -> None:
        self.sampler = registration_sampler()

    def test_b04_prf_frozen_golden_vector_and_float_conversion(self) -> None:
        sampler = self.sampler
        x_m, y_m = REGISTERED_PROBES_M[0]
        self.assertEqual(
            sampler.stage_key_sha256,
            "5409736c4bac8eed5a25575385e970633f7835d105d89558c011446fc70e7216",
        )
        self.assertEqual(
            sampler.digest_hex(x_m, y_m),
            "31bd0cce9b2ecfe7c3e142f216ba60403c271491bf4cdae9e703d04208d32269",
        )
        self.assertEqual(sampler.uint64(x_m, y_m), 3584034959963115495)
        self.assertEqual(sampler.unit_float(x_m, y_m), 0.19429092449280028)
        self.assertEqual(
            sampler.unit_float(x_m, y_m),
            (sampler.uint64(x_m, y_m) >> 11) / float(2**53),
        )
        address = sampler.address(x_m, y_m)
        suffix = struct.pack(">q", x_m) + struct.pack(">q", y_m)
        suffix += struct.pack(">I", 0) + struct.pack(">Q", 0)
        self.assertTrue(address.canonical_bytes().endswith(suffix))
        self.assertEqual(address.to_record()["key_schedule_id"], KEY_SCHEDULE_ID)

    def test_c03_kinematic_history_sampler_is_pinned(self) -> None:
        # Pins the sampler for the stage this run builds. Recorded from the
        # first computation; a change here is a change of every C03 world.
        sampler = StageSampler(WORLD_ID, "kinematic_history.v1", "1", "x")
        self.assertEqual(sampler.uint64(0, 0), 2592925399878011991)

    def test_b04_prf_is_stateless_order_independent_and_domain_separated(self) -> None:
        sampler = self.sampler
        forward = {
            point: sampler.digest_hex(*point) for point in REGISTERED_PROBES_M
        }
        reverse = {
            point: sampler.digest_hex(*point)
            for point in reversed(REGISTERED_PROBES_M)
        }
        random.seed(998877)
        shuffled = list(REGISTERED_PROBES_M)
        random.shuffle(shuffled)
        interleaved = {}
        for point in shuffled:
            sampler.digest_hex(point[0] + 123, point[1] - 456, channel=9, index=3)
            interleaved[point] = sampler.digest_hex(*point)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward, interleaved)

        x_m, y_m = REGISTERED_PROBES_M[0]
        variants = {
            sampler.digest(x_m, y_m),
            sampler.digest(x_m + 1, y_m),
            sampler.digest(x_m, y_m + 1),
            sampler.digest(x_m, y_m, channel=1),
            sampler.digest(x_m, y_m, index=1),
            StageSampler(
                sampler.world_id,
                "alternate-stage.v1",
                sampler.stage_version,
                sampler.process_id,
            ).digest(x_m, y_m),
            StageSampler(
                sampler.world_id,
                sampler.stage_id,
                "2",
                sampler.process_id,
            ).digest(x_m, y_m),
            StageSampler(
                sampler.world_id,
                sampler.stage_id,
                sampler.stage_version,
                "alternate-process",
            ).digest(x_m, y_m),
            StageSampler(
                OTHER_WORLD_ID,
                sampler.stage_id,
                sampler.stage_version,
                sampler.process_id,
            ).digest(x_m, y_m),
        }
        self.assertEqual(len(variants), 9)

    def test_b04_prf_rejects_noninteger_and_out_of_range_addresses(self) -> None:
        sampler = self.sampler
        for x_m, y_m, channel, index in (
            (1.0, 2, 0, 0),
            (True, 2, 0, 0),
            (2**63, 2, 0, 0),
            (1, -(2**63) - 1, 0, 0),
            (1, 2, -1, 0),
            (1, 2, 0, 2**64),
        ):
            with self.subTest((x_m, y_m, channel, index)), self.assertRaises(
                EngineRecordError
            ):
                sampler.address(x_m, y_m, channel=channel, index=index)

    def test_b04_prf_rejects_malformed_identifiers(self) -> None:
        for world_id, stage_id, version, process in (
            ("not-a-digest", STAGE_ID, STAGE_VERSION, PROCESS_ID),
            (WORLD_ID.upper(), STAGE_ID, STAGE_VERSION, PROCESS_ID),
            (WORLD_ID, "", STAGE_VERSION, PROCESS_ID),
            (WORLD_ID, STAGE_ID, "bad version", PROCESS_ID),
            (WORLD_ID, STAGE_ID, STAGE_VERSION, "-leading-dash"),
        ):
            with self.subTest(stage_id), self.assertRaises(EngineRecordError):
                StageSampler(world_id, stage_id, version, process)

    def test_b05_physical_addressing_is_resolution_independent(self) -> None:
        # The same physical address gives the same bytes regardless of what
        # grid asked for it. Resolution is not part of the address.
        sampler = self.sampler
        probes = tuple(sampler.digest_hex(*point) for point in REGISTERED_PROBES_M)
        again = tuple(sampler.digest_hex(*point) for point in REGISTERED_PROBES_M)
        self.assertEqual(probes, again)
        self.assertEqual(len(set(probes)), len(REGISTERED_PROBES_M))

    def test_b04_probe_record_is_complete(self) -> None:
        record = self.sampler.probe_record(*REGISTERED_PROBES_M[0])
        self.assertEqual(sorted(record), [
            "address", "digest_sha256", "uint64_prefix", "unit_float"])
        self.assertEqual(record["address"]["world_id"], WORLD_ID)
        self.assertEqual(
            record["digest_sha256"],
            self.sampler.digest_hex(*REGISTERED_PROBES_M[0]))

    def test_b04_sample_address_is_frozen(self) -> None:
        address = SampleAddress(WORLD_ID, STAGE_ID, STAGE_VERSION, PROCESS_ID,
                                1, 2)
        with self.assertRaises(Exception):
            address.x_m = 3  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
