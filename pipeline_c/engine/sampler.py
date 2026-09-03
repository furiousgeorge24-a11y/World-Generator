"""Stateless SHA-256 address PRF for C4 world sampling."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._util import EngineRecordError, require_hash, require_id, require_int


KEY_SCHEDULE_ID = "pipeline-c-sha256-address-prf-v1"


SIGNED_64_MIN = -(2**63)
SIGNED_64_MAX = 2**63 - 1
UNSIGNED_64_MAX = 2**64 - 1
UNSIGNED_32_MAX = 2**32 - 1


def _text(value: str, label: str) -> bytes:
    require_id(value, label)
    encoded = value.encode("utf-8")
    if len(encoded) > UNSIGNED_32_MAX:
        raise EngineRecordError(f"{label} is too long for PRF encoding")
    return struct.pack(">I", len(encoded)) + encoded


@dataclass(frozen=True, slots=True)
class SampleAddress:
    world_id: str
    stage_id: str
    stage_version: str
    process_id: str
    x_m: int
    y_m: int
    channel: int = 0
    index: int = 0

    def __post_init__(self) -> None:
        require_hash(self.world_id, "world_id")
        for name in ("stage_id", "stage_version", "process_id"):
            require_id(getattr(self, name), name)
        require_int(self.x_m, "x_m", minimum=SIGNED_64_MIN, maximum=SIGNED_64_MAX)
        require_int(self.y_m, "y_m", minimum=SIGNED_64_MIN, maximum=SIGNED_64_MAX)
        require_int(self.channel, "channel", minimum=0, maximum=UNSIGNED_32_MAX)
        require_int(self.index, "index", minimum=0, maximum=UNSIGNED_64_MAX)

    def canonical_bytes(self) -> bytes:
        """Encode the frozen address in unambiguous network byte order."""

        return b"".join(
            (
                _text(KEY_SCHEDULE_ID, "key_schedule_id"),
                _text(self.world_id, "world_id"),
                _text(self.stage_id, "stage_id"),
                _text(self.stage_version, "stage_version"),
                _text(self.process_id, "process_id"),
                struct.pack(">q", self.x_m),
                struct.pack(">q", self.y_m),
                struct.pack(">I", self.channel),
                struct.pack(">Q", self.index),
            )
        )

    def to_record(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "index": self.index,
            "key_schedule_id": KEY_SCHEDULE_ID,
            "process_id": self.process_id,
            "stage_id": self.stage_id,
            "stage_version": self.stage_version,
            "world_id": self.world_id,
            "x_m": self.x_m,
            "y_m": self.y_m,
        }


@dataclass(frozen=True, slots=True)
class StageSampler:
    world_id: str
    stage_id: str
    stage_version: str
    process_id: str

    def __post_init__(self) -> None:
        require_hash(self.world_id, "world_id")
        for name in ("stage_id", "stage_version", "process_id"):
            require_id(getattr(self, name), name)

    @property
    def stage_key_sha256(self) -> str:
        material = b"".join(
            (
                _text(KEY_SCHEDULE_ID, "key_schedule_id"),
                _text(self.world_id, "world_id"),
                _text(self.stage_id, "stage_id"),
                _text(self.stage_version, "stage_version"),
                _text(self.process_id, "process_id"),
            )
        )
        return hashlib.sha256(material).hexdigest()

    def address(
        self,
        x_m: int,
        y_m: int,
        *,
        channel: int = 0,
        index: int = 0,
    ) -> SampleAddress:
        return SampleAddress(
            world_id=self.world_id,
            stage_id=self.stage_id,
            stage_version=self.stage_version,
            process_id=self.process_id,
            x_m=x_m,
            y_m=y_m,
            channel=channel,
            index=index,
        )

    def digest(
        self,
        x_m: int,
        y_m: int,
        *,
        channel: int = 0,
        index: int = 0,
    ) -> bytes:
        return hashlib.sha256(
            self.address(x_m, y_m, channel=channel, index=index).canonical_bytes()
        ).digest()

    def digest_hex(self, x_m: int, y_m: int, *, channel: int = 0,
                   index: int = 0) -> str:
        return self.digest(x_m, y_m, channel=channel, index=index).hex()

    def uint64(self, x_m: int, y_m: int, *, channel: int = 0,
               index: int = 0) -> int:
        return int.from_bytes(
            self.digest(x_m, y_m, channel=channel, index=index)[0:8],
            "big",
            signed=False,
        )

    def unit_float(self, x_m: int, y_m: int, *, channel: int = 0,
                   index: int = 0) -> float:
        first_53_bits = self.uint64(
            x_m, y_m, channel=channel, index=index
        ) >> 11
        return first_53_bits / float(2**53)

    def _prefix(self) -> bytes:
        """The address-independent head of every `canonical_bytes` from here.

        `SampleAddress.canonical_bytes` writes the key schedule and the four
        identifiers first and the four integers last, so the head is constant
        for one sampler. Hashing `prefix + pack(x, y, channel, index)` is the
        same byte string the per-address path builds, and it is what makes a
        whole-grid draw affordable.
        """
        return b"".join(
            (
                _text(KEY_SCHEDULE_ID, "key_schedule_id"),
                _text(self.world_id, "world_id"),
                _text(self.stage_id, "stage_id"),
                _text(self.stage_version, "stage_version"),
                _text(self.process_id, "process_id"),
            )
        )

    def unit_float_lattice(self, xs_m: Sequence[int], ys_m: Sequence[int], *,
                           channel: int = 0, index: int = 0) -> np.ndarray:
        """`unit_float` over the outer product of two address axes.

        Returns a `(len(ys_m), len(xs_m))` array whose entry `[j, i]` equals
        `unit_float(xs_m[i], ys_m[j], channel=channel, index=index)` exactly.
        Every address is validated the way `SampleAddress` validates it; the
        only thing this skips is rebuilding the constant head of the address
        encoding once per cell.
        """
        require_int(channel, "channel", minimum=0, maximum=UNSIGNED_32_MAX)
        require_int(index, "index", minimum=0, maximum=UNSIGNED_64_MAX)
        columns = [require_int(value, "x_m", minimum=SIGNED_64_MIN,
                               maximum=SIGNED_64_MAX) for value in xs_m]
        rows = [require_int(value, "y_m", minimum=SIGNED_64_MIN,
                            maximum=SIGNED_64_MAX) for value in ys_m]
        head = hashlib.sha256(self._prefix())
        tail = struct.pack(">IQ", channel, index)
        packed_columns = [struct.pack(">q", value) for value in columns]
        out = np.empty((len(rows), len(columns)), dtype=np.float64)
        scale = float(2**53)
        for j, y_m in enumerate(rows):
            packed_row = struct.pack(">q", y_m)
            row_out = out[j]
            for i, packed_column in enumerate(packed_columns):
                digest = head.copy()
                digest.update(packed_column + packed_row + tail)
                row_out[i] = (int.from_bytes(digest.digest()[0:8], "big",
                                             signed=False) >> 11) / scale
        return out

    def probe_record(self, x_m: int, y_m: int, *, channel: int = 0,
                     index: int = 0) -> dict[str, object]:
        digest = self.digest(x_m, y_m, channel=channel, index=index)
        prefix = int.from_bytes(digest[0:8], "big", signed=False)
        return {
            "address": self.address(
                x_m, y_m, channel=channel, index=index
            ).to_record(),
            "digest_sha256": digest.hex(),
            "unit_float": (prefix >> 11) / float(2**53),
            "uint64_prefix": prefix,
        }


__all__ = ["SampleAddress", "StageSampler"]
