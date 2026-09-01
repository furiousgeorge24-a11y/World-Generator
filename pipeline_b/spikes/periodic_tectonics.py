"""Private translation-only tectonic transport on a flat periodic atlas.

This module exists only for the boundaryless field-accretion feasibility
experiment.  Every spatial read and neighborhood operation wraps on both
axes.  Plates translate with wandering velocities but do not rotate: an
arbitrary Euclidean rotation is not a well-defined map of a square flat
torus.  The public structural builder is neither imported nor modified by
this module's callers.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from engine.rng import stage_rng
from engine.tectonics import (
    CONT_BORN,
    DT_MYR,
    EVENT_MEMORY,
    Structure,
)


@dataclass
class _PeriodicPlate:
    exists: np.ndarray
    cont: np.ndarray
    born: np.ndarray
    belt: np.ndarray
    belt_age: np.ndarray
    displacement_yx_km: np.ndarray

    @classmethod
    def empty(cls, n: int) -> "_PeriodicPlate":
        return cls(
            exists=np.zeros((n, n), bool),
            cont=np.zeros((n, n), bool),
            born=np.zeros((n, n), np.int16),
            belt=np.zeros((n, n), np.float32),
            belt_age=np.full((n, n), -1, np.int16),
            displacement_yx_km=np.zeros(2, np.float64),
        )


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(a, shift=(dy, dx), axis=(0, 1))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = np.asarray(mask, bool).copy()
    for _ in range(radius):
        source = out.copy()
        out |= _shift(source, 0, 1)
        out |= _shift(source, 0, -1)
        out |= _shift(source, 1, 0)
        out |= _shift(source, -1, 0)
    return out


def _fill_owner(label: np.ndarray) -> np.ndarray:
    """Fill periodic gaps in deterministic cardinal direction order."""
    lab = np.asarray(label, np.int32).copy()
    if not np.any(lab >= 0):
        raise ValueError("periodic owner fill has no surviving plate")
    empty = lab < 0
    for _ in range(lab.shape[0]):
        if not empty.any():
            break
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            neighbor = _shift(lab, dy, dx)
            take = empty & (neighbor >= 0)
            lab[take] = neighbor[take]
            empty = lab < 0
    if empty.any():
        raise AssertionError("periodic owner fill did not close")
    return lab


def build_periodic_structure(
        seed, cfg, *, world_km: float, coarse_km: float,
        partitioner, initial_age_sampler, continent_sampler,
        material_tag_sampler, cut_cells_yx=(0, 0)):
    """Transport a complete flat-torus structural atlas.

    The private callbacks all receive wrapped material coordinates.  No
    delivered crop origin, border, mask, or distance is accepted.
    """
    started = time.perf_counter()
    world_km = float(world_km)
    coarse_km = float(coarse_km)
    if world_km <= 0.0 or coarse_km <= 0.0:
        raise ValueError("invalid periodic structural extent")
    n = int(round(world_km / coarse_km))
    ck = world_km / n
    cut_y, cut_x = (int(cut_cells_yx[0]) % n,
                    int(cut_cells_yx[1]) % n)
    qy = ((np.arange(n) + cut_y) % n + 0.5) * ck
    qx = ((np.arange(n) + cut_x) % n + 0.5) * ck
    X, Y = np.meshgrid(qx, qy)

    label0 = np.asarray(partitioner(seed, Y, X, ck, cfg), np.int32)
    if label0.shape != (n, n):
        raise ValueError("periodic partitioner returned wrong shape")
    if label0.min() < 0 or label0.max() >= cfg.plates:
        raise ValueError("periodic partition labels out of range")
    ocean_born = np.asarray(
        initial_age_sampler(seed, Y, X, ck, cfg), np.int16)
    if ocean_born.shape != (n, n):
        raise ValueError("periodic initial age returned wrong shape")

    plates = [_PeriodicPlate.empty(n) for _ in range(cfg.plates)]
    for plate_id, plate in enumerate(plates):
        owned = label0 == plate_id
        plate.exists[owned] = True
        plate.born[owned] = ocean_born[owned]
        initial_cont = np.asarray(
            continent_sampler(plate_id, Y, X), bool)
        if initial_cont.shape != (n, n):
            raise ValueError("periodic continent sampler returned wrong shape")
        plate.cont |= initial_cont & owned
        plate.born[plate.cont] = CONT_BORN

    rng = stage_rng(seed, "periodic-tect-kinematics-v1")
    angle = rng.uniform(0.0, 2.0 * np.pi, cfg.plates)
    speed = rng.uniform(0.6, 1.4, cfg.plates) * cfg.plate_speed
    conv_hist = [np.zeros((n, n), bool) for _ in range(EVENT_MEMORY)]
    div_hist = [np.zeros((n, n), bool) for _ in range(EVENT_MEMORY)]

    def rasterize(Yq=None, Xq=None, *, include_tags=False):
        Yquery = Y if Yq is None else np.asarray(Yq, np.float64)
        Xquery = X if Xq is None else np.asarray(Xq, np.float64)
        claims = []
        for plate_id, plate in enumerate(plates):
            my = np.mod(
                Yquery - plate.displacement_yx_km[0], world_km)
            mx = np.mod(
                Xquery - plate.displacement_yx_km[1], world_km)
            iy = (np.floor(my / ck).astype(np.int64) - cut_y) % n
            ix = (np.floor(mx / ck).astype(np.int64) - cut_x) % n
            mask = plate.exists[iy, ix]
            cont = np.asarray(
                continent_sampler(plate_id, my, mx), bool)
            if cont.shape != mask.shape:
                raise ValueError(
                    "periodic continent sampler returned wrong query shape")
            tags = None
            if include_tags:
                tags = np.asarray(
                    material_tag_sampler(plate_id, my, mx))
                if tags.shape != mask.shape:
                    raise ValueError(
                        "periodic tag sampler returned wrong query shape")
                if not np.issubdtype(tags.dtype, np.integer):
                    raise ValueError("periodic material tags must be integers")
                tags = tags.astype(np.int64, copy=False)
            claims.append((mask, iy, ix, cont, tags))
        return claims

    def resolve(claims, era: int, *, mutate: bool):
        label = np.full((n, n), -1, np.int32)
        winning_cont = np.zeros((n, n), bool)
        winning_my = np.zeros((n, n), np.int64)
        winning_mx = np.zeros((n, n), np.int64)
        convergent = np.zeros((n, n), bool)
        for plate_id, (mask, iy, ix, cont, _) in enumerate(claims):
            if not mask.any():
                continue
            occupied = label >= 0
            collide = mask & occupied
            convergent |= collide
            win = mask & (~occupied | (cont & ~winning_cont))
            lose = mask & ~win
            displaced = collide & win

            if mutate and lose.any():
                lose_ocean = lose & ~cont
                plates[plate_id].exists[
                    iy[lose_ocean], ix[lose_ocean]] = False
                for other in np.unique(label[lose]):
                    selected = lose & (label == other)
                    amount = np.where(
                        cont[selected], 2.0, 1.0).astype(np.float32)
                    np.add.at(
                        plates[other].belt,
                        (winning_my[selected], winning_mx[selected]),
                        amount)
                    plates[other].belt_age[
                        winning_my[selected], winning_mx[selected]] = era
            if mutate and displaced.any():
                for other in np.unique(label[displaced]):
                    selected = displaced & (label == other)
                    plates[other].exists[
                        winning_my[selected], winning_mx[selected]] = False
                np.add.at(
                    plates[plate_id].belt,
                    (iy[displaced], ix[displaced]),
                    np.ones(int(displaced.sum()), np.float32))
                plates[plate_id].belt_age[
                    iy[displaced], ix[displaced]] = era

            label[win] = plate_id
            winning_cont[win] = cont[win]
            winning_my[win] = iy[win]
            winning_mx[win] = ix[win]
        return label, convergent

    label = label0.copy()
    for era in range(cfg.eras):
        angle += rng.normal(0.0, cfg.wander, cfg.plates)
        speed = np.clip(
            speed + rng.normal(
                0.0, 0.03 * cfg.plate_speed, cfg.plates),
            0.2 * cfg.plate_speed, 2.2 * cfg.plate_speed)
        for plate_id, plate in enumerate(plates):
            velocity = np.array([
                speed[plate_id] * np.sin(angle[plate_id]),
                speed[plate_id] * np.cos(angle[plate_id]),
            ])
            plate.displacement_yx_km[:] = np.mod(
                plate.displacement_yx_km + velocity, world_km)

        claims = rasterize()
        label, convergent = resolve(claims, era, mutate=True)
        gap = label < 0
        if gap.any():
            label = _fill_owner(label)
            for plate_id in np.unique(label[gap]):
                selected = gap & (label == plate_id)
                gy, gx = np.nonzero(selected)
                material_y = np.mod(
                    qy[gy]
                    - plates[plate_id].displacement_yx_km[0], world_km)
                material_x = np.mod(
                    qx[gx]
                    - plates[plate_id].displacement_yx_km[1], world_km)
                iy = ((np.floor(material_y / ck).astype(np.int64)
                       - cut_y) % n)
                ix = ((np.floor(material_x / ck).astype(np.int64)
                       - cut_x) % n)
                fresh = ~plates[plate_id].exists[iy, ix]
                plates[plate_id].exists[iy[fresh], ix[fresh]] = True
                plates[plate_id].cont[iy[fresh], ix[fresh]] = False
                plates[plate_id].born[iy[fresh], ix[fresh]] = era
        conv_hist[era % EVENT_MEMORY] = convergent
        div_hist[era % EVENT_MEMORY] = gap

    claims = rasterize()
    label, _ = resolve(claims, cfg.eras, mutate=False)
    if np.any(label < 0):
        label = _fill_owner(label)

    cont_frac = np.zeros((n, n), np.float64)
    born_f = np.zeros((n, n), np.float64)
    belt = np.zeros((n, n), np.float32)
    belt_age_weighted = np.zeros((n, n), np.float64)
    material_tag_samples = np.full((4, n, n), -1, np.int64)
    for sample_index, (oy, ox) in enumerate((
            (-0.25, -0.25), (-0.25, 0.25),
            (0.25, -0.25), (0.25, 0.25))):
        claims = rasterize(
            np.mod(Y + oy * ck, world_km),
            np.mod(X + ox * ck, world_km),
            include_tags=True)
        sample_label, _ = resolve(claims, cfg.eras, mutate=False)
        sample_cont = np.zeros((n, n), bool)
        sample_born = np.full((n, n), float(cfg.eras))
        sample_belt = np.zeros((n, n), np.float32)
        sample_belt_age = np.full((n, n), -1.0)
        for plate_id, (mask, iy, ix, cont, tags) in enumerate(claims):
            own = (sample_label == plate_id) & mask
            sample_cont[own] = cont[own]
            tagged = own & cont
            material_tag_samples[sample_index, tagged] = tags[tagged]
            sample_born[own] = plates[plate_id].born[iy[own], ix[own]]
            sample_belt[own] = plates[plate_id].belt[iy[own], ix[own]]
            sample_belt_age[own] = plates[plate_id].belt_age[
                iy[own], ix[own]]
        cont_frac += 0.25 * sample_cont
        born_f += 0.25 * sample_born
        belt += 0.25 * sample_belt
        belt_age_weighted += (
            0.25 * sample_belt * np.maximum(sample_belt_age, 0.0))

    cont = cont_frac >= 0.5
    with np.errstate(invalid="ignore"):
        belt_age = np.where(
            belt > 0.0,
            belt_age_weighted / np.maximum(belt, 1e-9),
            -1.0)

    ocean = ~cont
    ocean_weight = ocean.astype(np.float64)
    born_weighted = born_f * ocean_weight
    numerator = born_weighted.copy()
    weight = ocean_weight.copy()
    for dy, dx in (
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)):
        numerator += _shift(born_weighted, dy, dx)
        weight += _shift(ocean_weight, dy, dx)
    born_smooth = np.where(
        ocean & (weight > 0.0),
        numerator / np.maximum(weight, 1e-9), born_f)
    age = (cfg.eras - born_smooth) * DT_MYR

    conv_recent = np.zeros((n, n), bool)
    div_recent = np.zeros((n, n), bool)
    for event in conv_hist:
        conv_recent |= event
    for event in div_hist:
        div_recent |= event
    coast = cont & (
        _shift(ocean, 0, 1) | _shift(ocean, 0, -1)
        | _shift(ocean, 1, 0) | _shift(ocean, -1, 0))
    active_margin = coast & _dilate(conv_recent, 3)
    passive_margin = coast & ~_dilate(conv_recent, 3)
    alive = int(len(np.unique(label[label >= 0])))

    structure = Structure(
        n=n,
        world_km=world_km,
        frame_slice=(0, n),
        label=label,
        cont=cont,
        cont_frac=cont_frac,
        age_myr=age,
        belt=belt,
        belt_age_era=belt_age,
        conv_recent=conv_recent,
        div_recent=div_recent,
        coast=coast,
        active_margin=active_margin,
        passive_margin=passive_margin,
        initial_label=label0,
        alive_plates=alive,
        eras=cfg.eras,
        timings={"periodic_structure_s": time.perf_counter() - started},
    )
    structure._material_tag_samples = material_tag_samples
    structure._periodic = True
    structure._translation_only = True
    structure._cut_cells_yx = (cut_y, cut_x)
    structure._plate_displacements_yx_km = np.stack([
        plate.displacement_yx_km for plate in plates])
    return structure


def self_check() -> dict:
    sample = np.arange(25).reshape(5, 5)
    shifted = _shift(sample, 1, -2)
    if not np.array_equal(shifted, np.roll(sample, (1, -2), (0, 1))):
        raise AssertionError("periodic shift changed")
    labels = np.full((5, 5), -1, np.int32)
    labels[0, 0] = 3
    filled = _fill_owner(labels)
    if not np.all(filled == 3):
        raise AssertionError("periodic owner fill failed")
    edge = np.zeros((5, 5), bool)
    edge[0, 0] = True
    dilated = _dilate(edge, 1)
    if not (dilated[-1, 0] and dilated[0, -1]):
        raise AssertionError("periodic dilation does not cross seam")
    return {
        "passed": True,
        "wrapped_shift": True,
        "wrapped_owner_fill": True,
        "wrapped_dilation": True,
        "arbitrary_plate_rotation": False,
        "translation_only_disclosed": True,
    }
