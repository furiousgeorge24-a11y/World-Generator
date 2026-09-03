"""Gates on the search server: its config, its run control, and its gallery.

The Flask test client drives the app; the process pool is replaced by a
sequential stub, so these tests run one small world at a time in this process
and nothing is spawned. The worlds are real histories at the floor resolution
with the shortest history the params allow, because the endpoints are only
worth testing over cells that actually exist on disk.
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys
import tempfile
import time
import unittest

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

import search  # noqa: E402
import search_server  # noqa: E402

#: The cheapest run that still visits all three stages: two stage-1 cells on
#: one seed, one refinement, and one confirmation on the twelve development
#: seeds, at the floor resolution over the shortest allowed history.
TINY = {
    "space": {"pixels": 128, "scale_km": 5, "history_myr": 52,
              "max_cycles": 5},
    "stages": {"stage1_cells": 2, "stage1_seeds": 1, "stage2_top": 1,
               "stage2_perturbations": 0, "stage2_seeds": 1, "stage3_top": 1},
    "run": {"search_seed": 5, "window": 2},
}

CELLS_BEFORE_STAGE_3 = 3        # two stage-1 cells and one stage-2 cell
DEADLINE_S = 300.0


class Harness(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="c03_7_search_"))
        self.state = search_server.SearchState(
            executor_factory=search.SequentialExecutor, root=self.root)
        self.app = search_server.create_app(self.state)
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        if self.state.run is not None:
            self.state.run.stop()
        if self.state.thread is not None:
            self.state.thread.join(timeout=DEADLINE_S)
        shutil.rmtree(self.root, ignore_errors=True)

    def wait_for(self, predicate, what: str):
        deadline = time.monotonic() + DEADLINE_S
        while time.monotonic() < deadline:
            status = self.client.get("/api/status").get_json()
            if predicate(status):
                return status
            if not status["running"] and status["run_id"] is not None:
                self.fail(f"the run finished before {what}: {status}")
            time.sleep(0.05)
        self.fail(f"timed out waiting for {what}")


class TheConfig(Harness):
    def test_it_offers_every_knob_with_its_default_and_its_meaning(self) -> None:
        body = self.client.get("/api/config").get_json()
        names = {(knob["group"], knob["name"]) for knob in body["knobs"]}
        for group, source in (("screen", search.Screen()),
                              ("space", search.Space()),
                              ("stages", search.Stages())):
            for field in source.__dataclass_fields__:
                self.assertIn((group, field), names)
        self.assertIn(("run", "search_seed"), names)
        for knob in body["knobs"]:
            with self.subTest(knob=knob["name"]):
                self.assertTrue(knob["meaning"].strip())
                self.assertIn(knob["ctype"], ("int", "float", "set_int"))
                self.assertIn(knob["name"], body["defaults"][knob["group"]])
        self.assertEqual(body["development_seeds"],
                         list(search.DEVELOPMENT_SEEDS))
        self.assertIn("not an approval", body["note"])

    def test_the_defaults_round_trip_through_start(self) -> None:
        defaults = self.client.get("/api/config").get_json()["defaults"]
        sent = {group: dict(values) for group, values in defaults.items()}
        sent["stages"]["stage1_cells"] = 1
        sent["stages"]["stage1_seeds"] = 1
        sent["space"]["pixels"] = 128
        sent["space"]["history_myr"] = 52
        sent["space"]["max_cycles"] = 5
        response = self.client.post("/api/start", json=sent)
        self.assertEqual(response.status_code, 200)
        echoed = response.get_json()["config"]
        self.assertEqual(echoed["screen"], defaults["screen"])
        self.assertEqual(echoed["stages"]["stage2_top"],
                         defaults["stages"]["stage2_top"])
        self.assertEqual(echoed["space"]["strength_exponent_set"],
                         defaults["space"]["strength_exponent_set"])

    def test_the_damage_law_is_a_knob_that_round_trips(self) -> None:
        body = self.client.get("/api/config").get_json()
        knob = next(row for row in body["knobs"]
                    if row["name"] == "work_damage")
        self.assertEqual((knob["group"], knob["ctype"], knob["lo"],
                          knob["hi"]), ("space", "int", 0, 1))
        # The search's default is 0, which under the seam formulation is the
        # slip-rate law: a slipping fault stays weak. At 1 an open fault
        # dissipates almost nothing and heals shut, which is what C04 ran.
        self.assertEqual(body["defaults"]["space"]["work_damage"], 0)
        response = self.client.post("/api/start", json={
            "space": {"work_damage": 1, "pixels": 128, "history_myr": 52,
                      "max_cycles": 5},
            "stages": {"stage1_cells": 1, "stage1_seeds": 1},
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["config"]["space"]["work_damage"], 1)

    def test_the_drive_wavelength_is_two_knobs_that_round_trip(self) -> None:
        body = self.client.get("/api/config").get_json()
        for name, default in (("drive_wavelength_km_lo", 2560.0),
                              ("drive_wavelength_km_hi", 10240.0)):
            with self.subTest(name=name):
                knob = next(row for row in body["knobs"]
                            if row["name"] == name)
                self.assertEqual(
                    (knob["group"], knob["ctype"], knob["lo"], knob["hi"]),
                    ("space", "float", 640.0, 40960.0))
                self.assertIn("kilometres", knob["meaning"])
                self.assertEqual(body["defaults"]["space"][name], default)
        config = search_server.config_from({
            "space": {"drive_wavelength_km_lo": 800.0,
                      "drive_wavelength_km_hi": 3200.0}})
        self.assertEqual(config.space.bounds("drive_wavelength_km"),
                         (800.0, 3200.0))
        for bad in ({"drive_wavelength_km_lo": 639.0},
                    {"drive_wavelength_km_hi": 40961.0},
                    {"drive_wavelength_km_lo": 20000.0,
                     "drive_wavelength_km_hi": 1000.0}):
            with self.subTest(bad=bad):
                with self.assertRaises(search_server.ConfigError):
                    search_server.config_from({"space": bad})

    def test_the_seam_knobs_are_offered_and_round_trip(self) -> None:
        body = self.client.get("/api/config").get_json()
        knobs = {row["name"]: row for row in body["knobs"]}
        for name in ("seams", "crack_speed_km_per_myr_lo",
                     "crack_speed_km_per_myr_hi", "nucleations_per_step_set"):
            with self.subTest(name=name):
                self.assertIn(name, knobs)
                self.assertEqual(knobs[name]["group"], "space")
        self.assertIn("one cell wide", knobs["seams"]["meaning"])
        self.assertIn("rigid bodies", knobs["seams"]["meaning"])
        self.assertEqual(body["defaults"]["space"]["seams"], 2)
        self.assertEqual(
            body["defaults"]["space"]["crack_speed_km_per_myr_lo"], 10.0)
        self.assertEqual(
            body["defaults"]["space"]["crack_speed_km_per_myr_hi"], 200.0)
        self.assertEqual(
            body["defaults"]["space"]["nucleations_per_step_set"], [1, 2, 4])
        config = search_server.config_from({
            "space": {"seams": 0, "crack_speed_km_per_myr_lo": 20.0,
                      "crack_speed_km_per_myr_hi": 50.0,
                      "nucleations_per_step_set": "1, 3"}})
        self.assertEqual(config.space.seams, 0)
        self.assertEqual(config.space.bounds("crack_speed_km_per_myr"),
                         (20.0, 50.0))
        self.assertEqual(config.space.values("nucleations_per_step"), (1, 3))
        for bad in ({"crack_speed_km_per_myr_lo": 100.0,
                     "crack_speed_km_per_myr_hi": 20.0},
                    {"seams": 3}, {"crack_speed_km_per_myr_hi": 401.0}):
            with self.subTest(bad=bad):
                with self.assertRaises(search_server.ConfigError):
                    search_server.config_from({"space": bad})

    def test_the_solve_divisor_is_a_knob_that_round_trips(self) -> None:
        body = self.client.get("/api/config").get_json()
        knob = next(row for row in body["knobs"]
                    if row["name"] == "solve_divisor")
        self.assertEqual((knob["group"], knob["ctype"], knob["lo"],
                          knob["hi"]), ("space", "int", 1, 2))
        self.assertEqual(body["defaults"]["space"]["solve_divisor"], 1)
        response = self.client.post("/api/start", json={
            "space": {"solve_divisor": 2, "pixels": 128, "history_myr": 52,
                      "max_cycles": 5},
            "stages": {"stage1_cells": 1, "stage1_seeds": 1},
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["config"]["space"]["solve_divisor"], 2)

    def test_a_solve_divisor_of_three_is_refused(self) -> None:
        response = self.client.post("/api/start",
                                    json={"space": {"solve_divisor": 3}})
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertEqual(payload["code"], "invalid_knob")
        self.assertIn("solve_divisor", payload["error"])

    def test_a_damage_law_of_two_is_refused(self) -> None:
        response = self.client.post("/api/start",
                                    json={"space": {"work_damage": 2}})
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertEqual(payload["code"], "invalid_knob")
        self.assertIn("work_damage", payload["error"])

    def test_it_takes_a_set_as_text(self) -> None:
        config = search_server.config_from(
            {"space": {"strength_exponent_set": "4, 2, 2, 3"}})
        self.assertEqual(config.space.strength_exponent_set, (4, 2, 3))

    def test_it_refuses_a_knob_it_cannot_use(self) -> None:
        for body, fragment in (
            ({"screen": {"weak_min": 0.4, "weak_max": 0.1}}, "weak_min"),
            ({"screen": {"plates_min": 9, "plates_max": 2}}, "plates_min"),
            ({"space": {"stiffness_fraction_lo": 0.0}}, "between 0.02"),
            ({"space": {"heal_time_myr_lo": 600.0}}, "must not exceed"),
            ({"space": {"pixels": 333}}, "pixels"),
            ({"screen": {"network_share_min": 4.0}}, "network_share_min"),
            ({"stages": {"stage1_seeds": "many"}}, "whole number"),
            ({"space": {"strength_exponent_set": ""}}, "at least one"),
        ):
            with self.subTest(body=body):
                response = self.client.post("/api/start", json=body)
                self.assertEqual(response.status_code, 400)
                payload = response.get_json()
                self.assertEqual(payload["code"], "invalid_knob")
                self.assertIn(fragment, payload["error"])


class TheRun(Harness):
    def test_it_starts_stops_and_refuses_a_second_run(self) -> None:
        self.assertEqual(self.client.get("/api/status").get_json()["stage"],
                         "idle")
        self.assertEqual(self.client.post("/api/stop", json={}).status_code,
                         409)

        started = self.client.post("/api/start", json=TINY)
        self.assertEqual(started.status_code, 200)
        run_id = started.get_json()["run_id"]
        self.assertTrue(run_id)

        second = self.client.post("/api/start", json=TINY)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(second.get_json()["code"], "run_active")

        self.wait_for(lambda status: status["cells_done"] >= 1, "a first cell")
        stopped = self.client.post("/api/stop", json={})
        self.assertEqual(stopped.status_code, 200)
        self.assertTrue(stopped.get_json()["stopping"])

        self.state.thread.join(timeout=DEADLINE_S)
        final = self.client.get("/api/status").get_json()
        self.assertFalse(final["running"])
        self.assertIsNone(final["error"])
        self.assertEqual(final["run_id"], run_id)
        self.assertGreaterEqual(final["cells_done"], 1)

    def test_the_run_is_on_disk_and_can_be_reopened(self) -> None:
        run_id = self.client.post("/api/start", json=TINY).get_json()["run_id"]
        self.wait_for(lambda status: status["cells_done"] >= 2, "two cells")
        self.client.post("/api/stop", json={})
        self.state.thread.join(timeout=DEADLINE_S)

        directory = self.root / run_id
        self.assertTrue((directory / "config.json").is_file())
        self.assertTrue((directory / "cells.jsonl").is_file())
        on_disk = search.load_cells(directory)
        self.assertGreaterEqual(len(on_disk), 2)

        runs = self.client.get("/api/runs").get_json()["runs"]
        self.assertEqual([row["run_id"] for row in runs], [run_id])
        self.assertEqual(runs[0]["cells"], len(on_disk))

        reopened = self.client.get(
            f"/api/cells?after=-1&run={run_id}").get_json()
        self.assertEqual(reopened["run_id"], run_id)
        self.assertEqual(len(reopened["cells"]), len(on_disk))


class ReopeningALongRun(Harness):
    """A reopened run keeps every passed cell inside the page limit."""

    def _fake_run(self, name: str, count: int, passed_at: set[int]) -> None:
        directory = self.root / name
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps(search.SearchConfig().to_json()), encoding="utf-8")
        with (directory / "cells.jsonl").open("w", encoding="utf-8") as fh:
            for index in range(count):
                fh.write(json.dumps({
                    "id": f"c{index:05d}", "index": index, "stage": 1,
                    "passed": index in passed_at, "invalid": False,
                    "finding": False, "soft_score": 1.0, "sheets": [],
                }) + "\n")

    def test_passed_cells_survive_the_limit_and_the_rest_are_the_newest(self) -> None:
        self._fake_run("20260101T000000Z-s1", 12, {1, 7})
        body = self.client.get(
            "/api/cells?after=-1&limit=5&keep_passed=1"
            "&run=20260101T000000Z-s1").get_json()
        self.assertEqual([c["index"] for c in body["cells"]], [1, 7, 9, 10, 11])
        self.assertEqual(body["total"], 12)

    def test_without_the_flag_the_page_is_the_oldest_cells(self) -> None:
        self._fake_run("20260101T000000Z-s1", 12, {7})
        body = self.client.get(
            "/api/cells?after=-1&limit=5&run=20260101T000000Z-s1").get_json()
        self.assertEqual([c["index"] for c in body["cells"]], [0, 1, 2, 3, 4])


class TheCells(Harness):
    def setUp(self) -> None:
        super().setUp()
        self.run_id = self.client.post("/api/start", json=TINY).get_json()["run_id"]
        self.wait_for(lambda status: status["cells_done"] >= CELLS_BEFORE_STAGE_3,
                      "the first three cells")
        self.client.post("/api/stop", json={})
        self.state.thread.join(timeout=DEADLINE_S)
        self.cells = self.client.get("/api/cells?after=-1").get_json()["cells"]

    def test_a_cell_carries_its_dials_metrics_and_verdict(self) -> None:
        cell = self.cells[0]
        self.assertEqual(sorted(cell["dials"]), sorted(search.DIAL_NAMES))
        self.assertEqual(len(cell["worlds"]), TINY["stages"]["stage1_seeds"])
        for key in ("weak_final", "weak_peak", "weak_drift", "plate_count",
                    "network_share", "edge_fraction", "residual_max"):
            self.assertIn(key, cell["worlds"][0])
        self.assertEqual(sorted(cell["terms"]),
                         sorted(name for name, _lo, _hi
                                in search.term_bounds(search.Screen())))
        self.assertIn(cell["passed"], (True, False))
        self.assertIn(cell["invalid"], (True, False))
        self.assertEqual(cell["sheets"], list(search.CELL_SHEETS))

    def test_the_cells_come_back_in_completion_order(self) -> None:
        indices = [cell["index"] for cell in self.cells]
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(indices, list(range(len(indices))))
        self.assertEqual([cell["stage"] for cell in self.cells][:2], [1, 1])

    def test_after_and_limit_page_through_them(self) -> None:
        first = self.client.get("/api/cells?after=-1&limit=1").get_json()
        self.assertEqual(len(first["cells"]), 1)
        self.assertEqual(first["cells"][0]["index"], 0)
        self.assertEqual(first["total"], len(self.cells))

        rest = self.client.get("/api/cells?after=0&limit=1").get_json()
        self.assertEqual(rest["cells"][0]["index"], 1)

        past_the_end = self.client.get(
            f"/api/cells?after={self.cells[-1]['index']}").get_json()
        self.assertEqual(past_the_end["cells"], [])

    def test_a_bad_page_query_is_refused(self) -> None:
        response = self.client.get("/api/cells?after=soon")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "invalid_query")

    def test_a_sheet_comes_back_as_a_png(self) -> None:
        cell = self.cells[0]
        for sheet in search.CELL_SHEETS:
            with self.subTest(sheet=sheet):
                response = self.client.get(
                    f"/api/cell/{cell['id']}/{sheet}.png")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.mimetype, "image/png")
                self.assertTrue(response.data.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_a_bad_sheet_request_is_a_404(self) -> None:
        for path in (
            "/api/cell/c99999/plates.png",
            f"/api/cell/{self.cells[0]['id']}/nosuchview.png",
            "/api/cell/..%2F..%2Fconfig/plates.png",
            f"/api/cell/{self.cells[0]['id']}/plates.png?run=nowhere",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)


class ThePage(Harness):
    def test_the_page_is_served_without_caching(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("no-store", response.headers["Cache-Control"])
        body = response.get_data(as_text=True)
        response.close()
        self.assertIn("/api/config", body)
        self.assertIn("/api/cells", body)
        self.assertIn("not an approval", body)


if __name__ == "__main__":
    unittest.main()


class TheClockAfterARun(Harness):
    """A finished run keeps the elapsed time and rate it ended with."""

    def test_elapsed_and_rate_stop_moving_once_the_run_is_over(self) -> None:
        self.client.post("/api/start", json=TINY)
        self.wait_for(lambda status: status["cells_done"] >= 1, "one cell")
        self.client.post("/api/stop", json={})
        self.state.thread.join(timeout=DEADLINE_S)

        first = self.client.get("/api/status").get_json()
        self.assertFalse(first["running"])
        time.sleep(1.1)
        second = self.client.get("/api/status").get_json()
        self.assertEqual(second["elapsed_s"], first["elapsed_s"])
        self.assertEqual(second["cells_per_minute"], first["cells_per_minute"])
