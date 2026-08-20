"""Contract tests for the online in-circle observable recorder (IET-20).

The recorder's whole justification is that it is invisible unless asked for, so
the first two tests are about it NOT doing anything.  The rest pin the physics
of the synthetic two-colour instrument, which is the part that could silently
produce plausible-but-wrong numbers.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from jax_fem_am.verification.online_observables import (
    C2,
    OnlineObservableRecorder,
    invert_two_colour_ratio,
    recorder_from_args,
    uniform_field_self_check,
    _spectral_radiance,
)

WAVELENGTHS = (0.95e-6, 1.05e-6)


class RecorderIsOffByDefaultTest(unittest.TestCase):
    def test_absent_flag_yields_no_recorder(self):
        # An args namespace that predates the flags entirely.
        self.assertIsNone(recorder_from_args(SimpleNamespace()))

    def test_explicit_false_yields_no_recorder(self):
        self.assertIsNone(
            recorder_from_args(SimpleNamespace(online_observables=False))
        )

    def test_enabled_flag_yields_a_recorder_with_the_documented_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = recorder_from_args(
                SimpleNamespace(online_observables=True, output_dir=temporary)
            )
            self.assertIsInstance(recorder, OnlineObservableRecorder)
            self.assertEqual(recorder.window_s, (0.45, 0.90))
            self.assertEqual(recorder.every, 1)
            self.assertEqual(recorder.probes_m, [])
            # wavelengths are given in um on the CLI and stored in m
            self.assertAlmostEqual(recorder.wavelengths_m[0], 0.95e-6)
            self.assertAlmostEqual(recorder.wavelengths_m[1], 1.05e-6)

    def test_probe_triples_are_parsed_and_bad_ones_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = recorder_from_args(SimpleNamespace(
                online_observables=True, output_dir=temporary,
                online_observables_probes="1e-3,2e-3,4.4e-4;2e-3,2e-3,4.4e-4"))
            self.assertEqual(recorder.probes_m,
                             [(1e-3, 2e-3, 4.4e-4), (2e-3, 2e-3, 4.4e-4)])
            with self.assertRaises(ValueError):
                recorder_from_args(SimpleNamespace(
                    online_observables=True, output_dir=temporary,
                    online_observables_probes="1e-3,2e-3"))

    def test_inverted_window_is_rejected_rather_than_silently_recording_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                recorder_from_args(SimpleNamespace(
                    online_observables=True, output_dir=temporary,
                    online_observables_window="0.9,0.4"))


class TwoColourPhysicsTest(unittest.TestCase):
    def test_second_radiation_constant_matches_codata(self):
        self.assertAlmostEqual(C2, 1.438776877e-2, places=10)

    def test_uniform_field_inverts_back_to_itself(self):
        ok, rows = uniform_field_self_check(WAVELENGTHS)
        self.assertTrue(ok, rows)
        for row in rows:
            self.assertLess(row["abs_error_K"], 1.0e-4)

    def test_planck_ratio_is_monotone_in_temperature(self):
        temps = np.linspace(500.0, 6000.0, 60)
        ratio = (_spectral_radiance(temps, WAVELENGTHS[0])
                 / _spectral_radiance(temps, WAVELENGTHS[1]))
        self.assertTrue(np.all(np.diff(ratio) > 0.0),
                        "ratio must increase with T or the bisection is invalid")

    def test_inversion_is_not_clamped_at_any_boiling_point(self):
        # A field far above IN625's boiling point must still invert honestly:
        # clamping here would be calibration (D-V2-26).
        hot = np.full(16, 5200.0)
        s1 = float(_spectral_radiance(hot, WAVELENGTHS[0]).mean())
        s2 = float(_spectral_radiance(hot, WAVELENGTHS[1]).mean())
        self.assertAlmostEqual(
            invert_two_colour_ratio(s1 / s2, WAVELENGTHS), 5200.0, places=3)

    def test_cold_cells_barely_move_the_reading(self):
        # The claim registered in D-V2-24: at ~1 um the Wien tail makes an
        # unthresholded integral safe. Measured, not asserted.
        hot = np.full(8, 3000.0)
        with_cold = np.concatenate([hot, np.full(1968, 400.0)])
        def read(field):
            s1 = float(_spectral_radiance(field, WAVELENGTHS[0]).mean())
            s2 = float(_spectral_radiance(field, WAVELENGTHS[1]).mean())
            return invert_two_colour_ratio(s1 / s2, WAVELENGTHS)
        self.assertLess(abs(read(with_cold) - read(hot)), 1.0)

    def test_a_ratio_outside_the_bracket_reports_missing_rather_than_guessing(self):
        self.assertIsNone(invert_two_colour_ratio(0.0, WAVELENGTHS))
        self.assertIsNone(invert_two_colour_ratio(float("nan"), WAVELENGTHS))


class _FakeFE:
    """Two stacked 2x2 layers of unit cells, so top-layer selection is testable."""

    def __init__(self):
        xs = np.array([0.0, 1.0e-3, 2.0e-3])
        zs = np.array([0.0, 4.0e-5, 8.0e-5])
        points, index = [], {}
        for k, z in enumerate(zs):
            for j, y in enumerate(xs):
                for i, x in enumerate(xs):
                    index[(i, j, k)] = len(points)
                    points.append([x, y, z])
        self.points = np.asarray(points)
        cells = []
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    cells.append([
                        index[(i, j, k)], index[(i + 1, j, k)],
                        index[(i + 1, j + 1, k)], index[(i, j + 1, k)],
                        index[(i, j, k + 1)], index[(i + 1, j, k + 1)],
                        index[(i + 1, j + 1, k + 1)], index[(i, j + 1, k + 1)],
                    ])
        self.cells = np.asarray(cells)


class _FakeProblem:
    def __init__(self):
        self.fes = [_FakeFE()]


def _step(dt, time_marker):
    return SimpleNamespace(dt=dt, global_step=time_marker, mode="scan",
                           laser_switch=1.0, laser_center=np.array([1.0e-3, 1.0e-3]))


class RecorderBehaviourTest(unittest.TestCase):
    def _recorder(self, temporary, **overrides):
        options = dict(
            spot_center_m=(1.0e-3, 1.0e-3), spot_diameter_m=4.0e-3,
            threshold_c=1000.0, range_max_c=3000.0,
            wavelengths_m=WAVELENGTHS, window_s=(0.0, 1.0), every=1,
            probes_m=[(0.0, 0.0, 8.0e-5)], layer_thickness_m=4.0e-5)
        options.update(overrides)
        return OnlineObservableRecorder(temporary, **options)

    def test_nothing_is_written_when_no_step_falls_in_the_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = self._recorder(temporary, window_s=(10.0, 20.0))
            problem = _FakeProblem()
            temperature = np.full((len(problem.fes[0].points), 1), 2000.0)
            for _ in range(5):
                recorder.observe(problem, temperature, _step(1.0e-3, 0))
            recorder.finalize()
            self.assertEqual(recorder.rows_written, 0)
            self.assertFalse(list(Path(temporary).glob("online_observables*")))

    def test_records_the_adopted_average_over_top_layer_cells_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = self._recorder(temporary)
            problem = _FakeProblem()
            points = problem.fes[0].points
            # top layer hot, bottom layer cold -- a top-layer-only reading must
            # not be dragged down by the sub-surface cells
            temperature = np.where(points[:, 2:3] > 3.0e-5, 2000.0, 400.0)
            recorder.observe(problem, temperature, _step(1.0e-3, 7))
            recorder.finalize()

            rows = [json.loads(line) for line in
                    (Path(temporary) / "online_observables.jsonl")
                    .read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            # 4 top-layer cells, each with 4 nodes at z=4e-5 (2000 K) and 4 at
            # z=8e-5 (2000 K) -> cell temperature 2000 K exactly
            self.assertEqual(row["n_hot"], 4)
            self.assertAlmostEqual(row["avg_K"], 2000.0, places=6)
            self.assertAlmostEqual(row["full_spot_avg_K"], 2000.0, places=6)
            self.assertAlmostEqual(row["two_colour_K"], 2000.0, places=3)
            self.assertFalse(row["two_colour_over_range"])
            self.assertEqual(row["time_s"], 1.0e-3)
            self.assertEqual(len(row["probe_K"]), 1)

            meta = json.loads((Path(temporary) / "online_observables_meta.json")
                              .read_text(encoding="utf-8"))
            self.assertEqual(meta["gauge_cells"], 4)
            self.assertTrue(meta["two_colour"]["uniform_field_self_check_passed"])

    def test_time_is_the_running_sum_of_dt(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = self._recorder(temporary)
            problem = _FakeProblem()
            temperature = np.full((len(problem.fes[0].points), 1), 2000.0)
            for _ in range(4):
                recorder.observe(problem, temperature, _step(2.5e-3, 0))
            recorder.finalize()
            rows = [json.loads(line) for line in
                    (Path(temporary) / "online_observables.jsonl")
                    .read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["time_s"] for r in rows],
                             [2.5e-3, 5.0e-3, 7.5e-3, 10.0e-3])

    def test_every_n_subsamples_without_disturbing_the_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = self._recorder(temporary, every=3)
            problem = _FakeProblem()
            temperature = np.full((len(problem.fes[0].points), 1), 2000.0)
            for _ in range(9):
                recorder.observe(problem, temperature, _step(1.0e-3, 0))
            recorder.finalize()
            rows = [json.loads(line) for line in
                    (Path(temporary) / "online_observables.jsonl")
                    .read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["step_index"] for r in rows], [0, 3, 6])
            self.assertEqual([round(r["time_s"], 9) for r in rows],
                             [1.0e-3, 4.0e-3, 7.0e-3])

    def test_a_recorder_fault_disables_the_recorder_and_never_raises(self):
        # An observability feature must not be able to kill a 7.5 h run.
        with tempfile.TemporaryDirectory() as temporary:
            recorder = self._recorder(temporary, spot_center_m=(9.0, 9.0))
            problem = _FakeProblem()
            temperature = np.full((len(problem.fes[0].points), 1), 2000.0)
            recorder.observe(problem, temperature, _step(1.0e-3, 0))
            self.assertIsNotNone(recorder.disabled_reason)
            self.assertIn("gauge cell set is empty", recorder.disabled_reason)
            recorder.observe(problem, temperature, _step(1.0e-3, 0))
            recorder.finalize()

    def test_over_range_is_flagged_but_the_reading_is_not_clamped(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = self._recorder(temporary)
            problem = _FakeProblem()
            temperature = np.full((len(problem.fes[0].points), 1), 4800.0)
            recorder.observe(problem, temperature, _step(1.0e-3, 0))
            recorder.finalize()
            row = json.loads((Path(temporary) / "online_observables.jsonl")
                             .read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(row["two_colour_over_range"])
            self.assertGreater(row["two_colour_K"], 3273.15)
            self.assertAlmostEqual(row["two_colour_K"], 4800.0, places=3)


if __name__ == "__main__":
    unittest.main()
