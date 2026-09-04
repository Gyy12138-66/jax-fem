"""Per-physics linear-solver registry (jax_fem_am.solvers.linear) and its wiring
into the acceleration wrapper's solver patch."""
import json
import unittest
from types import SimpleNamespace

import numpy as np

from jax_fem_am.solvers import linear as registry
from jax_fem_am.simulation import acceleration


class TransientThermal:  # scope detection is by class name (see acceleration)
    def __init__(self):
        self.fes = [SimpleNamespace(points=np.zeros((4, 3)), vec=1)]


class ThermoMechanical:
    def __init__(self, prefer_direct=False):
        self.fes = [SimpleNamespace(points=np.zeros((4, 3)), vec=3)]
        if prefer_direct:
            self.prefer_direct_linear_solver = True


class _RecordingCustom:
    label = "recording_custom"

    def __init__(self):
        self.bound = []

    def __deepcopy__(self, memo):  # option injection deep-copies the Newton dict
        return self

    def bind_problem(self, problem):
        self.bound.append(problem)

    def __call__(self, A, b, x0, linear_options):  # pragma: no cover - not exercised
        raise AssertionError("not called in these tests")


class SpecParsingTest(unittest.TestCase):
    def test_keep_words_map_to_none(self):
        for value in (None, "", "keep", "preserve", "KEEP"):
            self.assertIsNone(registry.parse_linear_solver_spec(value))

    def test_shorthand_forms(self):
        self.assertEqual(
            registry.parse_linear_solver_spec("jax:cg:jacobi"),
            {"backend": "jax", "method": "cg", "precond": "jacobi"},
        )
        self.assertEqual(
            registry.parse_linear_solver_spec("jax:bicgstab:none"),
            {"backend": "jax", "method": "bicgstab", "precond": "none"},
        )
        self.assertEqual(
            registry.parse_linear_solver_spec("pardiso:phase23"),
            {"backend": "pardiso", "mode": "phase23"},
        )
        self.assertEqual(registry.parse_linear_solver_spec("spsolve"), {"backend": "spsolve"})
        self.assertEqual(
            registry.parse_linear_solver_spec("pyamg:cg:rigid_body")["near_nullspace"],
            "rigid_body",
        )

    def test_json_form_and_bool_precond(self):
        spec = registry.parse_linear_solver_spec(
            json.dumps({"backend": "jax", "method": "cg", "precond": True, "tol": "1e-6",
                        "check_residual": "false", "maxiter": 200})
        )
        self.assertEqual(spec["precond"], "jacobi")
        self.assertEqual(spec["tol"], 1e-6)
        self.assertIs(spec["check_residual"], False)
        self.assertEqual(spec["maxiter"], 200)

    def test_rejects_unknown_backend_key_and_choice(self):
        with self.assertRaises(ValueError):
            registry.parse_linear_solver_spec("mumps")
        with self.assertRaises(ValueError):
            registry.parse_linear_solver_spec({"backend": "jax", "methd": "cg"})
        with self.assertRaises(ValueError):
            registry.parse_linear_solver_spec("jax:minres")
        with self.assertRaises(ValueError):
            registry.parse_linear_solver_spec("pardiso:phase99")
        with self.assertRaises(ValueError):
            registry.parse_linear_solver_spec("spsolve:extra")


class BlockBuildingTest(unittest.TestCase):
    def setUp(self):
        registry.reset_shared_solvers()

    def test_jax_block_maps_precond_and_forwards_controls(self):
        block = registry.build_linear_block(
            {"backend": "jax", "method": "cg", "precond": "jacobi", "tol": 1e-6,
             "atol": 1e-6, "maxiter": 10000, "check_residual": False, "check_factor": 50}
        )
        self.assertEqual(
            block,
            {"jax_solver": {"precond": True, "method": "cg", "tol": 1e-6, "atol": 1e-6,
                            "maxiter": 10000, "check_residual": False, "check_factor": 50.0}},
        )
        no_pc = registry.build_linear_block("jax:bicgstab:none")
        self.assertEqual(no_pc["jax_solver"], {"precond": False, "method": "bicgstab"})

    def test_pardiso_block_is_shared_per_mode(self):
        a = registry.build_linear_block({"backend": "pardiso", "mode": "phase23"})
        b = registry.build_linear_block("pardiso:phase23")
        self.assertIs(a["custom_solver"], b["custom_solver"])
        self.assertEqual(a["custom_solver"].label, "pardiso_v07(phase23)")
        base = registry.build_linear_block("pardiso")
        self.assertIsNot(base["custom_solver"], a["custom_solver"])
        self.assertEqual(base["custom_solver"].label, "pardiso_solver(mkl multithreaded direct)")
        # the run-wide default mode fills a bare "pardiso"
        defaulted = registry.build_linear_block("pardiso", pardiso_mode_default="phase23")
        self.assertIs(defaulted["custom_solver"], a["custom_solver"])

    def test_other_backends(self):
        self.assertEqual(registry.build_linear_block("spsolve"), {"spsolve_solver": {}})
        petsc = registry.build_linear_block({"backend": "petsc", "ksp_type": "cg", "gpu": True})
        self.assertEqual(
            petsc["petsc_solver"],
            {"ksp_type": "cg", "pc_type": "jacobi", "mat_type": "aijcusparse", "vec_type": "cuda"},
        )
        amgx = registry.build_linear_block({"backend": "amgx", "cfg_path": "x.json"})
        self.assertEqual(amgx["amgx_solver"], {"persistent_resources": True, "cfg_path": "x.json"})

    def test_fallback_blocks(self):
        self.assertEqual(registry.build_fallback_linear_block(None), {"spsolve_solver": {}})
        self.assertEqual(registry.build_fallback_linear_block("spsolve"), {"spsolve_solver": {}})
        self.assertIsNone(registry.build_fallback_linear_block("none"))
        pardiso = registry.build_fallback_linear_block("pardiso", pardiso_mode_default="phase23")
        self.assertEqual(pardiso["custom_solver"].label, "pardiso_v07(phase23)")

    def test_labels_and_same_backend(self):
        self.assertEqual(
            registry.linear_block_label({"jax_solver": {"method": "cg", "precond": True, "tol": 1e-6}}),
            "jax_solver(method=cg, precond=True, tol=1e-06)",
        )
        self.assertEqual(registry.linear_block_label(None), "preserve original solver_options")
        p = registry.build_linear_block("pardiso:phase23")
        self.assertTrue(registry.same_linear_backend(p, registry.build_fallback_linear_block("pardiso", pardiso_mode_default="phase23")))
        self.assertFalse(registry.same_linear_backend(p, {"spsolve_solver": {}}))
        self.assertTrue(registry.same_linear_backend({"spsolve_solver": {}}, {"spsolve_solver": {}}))


class ScopedDefaultsTest(unittest.TestCase):
    def setUp(self):
        registry.reset_shared_solvers()

    def test_auto_resolves_thermal_cg_jacobi_and_mechanics_pardiso(self):
        args = SimpleNamespace(xla_linear_solver="auto", xla_pardiso_mode="phase23")
        specs = registry.scoped_specs_from_args(args)
        self.assertEqual(specs["thermal"], registry.DEFAULT_SCOPED_SPECS["thermal"])
        self.assertEqual(specs["mechanics"], {"backend": "pardiso", "mode": "phase23"})
        blocks = registry.scoped_linear_options_from_args(args)
        self.assertEqual(
            blocks["thermal"],
            {"jax_solver": {"precond": True, "method": "cg", "tol": 1e-6, "atol": 1e-6,
                            "maxiter": 10000}},
        )
        self.assertEqual(blocks["mechanics"]["custom_solver"].label, "pardiso_v07(phase23)")

    def test_explicit_scope_wins_over_auto(self):
        args = SimpleNamespace(
            xla_linear_solver="auto", xla_pardiso_mode="phase23",
            thermal_linear_solver="jax:bicgstab:none", mechanics_linear_solver="spsolve",
        )
        blocks = registry.scoped_linear_options_from_args(args)
        self.assertEqual(blocks["thermal"], {"jax_solver": {"precond": False, "method": "bicgstab"}})
        self.assertEqual(blocks["mechanics"], {"spsolve_solver": {}})

    def test_legacy_global_choice_leaves_scopes_empty(self):
        args = SimpleNamespace(xla_linear_solver="pardiso", xla_pardiso_mode="phase23")
        self.assertEqual(registry.scoped_linear_options_from_args(args), {"thermal": None, "mechanics": None})

    def test_mechanics_direct_alias_routes_mechanics_to_pardiso(self):
        args = SimpleNamespace(xla_linear_solver="jax", xla_pardiso_mode=None, mechanics_direct_solver=True)
        blocks = registry.scoped_linear_options_from_args(args)
        self.assertIsNone(blocks["thermal"])
        self.assertEqual(blocks["mechanics"]["custom_solver"].label, "pardiso_v07(phase23)")

    def test_argparse_defaults_are_auto(self):
        parser = acceleration.build_arg_parser()
        args = parser.parse_args([])
        self.assertEqual(args.xla_linear_solver, "auto")
        self.assertEqual(args.xla_fallback_solver, "spsolve")
        self.assertIsNone(acceleration.linear_options_from_args(args))
        blocks = registry.scoped_linear_options_from_args(args)
        self.assertIn("jax_solver", blocks["thermal"])
        self.assertIn("custom_solver", blocks["mechanics"])
        args = parser.parse_args(["--mechanics-linear-solver", '{"backend": "pyamg", "maxiter": 50}'])
        self.assertEqual(registry.scoped_specs_from_args(args)["mechanics"]["maxiter"], 50)


class ScopedSolverPatchTest(unittest.TestCase):
    def setUp(self):
        registry.reset_shared_solvers()

    def _install(self, scoped, linear_options=None, fallback_options=None, fallback=True, fail_first=False):
        calls = []

        def fake_solver(problem, solver_options=None):
            calls.append((problem, solver_options))
            if fail_first and len(calls) == 1:
                raise RuntimeError("iterative solve blew up")
            return "ok"

        module = SimpleNamespace(solver=fake_solver)
        acceleration.install_solver_patch(
            module, linear_options, fallback_to_spsolve=fallback,
            scoped_linear_options=scoped, fallback_options=fallback_options,
        )
        return module, calls

    def test_each_physics_gets_its_own_linear_block(self):
        custom = _RecordingCustom()
        thermal_block = {"jax_solver": {"precond": True, "method": "cg"}}
        module, calls = self._install({"thermal": thermal_block, "mechanics": {"custom_solver": custom}})
        thermal = TransientThermal()
        mechanics = ThermoMechanical()
        module.solver(thermal, solver_options={"newton": {"linear": {"spsolve_solver": {}}, "tol": 1e-6}})
        module.solver(mechanics, solver_options={"newton": {"linear": {"spsolve_solver": {}}, "rel_tol": 5e-5}})
        self.assertEqual(calls[0][1]["newton"]["linear"], thermal_block)
        self.assertEqual(calls[0][1]["newton"]["tol"], 1e-6)
        self.assertIs(calls[1][1]["newton"]["linear"]["custom_solver"], custom)
        self.assertEqual(calls[1][1]["newton"]["rel_tol"], 5e-5)
        self.assertEqual(custom.bound, [mechanics])

    def test_scope_without_block_falls_through_to_global(self):
        module, calls = self._install({"thermal": None, "mechanics": None}, linear_options={"spsolve_solver": {}})
        module.solver(TransientThermal(), solver_options={"newton": {"linear": {"jax_solver": {}}}})
        self.assertEqual(calls[0][1]["newton"]["linear"], {"spsolve_solver": {}})
        module, calls = self._install({"thermal": None}, linear_options=None)
        module.solver(TransientThermal(), solver_options={"newton": {"linear": {"jax_solver": {}}}})
        self.assertEqual(calls[0][1]["newton"]["linear"], {"jax_solver": {}})

    def test_release_marker_routes_iterative_mechanics_to_pardiso(self):
        module, calls = self._install({"mechanics": {"jax_solver": {"precond": True}}})
        module.solver(ThermoMechanical(prefer_direct=True), solver_options={"newton": {"linear": {"spsolve_solver": {}}}})
        block = calls[0][1]["newton"]["linear"]
        self.assertIs(block["custom_solver"], registry.shared_pardiso_solver("phase23"))
        # a direct mechanics block is left alone
        custom = _RecordingCustom()
        module, calls = self._install({"mechanics": {"custom_solver": custom}})
        module.solver(ThermoMechanical(prefer_direct=True), solver_options={"newton": {"linear": {"spsolve_solver": {}}}})
        self.assertIs(calls[0][1]["newton"]["linear"]["custom_solver"], custom)

    def test_fallback_uses_configured_block(self):
        fallback = registry.build_fallback_linear_block("pardiso", pardiso_mode_default="phase23")
        module, calls = self._install(
            {"thermal": {"jax_solver": {"precond": True}}}, fallback_options=fallback, fail_first=True,
        )
        self.assertEqual(module.solver(TransientThermal(), solver_options={"newton": {}}), "ok")
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[1][1]["newton"]["linear"]["custom_solver"], fallback["custom_solver"])

    def test_newton_stall_retries_once_under_iterative_block(self):
        fallback = registry.build_fallback_linear_block("pardiso", pardiso_mode_default="phase23")
        calls = []

        def fake_solver(problem, solver_options=None):
            calls.append(solver_options)
            if len(calls) == 1:
                raise RuntimeError("Newton solver did not converge within max_iter=100 iterations")
            return "ok"

        module = SimpleNamespace(solver=fake_solver)
        report = acceleration.ProfilingReport(label="unit")
        acceleration.install_solver_patch(
            module, None, fallback_to_spsolve=True, profiler=report,
            scoped_linear_options={"thermal": {"jax_solver": {"precond": True, "method": "cg"}}},
            fallback_options=fallback,
        )
        self.assertEqual(module.solver(TransientThermal(), solver_options={"newton": {}}), "ok")
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[1]["newton"]["linear"]["custom_solver"], fallback["custom_solver"])
        self.assertEqual(report.meta["solver_fallbacks"], 1)
        self.assertEqual(report.meta["newton_stall_fallbacks"], 1)

    def test_newton_stall_under_direct_block_is_not_retried(self):
        calls = []

        def fake_solver(problem, solver_options=None):
            calls.append(solver_options)
            raise RuntimeError("Newton solver did not converge within max_iter=50 iterations")

        module = SimpleNamespace(solver=fake_solver)
        acceleration.install_solver_patch(
            module, None, fallback_to_spsolve=True,
            scoped_linear_options={"mechanics": registry.build_linear_block("pardiso:phase23")},
            fallback_options={"spsolve_solver": {}},
        )
        with self.assertRaises(RuntimeError):
            module.solver(ThermoMechanical(), solver_options={"newton": {}})
        self.assertEqual(len(calls), 1)

    def test_iterative_block_detection(self):
        self.assertTrue(registry.is_iterative_linear_block({"jax_solver": {}}))
        self.assertTrue(registry.is_iterative_linear_block(registry.build_linear_block("pyamg")))
        self.assertFalse(registry.is_iterative_linear_block(registry.build_linear_block("pardiso:phase23")))
        self.assertFalse(registry.is_iterative_linear_block({"spsolve_solver": {}}))
        self.assertFalse(registry.is_iterative_linear_block(None))

    def test_no_retry_when_active_block_is_the_fallback_backend(self):
        fallback = registry.build_fallback_linear_block("pardiso", pardiso_mode_default="phase23")
        module, calls = self._install(
            {"mechanics": registry.build_linear_block("pardiso:phase23")}, fallback_options=fallback, fail_first=True,
        )
        with self.assertRaises(RuntimeError):
            module.solver(ThermoMechanical(), solver_options={"newton": {}})
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()


import pytest as _pytest_lane
pytestmark = _pytest_lane.mark.solver
