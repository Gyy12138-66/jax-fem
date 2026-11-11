"""Physical integration contract for modified Newton on B-bar/J2 mechanics.

The comparison is deliberately numerical rather than a wall-time assertion:
the optimized arm must reach the same equilibrium and material state while
turning at least one repeated tangent solve into a PARDISO phase-33 backsolve.
"""

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    import numpy as onp
    import jax.numpy as jnp
    import pypardiso  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on local runtime
    onp = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests.unit.test_v03_bbar_hex8 import (  # noqa: E402
    BBarTestBase,
    NEWTON,
    beam_grid,
)

from jax_fem.generate_mesh import Mesh  # noqa: E402
from jax_fem.solver import apply_bc_vec  # noqa: E402
from jax_fem_am.simulation.acceleration import (  # noqa: E402
    _PardisoCustomSolver,
)


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"JAX/PARDISO runtime unavailable: {IMPORT_ERROR}",
)
class ModifiedNewtonJ2IntegrationTest(BBarTestBase):
    """32-HEX8 thermal-gradient cantilever with active J2 plastic flow."""

    NX, NY, NZ = 8, 2, 2
    LX, LY, LZ = 0.8e-3, 0.2e-3, 0.2e-3

    def build_case(self):
        points, hexes = beam_grid(
            self.NX,
            self.NY,
            self.NZ,
            self.LX,
            self.LY,
            self.LZ,
        )
        centroids = onp.asarray(points)[hexes].mean(axis=1)

        def clamp_x0(point):
            return point[0] < 1e-9

        def zero(_point):
            return 0.0

        problem = self.v03.ThermoMechanical(
            mesh=Mesh(points, hexes, ele_type="HEX8"),
            vec=3,
            dim=3,
            ele_type="HEX8",
            quadrature_order=2,
            dirichlet_bc_info=[
                [clamp_x0] * 3,
                [0, 1, 2],
                [zero] * 3,
            ],
            additional_info=("j2_plastic", None, 0.0, 0.0, (), True),
        )
        params = self.uniform_params(
            problem,
            dT=0.0,
            yield_stress=60e6,
        )
        dT_cell = -400.0 * centroids[:, 2] / self.LZ
        params[1] = jnp.asarray(
            onp.broadcast_to(
                dT_cell[:, None, None],
                (len(hexes), problem.fes[0].num_quads, 1),
            ).copy()
        )
        return problem, params, jnp.zeros((len(points), 3))

    @staticmethod
    def true_residual(problem, displacement, params):
        """Residual-only equilibrium vector with production row BCs."""
        problem.set_params(params)
        residual = problem.compute_residual([displacement])[0].reshape(-1)
        dofs = displacement.reshape(-1)
        return onp.asarray(apply_bc_vec(residual, dofs, problem))

    def solve_arm(self, jacobian_reuse=None):
        problem, params, u0 = self.build_case()
        initial_residual = self.true_residual(problem, u0, params)
        pardiso = _PardisoCustomSolver("phase23")
        newton = dict(NEWTON)
        newton["linear"] = {"custom_solver": pardiso}
        if jacobian_reuse is not None:
            newton["jacobian_reuse"] = dict(jacobian_reuse)

        displacement = self.v03.run_mechanics(
            problem,
            [u0],
            params,
            newton,
        )[0]
        residual = self.true_residual(problem, displacement, params)
        stress = problem.compute_cell_stress(displacement, params)
        eqp = problem.compute_eqp_update(displacement, params)
        stats = pardiso._v07_variant.stats_snapshot()

        return {
            "u": onp.asarray(displacement),
            "initial_residual": initial_residual,
            "residual": residual,
            "stress": onp.asarray(stress["stress_quad"]),
            "vm": onp.asarray(stress["vm_quad"]),
            "eqp": onp.asarray(eqp),
            "stats": stats,
            "newton": newton,
        }

    def assert_converged(self, result):
        initial_norm = float(onp.linalg.norm(result["initial_residual"]))
        final_norm = float(onp.linalg.norm(result["residual"]))
        limit = max(
            float(result["newton"]["tol"]),
            float(result["newton"]["rel_tol"]) * initial_norm,
        )
        self.assertLessEqual(
            final_norm,
            1.01 * limit,
            f"true residual {final_norm:.6e} exceeds Newton limit "
            f"{limit:.6e}",
        )
        return limit

    def assert_relative_l2(self, name, actual, reference, tolerance):
        numerator = float(onp.linalg.norm(actual - reference))
        denominator = max(
            float(onp.linalg.norm(reference)),
            onp.finfo(onp.float64).tiny,
        )
        relative = numerator / denominator
        self.assertLessEqual(
            relative,
            tolerance,
            f"{name} relative L2 difference {relative:.6e} exceeds "
            f"{tolerance:.1e}",
        )

    def test_modified_newton_reuses_factorization_without_changing_j2_state(self):
        full = self.solve_arm()
        modified = self.solve_arm(
            {"max_reuse": 2, "refresh_residual_ratio": 0.9}
        )

        full_limit = self.assert_converged(full)
        modified_limit = self.assert_converged(modified)
        residual_difference = float(
            onp.linalg.norm(modified["residual"] - full["residual"])
        )
        self.assertLessEqual(
            residual_difference,
            full_limit + modified_limit,
            "both arms must represent the same equilibrium root within their "
            "declared nonlinear tolerances",
        )

        # These tolerances are tighter than the nonlinear solve tolerance in
        # field space, while allowing the two valid Newton paths to stop on
        # opposite sides of the same residual threshold.
        self.assert_relative_l2("displacement", modified["u"], full["u"], 5e-4)
        self.assert_relative_l2("stress", modified["stress"], full["stress"], 1e-3)
        self.assert_relative_l2("von Mises", modified["vm"], full["vm"], 1e-3)
        self.assert_relative_l2("eqp", modified["eqp"], full["eqp"], 1e-3)
        self.assertGreater(float(full["eqp"].max()), 1e-4)
        self.assertGreater(float(modified["eqp"].max()), 1e-4)

        self.assertGreaterEqual(modified["stats"]["phase33_calls"], 1)
        self.assertGreaterEqual(modified["stats"]["backsolve_hits"], 1)
        self.assertLess(
            modified["stats"]["numeric_factorizations"],
            full["stats"]["numeric_factorizations"],
        )


if __name__ == "__main__":
    unittest.main()
