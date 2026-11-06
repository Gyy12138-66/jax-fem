"""thermal_ledger_device (plan C): the device gather produces the NumPy gather's row, and falls back to it (same
error) on every input the NumPy gather rejects."""
import itertools
import os
import unittest
from types import SimpleNamespace

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jax_fem_am.verification import thermal_ledger as tl  # noqa: E402
from jax_fem_am.verification import thermal_ledger_device as tld  # noqa: E402

NC, NQ, NN, NNODE, NF, FQ = 40, 4, 4, 60, 15, 3


def make_problem(seed=0, *, latent=False, source_model="legacy", front=False, env=True, surfaces=2, surf_amb=True,
                 cutoff=0.0, renorm=False, as_jax=False):
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, NNODE, size=(NC, NN))
    shape_vals = rng.random((NQ, NN))
    shape_vals /= shape_vals.sum(1, keepdims=True)
    support = np.zeros((NC, NQ, 1))
    support[:30] = 1.0
    active = support * (rng.random((NC, NQ, 1)) < 0.6)
    active[:30] = np.repeat(active[:30, :1], NQ, axis=1)
    cooling = support * (1.0 - active) * (rng.random((NC, 1, 1)) < 0.5)
    full = lambda v: np.full((NC, NQ, 1), float(v))  # noqa: E731
    internal = [
        300.0 + 900.0 * rng.random((NC, NQ, 1)),           # T_old
        full(2.5e-3),                                      # dt
        np.broadcast_to(np.array([4e-4, 5e-4, 1.2e-3]), (NC, NQ, 3)).copy(),
        full(180.0),                                       # effective power
        full(6e-5),                                        # beam radius
        full(4e-5),                                        # source depth
        full(1.0),                                         # switch
        active,
        8100.0 * support,                                  # rho
        (430.0 + 200.0 * rng.random((NC, NQ, 1))) * support,
        (11.0 + 15.0 * rng.random((NC, NQ, 1))) * support,  # conductivity
        (0.0 if latent else 35.0) * rng.random((NC, NQ, 1)) * support,
        cooling,
        full(12.0),                                        # old-layer h
    ]
    if env:
        internal.append(full(293.0))
    vals_list = [np.full(8, 300.0), np.asarray([310.0, 320.0])]

    class FE:
        JxW = 1e-12 * (0.5 + rng.random((NC, NQ)))
        node_inds_list = [np.arange(8), np.asarray([20, 21])]
        vec_inds_list = [np.zeros(8, int), np.zeros(2, int)]

        @staticmethod
        def convert_from_dof_to_quad(values):
            v = np.asarray(values)
            return np.sum(v[cells][:, None, :, :] * shape_vals[None, :, :, None], axis=2)
    FE.cells = cells
    FE.vals_list = vals_list
    binds = [np.stack([rng.integers(0, NC, NF), rng.integers(0, 6, NF)], 1) for _ in range(surfaces)]
    sv = [rng.random((NF, FQ, NN)) / NN for _ in range(surfaces)]
    ns = [1e-8 * rng.random((NF, 1, FQ)) for _ in range(surfaces)]
    surf_vars = []
    for _ in range(surfaces):
        sa = (rng.random((NF, FQ, 1)) < 0.7).astype(float)
        surf_vars.append([sa, np.full((NF, FQ, 1), 291.0)] if surf_amb else [sa])
    res = 1e-3 * np.sin(np.arange(NNODE, dtype=float))[:, None]
    conv = (lambda x: jnp.asarray(x)) if as_jax else (lambda x: x)  # noqa: E731
    p = SimpleNamespace(
        fes=[FE()], physical_quad_points=1e-3 * rng.random((NC, NQ, 3)), internal_vars=[conv(x) for x in internal],
        boundary_inds_list=binds, selected_face_shape_vals=sv, nanson_scale=ns, internal_vars_surfaces=surf_vars,
        convection_h=10.0, ambient=295.0, emissivity=0.4, stefan_boltzmann=5.670374419e-8, build_axis_id=2,
        plane_axis0_id=0, plane_axis1_id=1, build_sign=1.0, front_surface_loss_h=40.0 if front else 0.0,
        front_surface_loss_thickness=5e-5, front_surface_loss_radiation=front, source_model=source_model,
        solidus_temperature=1530.0, liquidus_temperature=1610.0, latent_heat=2.7e5 if latent else 0.0,
        source_depth_cutoff=cutoff, source_cutoff_renormalize=renorm,
        compute_residual=lambda sol: [conv(res)])
    T_new = 300.0 + 1100.0 * rng.random((NNODE, 1))
    prev = 300.0 + 800.0 * rng.random((NNODE, 1))
    return p, [conv(T_new)], conv(prev)


KW = dict(step_index=7, step_state=SimpleNamespace(global_step=7, mode="scan", layer_idx=1, hatch_idx=0, scan_idx=3,
                                                   laser_power=300.0, laser_switch=1.0),
          absorptivity=0.6, temperature_atol_k=1e-3, solver_residual_tolerance_w=1e-6)


class DeviceLedgerMatchesNumpy(unittest.TestCase):
    def setUp(self):
        os.environ["JAX_FEM_LEDGER_SHADOW"] = "0"

    def _both(self, prev=True, **cfg):
        p, sol, pv = make_problem(**cfg)
        kw = dict(KW, previous_solution=pv if prev else None)
        ref = tl.extract_solver_step(p, sol, **kw)
        dev, reason = tld._device_row(p, sol, relative_balance_tolerance=1e-6, **kw)
        return ref, dev, reason

    def test_every_branch(self):
        combos = itertools.product((False, True), ("legacy", "paper_hemispherical"), (False, True), (False, True),
                                   (0, 2), (False, True), (False, True))
        n = 0
        for latent, model, front, env, surfaces, surf_amb, prev in combos:
            for cutoff, renorm in ((0.0, False), (6e-5, False), (6e-5, True)):
                if model != "legacy" and cutoff > 0.0:
                    continue
                ref, dev, reason = self._both(prev=prev, latent=latent, source_model=model, front=front, env=env,
                                              surfaces=surfaces, surf_amb=surf_amb, cutoff=cutoff, renorm=renorm)
                self.assertIsNone(reason, (latent, model, front, env, surfaces, surf_amb, prev, cutoff, renorm))
                worst, mism = tld.compare_rows(dev, ref, record=False)
                self.assertEqual(mism, [], (latent, model, front, env, surfaces, prev, cutoff))
                self.assertLess(worst, 1e-12)
                self.assertEqual(set(dev), set(ref))
                n += 1
        self.assertGreater(n, 100)

    def test_device_arrays_take_the_same_path(self):
        ref, _, _ = self._both(latent=True, front=True)
        p, sol, pv = make_problem(latent=True, front=True, as_jax=True)
        dev, reason = tld._device_row(p, sol, previous_solution=pv, relative_balance_tolerance=1e-6, **KW)
        self.assertIsNone(reason)
        worst, mism = tld.compare_rows(dev, ref, record=False)
        self.assertEqual(mism, [])
        self.assertLess(worst, 1e-12)

    def test_terms_are_nontrivial(self):
        ref, dev, _ = self._both(latent=True, front=True)
        for k in ("storage_j", "laser_deposited_j", "front_loss_j", "old_layer_loss_j", "surface_loss_j",
                  "dirichlet_exchange_into_domain_j", "pre_solve_state_override_j"):
            self.assertNotEqual(ref[k], 0.0, k)


class DeviceLedgerFallsBack(unittest.TestCase):
    def setUp(self):
        os.environ["JAX_FEM_LEDGER_SHADOW"] = "0"

    def _same_error(self, mutate):
        p, sol, pv = make_problem()
        mutate(p, sol)
        kw = dict(KW, previous_solution=pv)
        with self.assertRaises(ValueError) as ref:
            tl.extract_solver_step(p, sol, **kw)
        before = tld.stats()["fallback_rows"]
        with self.assertRaises(ValueError) as dev:
            tld.extract_solver_step_device(p, sol, **kw)
        self.assertEqual(str(dev.exception), str(ref.exception))
        self.assertEqual(tld.stats()["fallback_rows"], before + 1)

    def test_nan_heat_capacity(self):
        def m(p, sol):
            p.internal_vars[9] = p.internal_vars[9].copy()
            p.internal_vars[9][0, 0, 0] = np.nan
        self._same_error(m)

    def test_nonuniform_dt(self):
        def m(p, sol):
            p.internal_vars[1] = p.internal_vars[1].copy()
            p.internal_vars[1][3, 1, 0] *= 1.001
        self._same_error(m)

    def test_negative_cp_and_bad_absorptivity_and_nonbinary_surface(self):
        def neg(p, sol):
            p.internal_vars[9] = p.internal_vars[9].copy()
            p.internal_vars[9][1, 0, 0] = -1.0
        self._same_error(neg)

        def surf(p, sol):
            p.internal_vars_surfaces[0][0] = p.internal_vars_surfaces[0][0].copy()
            p.internal_vars_surfaces[0][0][0, 0, 0] = 0.5
        self._same_error(surf)
        p, sol, pv = make_problem()
        kw = dict(KW, previous_solution=pv, absorptivity=1.5)
        with self.assertRaises(ValueError) as ref:
            tl.extract_solver_step(p, sol, **kw)
        with self.assertRaises(ValueError) as dev:
            tld.extract_solver_step_device(p, sol, **kw)
        self.assertEqual(str(dev.exception), str(ref.exception))

    def test_shadow_counts(self):
        os.environ["JAX_FEM_LEDGER_SHADOW"] = "1"
        p, sol, pv = make_problem(latent=True)
        s0 = dict(tld.stats())
        row = tld.extract_solver_step_device(p, sol, **dict(KW, previous_solution=pv))
        self.assertEqual(tld.stats()["shadow_rows"], s0["shadow_rows"] + 1)
        self.assertEqual(tld.stats()["shadow_mismatch"], s0["shadow_mismatch"])
        self.assertEqual(row["step_index"], KW["step_index"])


if __name__ == "__main__":
    unittest.main()
