"""Half-model support (branch halfmodel-0911): symmetry-plane build/release restraints,
face-level exclusion of the cut plane from surface exchange, and the preflight
time/energy rule for a mesh that represents a fraction of the real part."""
import importlib.util
import json
import math
import pathlib
import tempfile
import unittest

import jax
import numpy as np

from jax_fem.fe import FiniteElement
from jax_fem.generate_mesh import Mesh
from jax_fem_am.physics import release

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parents[2]


def _grid(nx, ny, nz, z0=0.0, dz=1.0):
    xs, ys, zs = np.arange(nx + 1.0), np.arange(ny + 1.0), z0 + dz * np.arange(nz + 1.0)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _hits(fn, points):
    """Node ids selected by a 1- or 2-argument Dirichlet location function."""
    n_args = fn.__code__.co_argcount
    return [i for i in range(len(points)) if (fn(points[i], i) if n_args == 2 else fn(points[i]))]


class SymmetryPlaneBuildBCTest(unittest.TestCase):
    def setUp(self):
        # half block x in [0, 4], y in [0, 3], z in [0, 5]; the cut plane is x = 0
        self.points = _grid(4, 3, 5)

    def test_plane_selection_hits_exactly_the_x_min_face(self):
        mask, value, tol = release.select_symmetry_plane_nodes(self.points, 0, "min")
        self.assertEqual(value, 0.0)
        np.testing.assert_array_equal(mask, self.points[:, 0] == 0.0)
        self.assertEqual(int(mask.sum()), 4 * 6)
        mask_max, value_max, _ = release.select_symmetry_plane_nodes(self.points, 0, "max")
        self.assertEqual(value_max, 4.0)
        self.assertEqual(int(mask_max.sum()), 4 * 6)

    def test_plane_only_bc_fixes_the_normal_component_on_plane_nodes_only(self):
        bc, meta = release.make_symmetry_plane_only_bc(self.points, plane_axis_id=0, side="min")
        self.assertEqual([len(bc[0]), len(bc[1]), len(bc[2])], [1, 1, 1])
        self.assertEqual(bc[1], [0])
        self.assertEqual(bc[2][0](self.points[0]), 0.0)
        hits = _hits(bc[0][0], self.points)
        self.assertEqual(hits, [int(i) for i in np.flatnonzero(self.points[:, 0] == 0.0)])
        self.assertEqual(meta["plane_nodes"], 24)
        self.assertEqual(meta["plane_axis"], "x")

    def test_fixed_symmetry_is_the_base_clamp_plus_the_plane_condition(self):
        def bottom(point):
            return point[2] <= 1e-9

        base_bc = release.make_full_bottom_mechanics_bc(bottom)
        plane_bc, _ = release.make_symmetry_plane_only_bc(self.points, plane_axis_id=0, side="min")
        combined = [base_bc[0] + plane_bc[0], base_bc[1] + plane_bc[1], base_bc[2] + plane_bc[2]]
        self.assertEqual(combined[1], [0, 1, 2, 0])
        self.assertEqual(len(combined[0]), 4)
        base_nodes = set(_hits(combined[0][0], self.points))
        plane_nodes = set(_hits(combined[0][3], self.points))
        self.assertEqual(base_nodes, set(np.flatnonzero(self.points[:, 2] == 0.0).tolist()))
        self.assertEqual(plane_nodes, set(np.flatnonzero(self.points[:, 0] == 0.0).tolist()))
        # every constrained value is zero
        self.assertTrue(all(fn(self.points[0]) == 0.0 for fn in combined[2]))


class SymmetryPlaneReleaseTest(unittest.TestCase):
    def setUp(self):
        # raft z in [0, 1] (wide), part z in [1, 6] on top; cut plane x = 0 for both
        raft = _grid(4, 4, 1)
        part = _grid(4, 2, 5, z0=1.0)
        self.points = np.vstack([raft, part])
        n_raft = len(raft)
        self.raft_nodes = np.arange(n_raft)
        self.part_nodes = np.arange(n_raft, len(self.points))
        self.plane_ids = np.flatnonzero(self.points[:, 0] == 0.0)

    def test_anchors_restricted_to_retained_nodes_have_rank_six(self):
        bc, meta = release.make_symmetry_plane_mechanics_bc(
            self.points, plane_axis_id=0, side="min", return_metadata=True,
            anchor_candidate_node_ids=self.part_nodes)
        self.assertEqual(bc[1], [0, 1, 2, 2])
        self.assertIn(meta["anchor_node_id"], set(self.part_nodes.tolist()))
        self.assertIn(meta["far_node_id"], set(self.part_nodes.tolist()))
        self.assertEqual(self.points[meta["anchor_node_id"], 0], 0.0)
        self.assertEqual(self.points[meta["far_node_id"], 0], 0.0)
        self.assertEqual(len(meta["anchor_dof_pairs"]), 3)
        rank = release.validate_symmetry_plane_release_rank(
            self.points, self.plane_ids, 0, meta["anchor_dof_pairs"], self.part_nodes)
        self.assertEqual(rank, 6)
        # the plane condition still covers every plane node, raft included
        self.assertEqual(set(_hits(bc[0][0], self.points)), set(self.plane_ids.tolist()))

    def test_unrestricted_anchors_land_on_the_raft_and_fail_validation(self):
        _, meta = release.make_symmetry_plane_mechanics_bc(
            self.points, plane_axis_id=0, side="min", return_metadata=True)
        self.assertIn(meta["anchor_node_id"], set(self.raft_nodes.tolist()))
        with self.assertRaisesRegex(ValueError, "retained after the cut"):
            release.validate_symmetry_plane_release_rank(
                self.points, self.plane_ids, 0, meta["anchor_dof_pairs"], self.part_nodes)

    def test_too_few_candidates_on_the_plane_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two candidate nodes"):
            release.make_symmetry_plane_mechanics_bc(
                self.points, plane_axis_id=0, side="min",
                anchor_candidate_node_ids=[int(self.plane_ids[0])])

    def test_degenerate_plane_is_rejected(self):
        # all plane nodes on one line (y varies, z constant): rotation about y survives
        pts = np.array([[0.0, y, 0.0] for y in range(5)] + [[1.0, y, 0.0] for y in range(5)])
        plane_ids = np.arange(5)
        pairs = [[0, 1], [0, 2], [4, 2]]
        with self.assertRaisesRegex(ValueError, "rigid-body modes"):
            release.validate_symmetry_plane_release_rank(pts, plane_ids, 0, pairs, np.arange(10))


class ExcludePlaneFaceSelectionTest(unittest.TestCase):
    def _two_stacked_hex(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                           [0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2]], dtype=np.float64)
        cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.int32)
        return points, cells

    def _face_x_coords(self, fe, inds):
        cell_points = np.asarray(fe.points)[np.asarray(fe.cells)]
        faces = cell_points[:, np.asarray(fe.face_inds)]  # (cells, faces, verts, 3)
        return [faces[int(c), int(f), :, 0] for c, f in np.asarray(inds)]

    def test_faces_lying_in_the_plane_are_dropped_but_touching_faces_stay(self):
        points, cells = self._two_stacked_hex()
        fe = FiniteElement(mesh=Mesh(points, cells, ele_type="HEX8"), vec=1, dim=3,
                           ele_type="HEX8", quadrature_order=2, dirichlet_bc_info=None)

        def everything(point):
            return point[0] > -1.0

        everything.exterior_only = True
        all_ext = fe.get_boundary_conditions_inds([everything])[0]
        self.assertEqual(len(all_ext), 10)

        def no_plane(point):
            return point[0] > -1.0

        no_plane.exterior_only = True
        no_plane.exclude_plane = (0, 0.0, 1e-8)
        kept = fe.get_boundary_conditions_inds([no_plane])[0]
        self.assertEqual(len(kept), 8)
        for xs in self._face_x_coords(fe, kept):
            self.assertFalse(np.allclose(xs, 0.0))          # no face lying in x = 0 survives
        touching = [xs for xs in self._face_x_coords(fe, kept) if np.any(np.isclose(xs, 0.0))]
        self.assertEqual(len(touching), 6)                  # top, bottom and the four y-faces touch x = 0


class PreflightRepresentedFractionTest(unittest.TestCase):
    CONFIG = REPO / "cases" / "159_simulation" / "inputs" / "0119-flash-voxel-fast-pyamg.json"

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "make_159_preflight", REPO / "cases" / "159_simulation" / "model" / "make_159_preflight.py")
        cls.pre = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.pre)

    def _tiny_mesh(self):
        dz = 0.0005
        base = 0.001
        # two 0.5 mm voxels stacked: slab 0 and slab 1
        P = np.array([[x, y, z] for z in (base, base + dz, base + 2 * dz) for y in (0.0, dz) for x in (0.0, dz)])
        conn = np.array([[0, 1, 3, 2, 4, 5, 7, 6], [4, 5, 7, 6, 8, 9, 11, 10]])
        vol = np.full(2, dz ** 3)
        return {"P": P, "conn_part": conn, "vol": vol, "dz": dz, "base_z": base, "slab": np.array([0, 1]),
                "n_slabs": 2, "area_m2": np.full(2, dz ** 2), "cells_per_slab": np.array([1, 1]),
                "part_xy_min": np.zeros(2), "part_xy_max": np.full(2, dz), "part_top_z": base + 2 * dz,
                "raft_box": [0.0, dz, 0.0, dz, 0.0, base], "z_levels": [0.0, base, base + dz, base + 2 * dz],
                "part_cells": 2, "raft_cells": 2, "part_volume_m3": float(vol.sum()), "nodes": 12}

    def test_fraction_scales_time_but_not_energy(self):
        cfg_full = self.pre.load_config(self.CONFIG)
        rows_full, led_full = self.pre.generate_schedule(cfg_full, self._tiny_mesh())
        cfg_half = self.pre.load_config(self.CONFIG)
        cfg_half["mesh"]["_represented_fraction"] = 0.5
        rows_half, led_half = self.pre.generate_schedule(cfg_half, self._tiny_mesh())
        self.assertEqual(len(rows_full), len(rows_half))
        self.assertAlmostEqual(led_half["nominal_laser_energy_J"], led_full["nominal_laser_energy_J"], places=12)
        self.assertAlmostEqual(led_half["hold_time_s"], 2.0 * led_full["hold_time_s"] + led_full["scan_time_s"], places=9)
        self.assertAlmostEqual(led_half["recoat_time_s"], led_full["recoat_time_s"], places=9)
        for lf, lh in zip(led_full["layers"], led_half["layers"]):
            self.assertAlmostEqual(lh["scan_time_s"], 2.0 * lf["scan_time_s"], places=12)
            self.assertAlmostEqual(lh["nominal_laser_energy_J"], lf["nominal_laser_energy_J"], places=12)
            self.assertAlmostEqual(lh["commanded_power_W"], lf["commanded_power_W"], places=3)
        self.assertEqual(led_full["symmetry_represented_fraction"], 1.0)
        self.assertEqual(led_half["symmetry_represented_fraction"], 0.5)

    def test_fraction_without_plane_boundary_conditions_is_rejected(self):
        cfg = json.loads(self.CONFIG.read_text(encoding="utf-8"))
        cfg["mesh"]["symmetry"] = {"represented_fraction": 0.5}
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "cfg.json"
            p.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaises(SystemExit):
                self.pre.load_config(p)
        cfg["runner"]["mechanics"]["bottom_bc"] = "fixed_symmetry"
        cfg["thermal_mechanical"]["release_anchor_mode"] = "symmetry_plane"
        cfg["runner"]["thermal_symmetry_plane"] = True
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "cfg.json"
            p.write_text(json.dumps(cfg), encoding="utf-8")
            loaded = self.pre.load_config(p)
        self.assertEqual(loaded["mesh"]["_represented_fraction"], 0.5)

    def test_contract_argv_carries_the_half_model_flags(self):
        cfg = self.pre.load_config(self.CONFIG)
        cfg["mesh"]["_represented_fraction"] = 0.5
        cfg["runner"]["mechanics"]["bottom_bc"] = "fixed_symmetry"
        cfg["thermal_mechanical"]["release_anchor_mode"] = "symmetry_plane"
        cfg["runner"]["thermal_symmetry_plane"] = True
        rows, ledger = self.pre.generate_schedule(cfg, self._tiny_mesh())
        contract = self.pre.runner_contract(cfg, mesh_path=pathlib.Path("m.inp"), path=pathlib.Path("p.csv"),
                                            material=pathlib.Path("mat.json"), ledger=ledger)
        argv = contract["argv"]
        self.assertIn("fixed_symmetry", argv[argv.index("--bottom-mechanics-bc") + 1])
        self.assertEqual(argv[argv.index("--release-anchor-mode") + 1], "symmetry_plane")
        self.assertIn("--thermal-symmetry-plane", argv)
        self.assertEqual(argv[argv.index("--symmetry-plane-axis") + 1], "x")
        self.assertEqual(contract["symmetry"]["represented_fraction"], 0.5)
        # the default (no symmetry keys) contract is unchanged
        cfg0 = self.pre.load_config(self.CONFIG)
        rows0, ledger0 = self.pre.generate_schedule(cfg0, self._tiny_mesh())
        argv0 = self.pre.runner_contract(cfg0, mesh_path=pathlib.Path("m.inp"), path=pathlib.Path("p.csv"),
                                         material=pathlib.Path("mat.json"), ledger=ledger0)["argv"]
        self.assertEqual(argv0[argv0.index("--bottom-mechanics-bc") + 1], "fixed")
        self.assertEqual(argv0[argv0.index("--release-anchor-mode") + 1], "rigid_body")
        self.assertNotIn("--thermal-symmetry-plane", argv0)
        self.assertNotIn("--symmetry-plane-axis", argv0)


if __name__ == "__main__":
    unittest.main()
