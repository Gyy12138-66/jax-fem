"""Release rigid-body anchors must be chosen from material that survives the cut
and must remove all six rigid-body modes."""
import unittest

import numpy as np

from jax_fem_am.physics import release


def _grid(nx, ny, nz, z0=0.0, dz=1.0):
    xs, ys, zs = np.arange(nx + 1.0), np.arange(ny + 1.0), z0 + dz * np.arange(nz + 1.0)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


class RigidBodyAnchorTest(unittest.TestCase):
    def setUp(self):
        # raft: wide plate z in [0, 1]; part: narrower block z in [1, 5] on top
        self.points = np.vstack([_grid(8, 8, 1), _grid(4, 4, 4, z0=1.0) + np.array([2.0, 2.0, 0.0])])
        n_raft = (8 + 1) * (8 + 1) * 2
        self.raft_nodes = np.arange(n_raft)
        self.part_nodes = np.arange(n_raft, len(self.points))

    def test_anchors_from_all_printed_nodes_land_on_the_raft_and_fail_validation(self):
        ids = release.select_rigid_body_anchor_nodes(self.points, np.arange(len(self.points)))
        self.assertTrue(any(i in set(self.raft_nodes.tolist()) for i in ids))
        with self.assertRaisesRegex(ValueError, "retained after the cut"):
            release.validate_rigid_body_anchor_rank(self.points, ids, self.part_nodes)

    def test_anchors_from_retained_part_nodes_have_rank_six(self):
        ids = release.select_rigid_body_anchor_nodes(self.points, self.part_nodes)
        self.assertTrue(all(i in set(self.part_nodes.tolist()) for i in ids))
        self.assertEqual(release.validate_rigid_body_anchor_rank(self.points, ids, self.part_nodes), 6)
        bc = release.make_anchor_mechanics_bc(self.points, candidate_node_ids=self.part_nodes)
        self.assertEqual(len(bc[0]), 6)
        self.assertEqual(bc[1], [0, 1, 2, 1, 2, 2])
        # the location functions hit exactly the selected nodes
        # (the synthetic raft/part grids share the z=1 plane, so coordinate
        # matching can also hit the duplicate raft node -- membership suffices)
        hits = [[int(j) for j in range(len(self.points)) if fn(self.points[j])] for fn in bc[0]]
        for hit, expected in zip(hits, [ids[0], ids[0], ids[0], ids[1], ids[1], ids[2]]):
            self.assertIn(expected, hit)
            np.testing.assert_allclose(self.points[hit], np.broadcast_to(self.points[expected], (len(hit), 3)))

    def test_degenerate_geometry_is_rejected(self):
        # three collinear anchors along x with the 3-2-1 components: rotation about x survives
        pts = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "rigid-body modes"):
            release.validate_rigid_body_anchor_rank(pts, [0, 3, 1], np.arange(4))

    def test_dof_pairs_follow_the_3_2_1_scheme(self):
        pairs = release.rigid_body_anchor_dof_pairs([7, 11, 13])
        self.assertEqual(pairs.tolist(), [[7, 0], [7, 1], [7, 2], [11, 1], [11, 2], [13, 2]])


if __name__ == "__main__":
    unittest.main()


class RobustAnchorSelectionTest(unittest.TestCase):
    """2026-09-24: automatic anchors always reach rank 6; geometries that passed
    before keep exactly the same anchors; manual points; rigid-motion removal."""

    def _cube(self):
        # Balbaa-cube-like retained part: 10 mm cube offset like the X1 mesh, uniform grid
        g = _grid(10, 10, 10) * 1e-3
        return g + np.array([0.005, 0.005, 0.006])

    def test_cube_legacy_rule_is_degenerate_and_new_selection_has_rank_six(self):
        pts = self._cube()
        ids = np.arange(len(pts))
        legacy = release._legacy_anchor_triple(pts, ids)
        self.assertLess(release._anchor_triple_rank(pts, legacy), 6)
        new = release.select_rigid_body_anchor_nodes(pts, ids)
        self.assertEqual(release.validate_rigid_body_anchor_rank(pts, new, ids), 6)

    def test_geometries_that_passed_keep_the_same_anchors(self):
        points = np.vstack([_grid(8, 8, 1), _grid(4, 4, 4, z0=1.0) + np.array([2.0, 2.0, 0.0])])
        part = np.arange(162, len(points))
        rng = np.random.default_rng(0)
        cloud = rng.random((500, 3)) * np.array([0.09, 0.1, 0.09])
        for pts, ids in ((points, part), (cloud, np.arange(len(cloud)))):
            legacy = release._legacy_anchor_triple(pts, ids)
            self.assertEqual(release._anchor_triple_rank(pts, legacy), 6)
            self.assertEqual(release.select_rigid_body_anchor_nodes(pts, ids), legacy)

    def test_manual_anchor_points_pick_the_nearest_retained_nodes(self):
        pts = self._cube()
        ids = np.arange(len(pts))
        targets = [0.005, 0.005, 0.006, 0.015, 0.005, 0.006, 0.005, 0.015, 0.006]
        got = release.select_rigid_body_anchor_nodes(pts, ids, targets)
        np.testing.assert_allclose(pts[got], np.reshape(targets, (3, 3)), atol=1e-12)
        self.assertEqual(release.validate_rigid_body_anchor_rank(pts, got, ids), 6)
        bc = release.make_anchor_mechanics_bc(pts, candidate_node_ids=ids, anchor_points=targets)
        self.assertTrue(bc[0][0](pts[got[0]]) and bc[0][3](pts[got[1]]) and bc[0][5](pts[got[2]]))

    def test_no_extent_in_xy_is_rejected(self):
        pts = np.array([[0.0, 0.0, z] for z in range(5)])
        with self.assertRaisesRegex(ValueError, "no extent"):
            release._robust_anchor_triple(pts, np.arange(5))

    def test_rigid_body_motion_removal_is_gauge_independent(self):
        pts = self._cube()
        ids = np.arange(len(pts))
        rng = np.random.default_rng(1)
        u_def = 1e-5 * rng.standard_normal(pts.shape)
        # make u_def orthogonal to rigid motion so the split is unique
        u_def, _, _ = release.remove_rigid_body_motion(pts, u_def, ids)
        c = pts.mean(axis=0)
        for t, w in (([1e-4, -2e-4, 3e-5], [1e-3, -2e-3, 5e-4]), ([0, 0, 0], [0, 0, 0])):
            u = u_def + np.asarray(t) + np.cross(np.asarray(w), pts - c)
            got, tt, ww = release.remove_rigid_body_motion(pts, u, ids)
            np.testing.assert_allclose(got, u_def, atol=1e-15)
            np.testing.assert_allclose(tt, t, atol=1e-15)
            np.testing.assert_allclose(ww, w, atol=1e-12)
