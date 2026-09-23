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
