"""Floating-island mechanical birth (--mechanics-island-deferral node).

The 0911 half model voxel mesh has cells whose load-carrying neighbourhood at
their layer holds no fixture: their rigid-body modes see no stiffness, the AMG
PCG stalls or returns NaN, and a PCG that "converges" in relative residual can
still carry a large rigid drift of the island (E10 pyamg, step 15,160).
"""

import unittest

import numpy as np

from jax_fem_am.config.schema import build_parser
from jax_fem_am.process.activation import (
    anchored_cell_mask,
    update_island_deferral,
)


def _column(num_cells):
    """Vertical stack of unit HEX8 cells; cell k spans z in [k, k+1]."""
    points = []
    for k in range(num_cells + 1):
        points += [[0, 0, k], [1, 0, k], [1, 1, k], [0, 1, k]]
    cells = [
        [4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3,
         4 * k + 4, 4 * k + 5, 4 * k + 6, 4 * k + 7]
        for k in range(num_cells)
    ]
    return np.asarray(points, dtype=float), np.asarray(cells, dtype=np.int64)


class AnchoredCellMaskTest(unittest.TestCase):
    def test_face_connected_column_is_anchored(self):
        _, cells = _column(3)
        anchored = anchored_cell_mask(
            cells, [True, True, True], [True, False, False]
        )
        np.testing.assert_array_equal(anchored, [True, True, True])

    def test_gap_cell_breaks_the_load_path(self):
        _, cells = _column(3)
        # middle cell carries no stiffness (unmelted powder): top floats
        anchored = anchored_cell_mask(
            cells, [True, False, True], [True, False, False]
        )
        np.testing.assert_array_equal(anchored, [True, False, False])

    def test_single_shared_vertex_counts_as_connected(self):
        cells = np.asarray(
            [[0, 1, 2, 3, 4, 5, 6, 7], [6, 8, 9, 10, 11, 12, 13, 14]]
        )
        anchored = anchored_cell_mask(cells, [True, True], [True, False])
        np.testing.assert_array_equal(anchored, [True, True])

    def test_anchor_cells_are_anchored_even_if_not_members(self):
        _, cells = _column(2)
        anchored = anchored_cell_mask(cells, [False, True], [True, False])
        np.testing.assert_array_equal(anchored, [True, True])

    def test_no_members_returns_all_false(self):
        _, cells = _column(2)
        anchored = anchored_cell_mask(cells, [False, False], [False, False])
        np.testing.assert_array_equal(anchored, [False, False])

    def test_shape_mismatch_is_rejected(self):
        _, cells = _column(2)
        with self.assertRaises(ValueError):
            anchored_cell_mask(cells, [True], [True, False])


class UpdateIslandDeferralTest(unittest.TestCase):
    def test_island_is_deferred_then_born_when_the_bridge_melts(self):
        _, cells = _column(3)
        anchor = np.asarray([True, False, False])
        # bridge (cell 1) still powder: cell 2 floats
        deferred, born = update_island_deferral(
            cells, [True, False, True], anchor, np.zeros(3, dtype=bool)
        )
        np.testing.assert_array_equal(deferred, [False, False, True])
        np.testing.assert_array_equal(born, [False, False, False])
        # bridge melted: cell 2 connects and is born this update
        deferred, born = update_island_deferral(
            cells, [True, True, True], anchor, deferred
        )
        np.testing.assert_array_equal(deferred, [False, False, False])
        np.testing.assert_array_equal(born, [False, False, True])
        # no second birth on the next update
        deferred, born = update_island_deferral(
            cells, [True, True, True], anchor, deferred
        )
        self.assertFalse(born.any())

    def test_anchored_growth_is_never_deferred_or_born(self):
        _, cells = _column(3)
        deferred, born = update_island_deferral(
            cells, [True, True, False], [True, False, False],
            np.zeros(3, dtype=bool),
        )
        self.assertFalse(deferred.any())
        self.assertFalse(born.any())


class IslandDeferralOptionTest(unittest.TestCase):
    def test_default_is_off_and_node_is_accepted(self):
        parser = build_parser()
        self.assertEqual(
            parser.parse_args([]).mechanics_island_deferral, "off"
        )
        self.assertEqual(
            parser.parse_args(
                ["--mechanics-island-deferral", "node"]
            ).mechanics_island_deferral,
            "node",
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(["--mechanics-island-deferral", "face"])


if __name__ == "__main__":
    unittest.main()
