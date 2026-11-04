"""cudss-plus 2026-11-04 (option D): physical_node_mask by boolean scatter equals the np.unique version."""
import numpy as onp
import pytest

from jax_fem_am.process.activation import physical_node_mask


def _reference(cells, physical_cell, num_nodes):
    mask = onp.zeros(num_nodes, dtype=bool)
    selected = cells[physical_cell]
    if selected.size:
        mask[onp.unique(selected)] = True
    return mask


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_unique_reference(seed):
    rng = onp.random.default_rng(seed)
    num_nodes = 900
    cells = rng.integers(0, num_nodes, size=(400, 8))
    for frac in (0.0, 0.05, 0.5, 1.0):
        physical = rng.random(400) < frac
        out = physical_node_mask(cells, physical, num_nodes=num_nodes)
        assert out.dtype == bool and out.shape == (num_nodes,)
        assert onp.array_equal(out, _reference(cells, physical, num_nodes))


def test_shared_nodes_and_default_num_nodes():
    cells = onp.array([[0, 1, 2, 3], [2, 3, 4, 5]])
    out = physical_node_mask(cells, onp.array([False, True]))
    assert out.tolist() == [False, False, True, True, True, True]
