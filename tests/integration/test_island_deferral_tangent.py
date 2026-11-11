"""A floating solid island makes the strict-domain mechanics tangent singular;
deferring it (--mechanics-island-deferral node) restores a positive-definite
system on the free DOFs.

Column of three HEX8 cells: fixture at the bottom, unmelted powder in the
middle (strict void layers: exactly zero stiffness), solid island on top.
"""

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse

from jax_fem.generate_mesh import Mesh
from jax_fem_am.materials.phases import (
    STATE_POWDER,
    STATE_SOLID,
    STATE_SUPPORT,
    make_quad_scalar,
    mechanics_material_quads,
)
from jax_fem_am.physics.mechanics import ThermoMechanical
from jax_fem_am.process.activation import (
    contributing_cell_mask,
    physical_node_mask,
    shared_node_cell_pairs,
    update_island_deferral,
)

jax.config.update("jax_enable_x64", True)

POINTS = np.asarray(
    [[x, y, z] for z in range(4) for (x, y) in ((0, 0), (1, 0), (1, 1), (0, 1))],
    dtype=np.float64,
)
CELLS = np.asarray(
    [[4 * k + i for i in range(8)] for k in range(3)], dtype=np.int32
)
PHASE_CELL = np.asarray([STATE_SUPPORT, STATE_POWDER, STATE_SOLID])
ANCHOR_CELL = np.asarray([True, False, False])
BOTTOM_NODES = np.arange(4)


def _args():
    return SimpleNamespace(
        young=1_000.0,
        alpha=0.0,
        poisson=0.3,
        mechanics_model="linear_elastic",
        inactive_mechanics_factor=1.0e-3,
        mushy_mechanics_factor=1.0e-3,
        liquid_mechanics_factor=1.0e-6,
        powder_solid_E=None,
        layer_activation_mode="layer_on_scan",
        future_layer_mode="void",
    )


def _free_tangent(mechanical_active_cell):
    problem = ThermoMechanical(
        mesh=Mesh(POINTS, CELLS, ele_type="HEX8"),
        vec=3,
        dim=3,
        ele_type="HEX8",
        dirichlet_bc_info=None,
        additional_info=("linear_elastic", None, 0.0, 0.0, (), False),
    )
    nq = problem.fes[0].num_quads
    active_quad = make_quad_scalar(
        np.asarray(mechanical_active_cell, dtype=np.float64), nq
    )
    phase_quad = make_quad_scalar(PHASE_CELL, nq)
    factor, young, alpha, poisson, yield_stress, hardening = (
        mechanics_material_quads(
            jnp.zeros_like(active_quad),
            active_quad,
            phase_quad,
            _args(),
            {"E": None, "alpha": None, "poisson": None},
        )
    )
    problem.set_params(
        [
            jnp.zeros_like(active_quad),
            jnp.zeros_like(active_quad),
            factor,
            young,
            alpha,
            poisson,
            yield_stress,
            hardening,
            jnp.zeros_like(active_quad),
        ]
    )
    problem.newton_update([jnp.zeros((len(POINTS), 3))])
    ndof = 3 * len(POINTS)
    tangent = scipy.sparse.coo_matrix(
        (problem.V, (problem.I, problem.J)), shape=(ndof, ndof)
    ).toarray()
    load_cell = contributing_cell_mask(factor)
    physical = physical_node_mask(CELLS, load_cell, num_nodes=len(POINTS))
    physical[BOTTOM_NODES] = False
    free = np.repeat(np.flatnonzero(physical) * 3, 3) + np.tile(
        np.arange(3), int(physical.sum())
    )
    return tangent[np.ix_(free, free)], load_cell


def _vertex_joint_tangent(mechanical_active_cell):
    """Fixed-bottom solid cell + solid cell touching it at one vertex (E10i-hybrid step 5070)."""
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    lattice = sorted({(x + dx, y + dy, z + dz) for (x, y, z) in ((0, 0, 0), (1, 1, 1))
                      for dx, dy, dz in corners})
    index = {p: k for k, p in enumerate(lattice)}
    points = np.asarray(lattice, dtype=np.float64)
    cells = np.asarray([[index[(x + dx, y + dy, z + dz)] for dx, dy, dz in corners]
                        for (x, y, z) in ((0, 0, 0), (1, 1, 1))], dtype=np.int32)
    problem = ThermoMechanical(
        mesh=Mesh(points, cells, ele_type="HEX8"), vec=3, dim=3, ele_type="HEX8",
        dirichlet_bc_info=None, additional_info=("linear_elastic", None, 0.0, 0.0, (), False),
    )
    nq = problem.fes[0].num_quads
    active_quad = make_quad_scalar(np.asarray(mechanical_active_cell, dtype=np.float64), nq)
    phase_quad = make_quad_scalar(np.asarray([STATE_SUPPORT, STATE_SOLID]), nq)
    factor, young, alpha, poisson, yield_stress, hardening = mechanics_material_quads(
        jnp.zeros_like(active_quad), active_quad, phase_quad, _args(),
        {"E": None, "alpha": None, "poisson": None},
    )
    problem.set_params([jnp.zeros_like(active_quad), jnp.zeros_like(active_quad), factor, young,
                        alpha, poisson, yield_stress, hardening, jnp.zeros_like(active_quad)])
    problem.newton_update([jnp.zeros((len(points), 3))])
    ndof = 3 * len(points)
    tangent = scipy.sparse.coo_matrix((problem.V, (problem.I, problem.J)), shape=(ndof, ndof)).toarray()
    load_cell = contributing_cell_mask(factor)
    physical = physical_node_mask(cells, load_cell, num_nodes=len(points))
    physical[points[:, 2] == 0.0] = False                      # clamped base
    free = np.repeat(np.flatnonzero(physical) * 3, 3) + np.tile(np.arange(3), int(physical.sum()))
    return tangent[np.ix_(free, free)], cells, load_cell, len(points)


def test_vertex_joint_is_a_mechanism_and_face_deferral_removes_it():
    stiff, cells, load_cell, n_points = _vertex_joint_tangent(np.ones(2, dtype=bool))
    eig = np.linalg.eigvalsh(stiff)
    # three rotations about the shared vertex carry no stiffness
    assert np.sum(eig < 1.0e-9 * eig.max()) == 3

    anchor = np.asarray([True, False])
    node_deferred, _ = update_island_deferral(
        cells, load_cell, anchor, np.zeros(2, dtype=bool), num_nodes=n_points
    )
    assert not node_deferred.any()          # 'node' accepts the vertex joint
    face_deferred, _ = update_island_deferral(
        cells, load_cell, anchor, np.zeros(2, dtype=bool), num_nodes=n_points,
        adjacency=shared_node_cell_pairs(cells, 3, num_nodes=n_points),
    )
    np.testing.assert_array_equal(face_deferred, [False, True])

    stiff_face, *_ = _vertex_joint_tangent(~face_deferred)
    eig_face = np.linalg.eigvalsh(stiff_face)
    assert eig_face.min() > 1.0e-6 * eig_face.max()


def test_floating_island_tangent_is_singular_and_deferral_fixes_it():
    printed = np.ones(3, dtype=bool)
    stiff_free, load_cell = _free_tangent(printed)
    np.testing.assert_array_equal(load_cell, [True, False, True])
    eig = np.linalg.eigvalsh(stiff_free)
    # six rigid-body modes of the island carry no stiffness at all
    assert np.sum(eig < 1.0e-9 * eig.max()) == 6

    deferred, born = update_island_deferral(
        CELLS, load_cell, ANCHOR_CELL, np.zeros(3, dtype=bool),
        num_nodes=len(POINTS),
    )
    np.testing.assert_array_equal(deferred, [False, False, True])
    assert not born.any()

    stiff_deferred, load_deferred = _free_tangent(printed & ~deferred)
    np.testing.assert_array_equal(load_deferred, [True, False, False])
    eig_deferred = np.linalg.eigvalsh(stiff_deferred)
    assert eig_deferred.min() > 1.0e-6 * eig_deferred.max()
