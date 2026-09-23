"""Unit tests for the weld_cfd2017 additions on the rongchi branch.

Covers the switches this branch adds on top of `test`:
  --prescribed-temperature-file   external nodal temperature history
  --solidification-reference      stress-free reference at solidification
  --bottom-mechanics-bc symmetry_plane / free_anchor
  --layer-activation-mode along_path + --bead-elsets   (weld bead element birth)
  cell-mean stress fields + --vtu-quad-arrays
"""
from types import SimpleNamespace

import numpy as onp
import pytest

from jax_fem_am.domain.events import update_phase_reference_and_eqp
from jax_fem_am.domain.state import StepState
from jax_fem_am.materials.phases import STATE_LIQUID, STATE_SOLID, STATE_VOID
from jax_fem_am.physics.release import make_symmetry_plane_mechanics_bc
from jax_fem_am.process.activation import (
    compute_along_path_cells,
    parse_bead_elsets,
    uses_strict_active_domain,
)
from jax_fem_am.simulation.stepper import (
    load_prescribed_temperature,
    prescribed_temperature_at,
)
import jax_fem_am.io.vtu as vtu


# --------------------------------------------------------------------- bead


def _state(step, bead, start, end, switch=1.0):
    return StepState(
        global_step=step, mode="weld", layer_idx=0, hatch_idx=0, scan_idx=step,
        laser_center=onp.array([0.5 * (start + end), 0.0, 0.006]), laser_power=0.0,
        laser_switch=switch, dt=0.01, scan_frac=0.0, hatch_frac=0.0, front_coord=0.006,
        layer_frac=0.0, bead_elset=bead,
        segment_start=None if start is None else onp.array([start, 0.0, 0.006]),
        segment_end=None if end is None else onp.array([end, 0.0, 0.006]),
        segment_id=step,
    )


def test_parse_bead_elsets():
    assert parse_bead_elsets(None) == ()
    assert parse_bead_elsets("BEAD") == ("BEAD",)
    assert parse_bead_elsets(" A , B ") == ("A", "B")
    with pytest.raises(ValueError):
        parse_bead_elsets("A,A")


def test_along_path_uses_strict_active_domain():
    assert uses_strict_active_domain(SimpleNamespace(layer_activation_mode="along_path"))
    assert not uses_strict_active_domain(
        SimpleNamespace(layer_activation_mode="front", future_layer_mode="void")
    )


def test_along_path_births_only_the_current_segment():
    # 6 base cells at x = 0.5..5.5 mm, 6 bead cells directly above them
    centroids = onp.zeros((12, 3))
    centroids[:6, 0] = onp.arange(6) + 0.5
    centroids[6:, 0] = onp.arange(6) + 0.5
    centroids[6:, 2] = 1.0
    bead = onp.zeros(12, dtype=bool)
    bead[6:] = True
    base = ~bead
    printed_bead = onp.zeros(12, dtype=bool)

    printed, active, cooling, segment = compute_along_path_cells(
        _state(0, "BEAD", 0.0, 2.0), {"BEAD": bead}, printed_bead, centroids, base
    )
    assert segment.sum() == 2                     # centroids 0.5 and 1.5
    assert printed[:6].all() and printed[6:8].all() and not printed[8:].any()
    assert active.tolist() == printed.tolist() and not cooling.any()

    # next segment adds two more; the earlier ones stay printed (cumulative)
    printed, _, _, segment = compute_along_path_cells(
        _state(1, "BEAD", 2.0, 4.0), {"BEAD": bead}, printed_bead, centroids, base
    )
    assert segment.sum() == 2
    assert printed[6:10].all() and not printed[10:].any()


def test_along_path_empty_segment_is_a_no_op():
    """A segment shorter than one cell deposits nothing instead of raising."""
    centroids = onp.zeros((4, 3))
    centroids[:, 0] = [0.5, 1.5, 0.5, 1.5]
    centroids[2:, 2] = 1.0
    bead = onp.array([False, False, True, True])
    printed_bead = onp.zeros(4, dtype=bool)
    printed, _, _, segment = compute_along_path_cells(
        _state(0, "BEAD", 0.0, 0.1), {"BEAD": bead}, printed_bead, centroids, ~bead
    )
    assert not segment.any()
    assert printed[:2].all() and not printed[2:].any()


def test_along_path_laser_off_deposits_nothing():
    centroids = onp.zeros((2, 3))
    centroids[:, 0] = [0.5, 0.5]
    centroids[1, 2] = 1.0
    bead = onp.array([False, True])
    printed, _, _, segment = compute_along_path_cells(
        _state(0, "BEAD", 0.0, 2.0, switch=0.0), {"BEAD": bead},
        onp.zeros(2, dtype=bool), centroids, ~bead
    )
    assert not segment.any() and not printed[1]


def test_along_path_unknown_elset_raises():
    centroids = onp.zeros((2, 3))
    bead = onp.array([False, True])
    with pytest.raises(ValueError, match="not in --bead-elsets"):
        compute_along_path_cells(
            _state(0, "OTHER", 0.0, 2.0), {"BEAD": bead},
            onp.zeros(2, dtype=bool), centroids, ~bead
        )


# ------------------------------------------------------------ symmetry BC


def _box_points(nx=3, ny=2, nz=2):
    x, y, z = onp.meshgrid(onp.arange(nx), onp.arange(ny), onp.arange(nz), indexing="ij")
    return onp.stack([x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")], axis=1).astype(float)


def test_symmetry_plane_bc_constrains_the_face_normal_plus_a_minimal_anchor():
    pts = _box_points()
    bc, meta = make_symmetry_plane_mechanics_bc(
        pts, plane_axis_id=1, side="min", return_metadata=True
    )
    assert meta["plane_axis"] == "y" and meta["side"] == "min"
    assert meta["plane_nodes"] == 6                      # nx*nz nodes at y=0
    assert meta["in_plane_axes"] == ["x", "z"]
    # four constraints: u_y on the plane, u_x and u_z at the anchor, u_z far away
    assert bc[1] == [1, 0, 2, 2]
    assert meta["anchor_node_id"] != meta["far_node_id"]
    assert pts[meta["anchor_node_id"], 1] == 0.0 and pts[meta["far_node_id"], 1] == 0.0
    assert pts[meta["far_node_id"], 0] > pts[meta["anchor_node_id"], 0]


def test_symmetry_plane_bc_rejects_a_degenerate_plane():
    flat = onp.zeros((4, 3))
    flat[:, 2] = [0, 1, 0, 1]
    with pytest.raises(ValueError, match="no extent along the in-plane axis"):
        make_symmetry_plane_mechanics_bc(flat, plane_axis_id=1, side="min", return_metadata=True)
    with pytest.raises(ValueError):
        make_symmetry_plane_mechanics_bc(_box_points(), plane_axis_id=3)


# ------------------------------------------------- prescribed temperature


def _write_npz(tmp_path, times, T, points=None):
    path = tmp_path / "T.npz"
    data = dict(time=onp.asarray(times), T=onp.asarray(T))
    if points is not None:
        data["points"] = onp.asarray(points)
    onp.savez(path, **data)
    return str(path)


def test_prescribed_temperature_interpolates_and_holds_the_ends(tmp_path):
    pts = onp.zeros((3, 3))
    p = load_prescribed_temperature(_write_npz(tmp_path, [0.0, 1.0], [[300.0] * 3, [400.0] * 3]), pts, 300.0)
    assert onp.allclose(onp.asarray(prescribed_temperature_at(p, -1.0)).ravel(), 300.0)
    assert onp.allclose(onp.asarray(prescribed_temperature_at(p, 0.25)).ravel(), 325.0)
    assert onp.allclose(onp.asarray(prescribed_temperature_at(p, 9.0)).ravel(), 400.0)
    assert onp.asarray(prescribed_temperature_at(p, 0.5)).shape == (3, 1)


def test_prescribed_temperature_validates_shape_times_and_node_order(tmp_path):
    pts = onp.zeros((3, 3))
    with pytest.raises(ValueError, match="n_nodes"):
        load_prescribed_temperature(_write_npz(tmp_path, [0.0], [[300.0, 300.0]]), pts, 300.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_prescribed_temperature(_write_npz(tmp_path, [1.0, 0.0], [[300.0] * 3] * 2), pts, 300.0)
    with pytest.raises(ValueError, match="non-finite"):
        load_prescribed_temperature(_write_npz(tmp_path, [0.0], [[300.0, onp.nan, 300.0]]), pts, 300.0)
    moved = _box_points(2, 2, 2)
    shifted = moved.copy()
    shifted[0, 0] += 0.5
    with pytest.raises(ValueError, match="node order mismatch"):
        load_prescribed_temperature(
            _write_npz(tmp_path, [0.0], [[300.0] * 8], points=shifted), moved, 300.0
        )


# --------------------------------------------- solidification reference


def _events_args(reference):
    return SimpleNamespace(
        solidus_temperature=858.0, liquidus_temperature=923.0,
        phase_history_model="legacy_reset", stress_relaxation_temperature=None,
        solidification_reference=reference, reset_plastic_on_melt=False,
        reset_plastic_on_solidify=True,
    )


@pytest.mark.parametrize("reference,expected", [("temperature", 700.0), ("solidus", 858.0)])
def test_solidification_reference(reference, expected):
    """A point seen solid at 700 K takes either that temperature or the solidus."""
    shape = (1, 1, 1)
    T = onp.full(shape, 700.0)
    active = onp.ones(shape)
    phase = onp.full(shape, STATE_LIQUID)
    T_ref = onp.full(shape, 298.0)
    eqp = onp.full(shape, 0.3)
    phase_new, T_ref_new, eqp_new, newly, _ = update_phase_reference_and_eqp(
        T, active, phase, T_ref, eqp, _events_args(reference)
    )
    assert bool(onp.asarray(newly).ravel()[0])
    assert onp.asarray(phase_new).ravel()[0] == STATE_SOLID
    assert onp.asarray(T_ref_new).ravel()[0] == pytest.approx(expected)
    assert onp.asarray(eqp_new).ravel()[0] == 0.0          # reset on solidify


def test_solidification_reference_leaves_unsolidified_points_alone():
    shape = (1, 1, 1)
    phase_new, T_ref_new, _, newly, _ = update_phase_reference_and_eqp(
        onp.full(shape, 1000.0), onp.ones(shape), onp.full(shape, STATE_VOID),
        onp.full(shape, 298.0), onp.zeros(shape), _events_args("solidus")
    )
    assert not bool(onp.asarray(newly).ravel()[0])
    assert onp.asarray(phase_new).ravel()[0] == STATE_LIQUID
    assert onp.asarray(T_ref_new).ravel()[0] == pytest.approx(298.0)


# ---------------------------------------------------------- VTU cell data


def _quad_stress(num_cells=2, num_quads=8):
    stress = onp.zeros((num_cells, num_quads, 3, 3))
    stress[:, :, 0, 0] = onp.arange(num_quads)[None, :]        # mean = 3.5
    stress[:, :, 0, 1] = 2.0
    stress[:, :, 1, 0] = 4.0                                   # xy mean = 3.0
    return {"stress_quad": stress, "vm_quad": onp.full((num_cells, num_quads), 7.0)}


def test_cell_mean_stress_fields_are_written():
    infos = dict((k, v) for k, v in vtu.make_quad_stress_cell_infos(_quad_stress()))
    assert onp.allclose(infos["sigma_xx"], 3.5)
    assert onp.allclose(infos["sigma_xy"], 3.0)                # symmetrised
    assert onp.allclose(infos["von_mises"], 7.0)
    assert "stress_quad0_xx" in infos and "vm_quad7" in infos


def test_quad_arrays_can_be_dropped_from_the_vtu():
    infos = vtu.make_quad_stress_cell_infos(_quad_stress())
    kept = [name for name, _ in infos if not vtu._QUAD_ARRAY_RE.search(name)]
    assert set(kept) == {"sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", "sigma_yz", "sigma_xz", "von_mises"}
    assert len(kept) < len(infos)
