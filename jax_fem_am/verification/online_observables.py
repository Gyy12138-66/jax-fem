"""Online in-circle pyrometer observables, accumulated every solver step.

WHY THIS EXISTS
---------------
The V2 thermal gate writes a VTU every ``--thermal-output-every`` steps.  At the
gate configuration that is 7.6923 ms, which beats against the 15.101 ms track
period: frames land on the beam only every other frame, every hot frame in the
production run is 15.33 ms apart, and the sampling phase walks across the 2 mm
circle instead of staying put (registered as D-V2-25).  Consequences measured on
the 2026-08-19 production run: the t = 0.548 s experimental point is
unscoreable, the 10 ms bin average degenerates to single-frame sampling, and the
peak-instant falsification check cannot be read.

The beam is inside the circle for only 2 mm / 650 mm/s = 3.077 ms.  Chasing this
with denser VTU output is the wrong tool -- 0.5 ms full-field frames over a
0.45 s window would be 900 whole-mesh files.  What the comparison actually needs
is a handful of scalars per step, so this module accumulates them ONLINE at the
solver's own cadence (dt = 76.9 us at the gate condition) and writes one small
JSONL row per step.  No VTU is written, no field is stored.

WHAT IT RECORDS (per step, inside the configured time window)
------------------------------------------------------------
Geometry is identical to ``analyze_pyrometer.py`` -- same circle centre, same
diameter, same top-layer-only depth, same 8-node-mean cell temperature -- so the
online series and the VTU-derived series are the same observable sampled at
different rates, and can be compared directly.

  time / global_step / mode / laser_on / laser centre / beam-to-centre distance
  n_hot, avg_K                 the ADOPTED Balbaa Sec 3.3 conditional average
  max_K, n_over_range          field peak and instrument-ceiling exceedances
  two_colour_K (+ S1, S2)      the D-V2-24 synthetic instrument
  full_spot_avg_K              the D-V2-24 unthresholded diagnostic bound
  probe_K[]                    fixed-point probes for the Fig 15 / 16 target
                               (D-V2-27) -- no conditional average, no n_hot.
                               Each probe reads the top-layer cell CONTAINING
                               its coordinate, as an 8-node mean, per scoring
                               spec 6.1; see `probe_resolution` in the meta file

The two channel signals S1 and S2 are written per step precisely so a consumer
can form a true 10 ms RESPONSE INTEGRAL by averaging radiance over the window
and inverting once, instead of averaging temperatures.

RED LINES (IET-20)
------------------
* Default OFF.  ``OnlineObservableRecorder`` is only constructed when
  ``--online-observables`` is passed; with no new flag the run takes exactly the
  same code path as before.
* It never touches the solution, the residual, the material state or any
  configuration key.  It reads the accepted temperature vector and writes a
  file.  It cannot change the physics of either arm.
* Failures are reported, not swallowed silently, but they abort the recorder
  rather than the run -- an observability feature must not be able to kill a
  7.5 h production run.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

C2K = 273.15

# CODATA 2018.  Kept local rather than imported from the V2 case model so this
# module has no dependency outside jax_fem_am; the two are cross-checked by
# tests/unit/test_online_observables.py.
_H = 6.62607015e-34
_C = 2.99792458e8
_KB = 1.380649e-23
C2 = _H * _C / _KB


def _spectral_radiance(temperature_k, wavelength_m):
    """Planck spectral radiance, black body.  expm1 keeps cold cells stable."""
    t = np.asarray(temperature_k, dtype=np.float64)
    out = np.zeros_like(t)
    ok = t > 1.0
    if not np.any(ok):
        return out
    with np.errstate(over="ignore"):
        denom = np.expm1(C2 / (wavelength_m * t[ok]))
    prefactor = 2.0 * _H * _C**2 / wavelength_m**5
    out[ok] = np.where(np.isfinite(denom) & (denom > 0.0), prefactor / denom, 0.0)
    return out


def _ratio_at(temperature_k, wavelengths):
    a = _spectral_radiance(np.atleast_1d(temperature_k), wavelengths[0])[0]
    b = _spectral_radiance(np.atleast_1d(temperature_k), wavelengths[1])[0]
    return a / b if b > 0.0 else math.inf


def invert_two_colour_ratio(ratio, wavelengths, t_lo=200.0, t_hi=100000.0):
    """Invert a radiance ratio for temperature.  Monotone in T for l1 < l2.

    The bracket is deliberately far above any boiling point: clamping the
    inversion at a physically motivated ceiling would be calibration.
    """
    if not np.isfinite(ratio) or ratio <= 0.0:
        return None
    lo, hi = float(t_lo), float(t_hi)
    if not (_ratio_at(lo, wavelengths) <= ratio <= _ratio_at(hi, wavelengths)):
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _ratio_at(mid, wavelengths) < ratio:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0e-9 * max(1.0, mid):
            break
    return 0.5 * (lo + hi)


# HEX8 reference-element corners in the C3D8 node order (bottom face CCW, then
# top face CCW), used for the trilinear gradient below.
_HEX8_REF = np.array([
    [-1.0, -1.0, -1.0], [+1.0, -1.0, -1.0], [+1.0, +1.0, -1.0], [-1.0, +1.0, -1.0],
    [-1.0, -1.0, +1.0], [+1.0, -1.0, +1.0], [+1.0, +1.0, +1.0], [-1.0, +1.0, +1.0],
], dtype=np.float64)


def trilinear_gradient_at_centre(node_xyz, node_values):
    """grad(T) of the trilinear field at the element centre (xi = eta = zeta = 0).

    Scoring spec 6.3.5 wants a model-side |dT/dx| and |dT/dy|; the definition
    adopted (Fable5 2026-08-20, registered under D-V2-27) is the trilinear
    interpolant's gradient at the centre of the CONTAINING cell -- the same cell
    the probe already reads, so the gradient and the temperature cannot come
    from different places.

    At the centre the shape-function derivatives are dN_i/dxi_b = xi_i[b] / 8,
    so with J[a][b] = sum_i (dN_i/dxi_b) * x_i[a] and dT/dxi_b = sum_i
    (dN_i/dxi_b) * T_i, the chain rule gives dT/dxi = J^T grad, i.e.
    grad = solve(J^T, dT/dxi). Done through the Jacobian rather than as a face
    difference so it stays exact for any hex, not just an axis-aligned brick;
    on a structured grid it reduces to the +/-x, +/-y face-mean difference
    quotient, which is what makes it auditable by hand.

    Returns None when the element is degenerate (singular Jacobian) rather than
    emitting a fabricated gradient.
    """
    xyz = np.asarray(node_xyz, dtype=np.float64)
    values = np.asarray(node_values, dtype=np.float64).reshape(-1)
    if xyz.shape != (8, 3) or values.shape != (8,):
        raise ValueError("trilinear gradient needs 8 nodes with 8 values")
    dn = _HEX8_REF / 8.0                       # (8, 3) dN_i/dxi_b at the centre
    jac = xyz.T @ dn                           # J[a][b] = dx_a / dxi_b
    dt_dxi = dn.T @ values                     # (3,)
    try:
        return np.linalg.solve(jac.T, dt_dxi)
    except np.linalg.LinAlgError:
        return None


def uniform_field_self_check(wavelengths, temperatures=(1273.15, 2500.0, 4800.0),
                             atol_k=1.0e-4):
    """A uniform field must invert back to itself.  Run before recording."""
    rows, ok = [], True
    for t0 in temperatures:
        field = np.full(32, float(t0))
        s1 = float(_spectral_radiance(field, wavelengths[0]).mean())
        s2 = float(_spectral_radiance(field, wavelengths[1]).mean())
        got = invert_two_colour_ratio(s1 / s2, wavelengths) if s2 > 0 else None
        err = None if got is None else abs(got - t0)
        good = err is not None and err <= atol_k
        ok = ok and good
        rows.append({"T_input_K": t0, "T_readback_K": got, "abs_error_K": err,
                     "pass": good})
    return ok, rows


class OnlineObservableRecorder:
    """Accumulate in-circle pyrometer observables at solver cadence.

    Constructed lazily on the first solver step that falls inside the window, so
    a run whose window never opens writes nothing at all.
    """

    SCHEMA = "v06.online-observables/1"

    def __init__(self, output_dir, *, spot_center_m, spot_diameter_m,
                 threshold_c, range_max_c, wavelengths_m, window_s,
                 every, probes_m, run_id=None, layer_top_z_m=None,
                 layer_thickness_m=40.0e-6,
                 all_depths=False, filename="online_observables.jsonl",
                 probe_containment_tol_m=1.0e-12, probe_tie_tol_m=1.0e-12):
        self.output_dir = str(output_dir)
        self.path = os.path.join(self.output_dir, filename)
        self.meta_path = os.path.join(
            self.output_dir, filename.replace(".jsonl", "_meta.json"))
        self.spot_center_m = spot_center_m       # None -> mesh xy bbox centre
        self.spot_diameter_m = float(spot_diameter_m)
        self.threshold_k = float(threshold_c) + C2K
        self.range_max_k = float(range_max_c) + C2K
        self.wavelengths_m = tuple(float(v) for v in wavelengths_m)
        self.window_s = (float(window_s[0]), float(window_s[1]))
        self.every = max(1, int(every))
        self.run_id = str(run_id) if run_id else None
        self.probes_m = [tuple(float(c) for c in p) for p in (probes_m or [])]
        self.layer_top_z_m = layer_top_z_m
        self.layer_thickness_m = float(layer_thickness_m)
        self.all_depths = bool(all_depths)
        # A pre-registered probe coordinate can land exactly on a cell face, so
        # containment needs a round-off tolerance; the tie tolerance implements
        # spec 6.1's "distances tied" branch. Both are round-off scale, not
        # snapping distances -- widening them would start moving probes, which
        # the spec forbids.
        self.probe_containment_tol_m = float(probe_containment_tol_m)
        self.probe_tie_tol_m = float(probe_tie_tol_m)

        self.time_s = 0.0
        self.step_index = 0
        self.rows_written = 0
        self.disabled_reason = None
        self._geometry = None
        self._handle = None

    # ---------------------------------------------------------------- geometry
    def _build_geometry(self, problem):
        """One-off: resolve the gauge cell set and the probe node indices."""
        fe = problem.fes[0]
        points = np.asarray(getattr(fe, "points"), dtype=np.float64)
        cells = np.asarray(getattr(fe, "cells"), dtype=np.int64)
        centers = points[cells].mean(axis=1)

        if self.spot_center_m is None:
            cx = 0.5 * (points[:, 0].min() + points[:, 0].max())
            cy = 0.5 * (points[:, 1].min() + points[:, 1].max())
        else:
            cx, cy = self.spot_center_m
        radius = 0.5 * self.spot_diameter_m
        in_circle = ((centers[:, 0] - cx) ** 2
                     + (centers[:, 1] - cy) ** 2) <= radius**2

        z_top = points[:, 2].max()
        layer_bottom = (self.layer_top_z_m if self.layer_top_z_m is not None
                        else z_top - self.layer_thickness_m)
        in_depth = (np.ones(len(centers), bool) if self.all_depths
                    else centers[:, 2] >= layer_bottom)
        gauge = in_circle & in_depth
        if not gauge.any():
            raise ValueError(
                "online observables: the gauge cell set is empty -- check "
                "--online-observables-spot-center / -diameter / depth options")

        # ---- probes: the CELL CONTAINING the point, per scoring spec 6.1 ----
        # NOT the nearest node. On a hex cell the eight corners are equidistant
        # from the cell centre, so a nearest-node argmin degenerates into "the
        # corner with the smallest node index" -- an arbitrary corner, not the
        # element. At the gradients this case actually runs (Fig 16 gives
        # |dT/dy| ~ 1750 degC/mm) a corner and the 8-node mean differ by tens of
        # K across a 40 um cell, which is wider than the +/-14 degC digitisation
        # budget the Fig 15/16 leg is scored against.
        #
        # The probe cell set is the WHOLE top layer, not the gauge set: the
        # pre-registered probes P1=(1,2) and P3=(3,2) mm sit on and outside the
        # 2 mm circle centred at (2,2) mm.
        probe_pool = np.where(in_depth)[0]
        if self.probes_m and probe_pool.size == 0:
            raise ValueError(
                "online observables: no top-layer cells to resolve probes into")
        node_xyz = points[cells[probe_pool]]          # (n_pool, 8, 3)
        lower = node_xyz.min(axis=1)
        upper = node_xyz.max(axis=1)
        pool_centers = centers[probe_pool]

        probe_records = []
        for probe in self.probes_m:
            point = np.asarray(probe, dtype=np.float64)
            # Containment on the element's axis-aligned bounds. Exact for the
            # structured hex meshes this case uses; tol absorbs the round-off
            # that puts a pre-registered coordinate exactly on a face.
            tol = self.probe_containment_tol_m
            inside = np.all((point >= lower - tol) & (point <= upper + tol), axis=1)
            offsets = np.linalg.norm(pool_centers - point, axis=1)
            candidates = np.where(inside)[0]
            contains = candidates.size > 0
            if not contains:
                # Spec 6.1 only legislates ties, not "outside the mesh". Resolve
                # to the nearest cell centre so an observability feature cannot
                # abort a run, but record contains_probe=false so the C package
                # can see the probe was never actually inside the domain.
                candidates = np.arange(pool_centers.shape[0])
            # Spec 6.1 tie-break, in order: nearest cell-centre Euclidean
            # distance, then smallest element id.
            best_offset = offsets[candidates].min()
            tied = candidates[
                np.isclose(offsets[candidates], best_offset,
                           rtol=0.0, atol=self.probe_tie_tol_m)]
            cell_index = int(probe_pool[tied].min())          # smallest element id
            local = int(np.where(probe_pool == cell_index)[0][0])
            probe_records.append({
                "requested_m": [float(v) for v in point],
                "cell_index": cell_index,
                "cell_center_m": [float(v) for v in centers[cell_index]],
                "offset_from_cell_center_m": float(offsets[local]),
                "contains_probe": bool(contains),
                "n_candidate_cells": int(candidates.size),
                "n_tied_on_distance": int(tied.size),
                "cell_bounds_m": {
                    "lower": [float(v) for v in lower[local]],
                    "upper": [float(v) for v in upper[local]],
                },
            })

        for record, probe in zip(probe_records, self.probes_m):
            if not record["contains_probe"]:
                print("online observables WARNING: probe "
                      f"{tuple(probe)} m is not inside any top-layer cell; "
                      f"resolved to the nearest one (element {record['cell_index']}, "
                      f"offset {record['offset_from_cell_center_m']:.3e} m). "
                      "Reported as contains_probe=false -- this is NOT a "
                      "spec-6.1-compliant probe reading.")

        self._geometry = {
            "gauge_conn": cells[gauge],
            "probe_conn": (cells[[r["cell_index"] for r in probe_records]]
                           if probe_records else np.zeros((0, 8), dtype=np.int64)),
            # nodal coordinates, kept for the spec-6.3.5 trilinear gradient
            "probe_points": points,
            "spot_center_m": [float(cx), float(cy)],
            "layer_bottom_z_m": float(layer_bottom),
            "gauge_cells": int(gauge.sum()),
            "gauge_cells_in_circle": int(in_circle.sum()),
            "top_layer_cells": int(in_depth.sum()),
            "probes": probe_records,
        }
        return self._geometry

    def _write_meta(self):
        geometry = self._geometry
        ok, self_check = uniform_field_self_check(self.wavelengths_m)
        meta = {
            "schema_version": self.SCHEMA,
            "claim_level": "solver_step_observable_extraction_only",
            "run_id": self.run_id,
            "_what": "in-circle pyrometer observables accumulated at solver "
                     "cadence; no field is stored and no VTU is written",
            "_why": "D-V2-25: 7.69 ms frame spacing beats against the 15.10 ms "
                    "track period, so VTU-derived series are phase-aliased",
            "geometry_matches": "analyze_pyrometer.py (same circle, depth and "
                                "cell-temperature definition), so the online "
                                "and VTU series are the same observable",
            "spot_center_m": geometry["spot_center_m"],
            "spot_diameter_m": self.spot_diameter_m,
            "threshold_C": self.threshold_k - C2K,
            "range_max_C": self.range_max_k - C2K,
            "depth_scope": ("all depths" if self.all_depths
                            else f"top layer (z >= {geometry['layer_bottom_z_m']:.6e} m)"),
            "gauge_cells": geometry["gauge_cells"],
            "gauge_cells_in_circle": geometry["gauge_cells_in_circle"],
            "top_layer_cells": geometry["top_layer_cells"],
            "window_s": list(self.window_s),
            "record_every_n_steps": self.every,
            "two_colour": {
                "wavelengths_m": list(self.wavelengths_m),
                "assumption": "grey body -- emissivity cancels in the ratio",
                "registered_as": "D-V2-24",
                "uniform_field_self_check_passed": ok,
                "uniform_field_self_check": self_check,
                "bin_reading_note": "S1/S2 are written per step so a consumer "
                                    "can average RADIANCE over a 10 ms response "
                                    "window and invert once",
            },
            "probes": geometry["probes"],
            "probe_purpose": "Fig 15/16 fixed-point target (D-V2-27): no "
                             "conditional average, so no 1/n_hot dilution",
            "probe_resolution": {
                "_spec": "scoring-spec-thermal-gate-v2.md 6.1 @ 3b9c220",
                "rule": "value is taken from the top-layer cell CONTAINING the "
                        "coordinate; cell temperature = arithmetic mean of its 8 "
                        "nodes. Boundary ambiguity resolves to the nearest cell "
                        "centre by Euclidean distance, ties to the smallest "
                        "element id. Probes are never moved to fit.",
                "element_id_convention": "0-based row index into fe.cells, i.e. "
                                         "the element order of the .inp mesh. The "
                                         "spec says 'smallest element id' without "
                                         "fixing the numbering base -- recorded "
                                         "here so the tie-break is auditable.",
                "pool": "the WHOLE top layer, not the gauge set: the "
                        "pre-registered probes at (1,2) and (3,2) mm sit on and "
                        "outside the 2 mm circle centred at (2,2) mm.",
                "containment_tol_m": self.probe_containment_tol_m,
                "tie_tol_m": self.probe_tie_tol_m,
                "not_nearest_node": "a nearest-node argmin would degenerate to an "
                                    "arbitrary corner (the 8 corners of a hex are "
                                    "equidistant from its centre) and, at the "
                                    "~1750 degC/mm gradients of Fig 16, differ "
                                    "from the cell mean by more than the +/-14 "
                                    "degC digitisation budget.",
                "all_probes_contained": all(p["contains_probe"]
                                            for p in geometry["probes"]),
            },
            "probe_gradient": {
                "_spec": "scoring-spec-thermal-gate-v2.md 6.3.5",
                "field": "probe_grad_K_per_m = [dT/dx, dT/dy, dT/dz] in K/m",
                "definition": "gradient of the trilinear interpolant evaluated at "
                              "the centre of the CONTAINING cell -- the same cell "
                              "the probe temperature is read from, so temperature "
                              "and gradient cannot come from different places",
                "equivalent_form": "on a structured axis-aligned grid this reduces "
                                   "to the +/-x and +/-y face-mean difference "
                                   "quotient, i.e. checkable by hand",
                "sign_convention": "the signed vector is written; spec 6.3.5 wants "
                                   "|dT/dx| and |dT/dy|, and taking the modulus "
                                   "downstream keeps the sign auditable here",
                "axis_swap_prohibition": "spec 6.3.5 forbids swapping x and y to "
                                         "improve agreement; scan direction is +x "
                                         "and hatch direction is +y by the D-V2-09 "
                                         "path convention, fixed before any run",
                "degenerate_element": "a singular Jacobian yields null rather than "
                                      "a fabricated gradient",
                "registered_as": "D-V2-27",
            },
            "band_accounting": {
                "_spec": "scoring-spec-thermal-gate-v2.md 3.2 two-colour reporting",
                "fields": ["band_cells", "band_radiance_weight_short",
                           "band_radiance_weight_long",
                           "two_colour_inversion_failed"],
                "bands": {"below_range": f"< {self.threshold_k - C2K:g} degC",
                          "in_range": f"{self.threshold_k - C2K:g}-"
                                      f"{self.range_max_k - C2K:g} degC",
                          "above_range": f"> {self.range_max_k - C2K:g} degC"},
                "why_online": "cell counts could be recomputed later, but the "
                              "RADIANCE weights depend on the field, and the field "
                              "is deliberately never written to disk -- so these "
                              "have to be produced during the solve or not at all",
                "time_weighting": "recoverable downstream: every record carries "
                                  "its own dt_s",
            },
            "zero_calibration": "nothing here is tunable toward the measurement",
        }
        if not ok:
            raise ValueError(
                "online observables: two-colour uniform-field self-check failed; "
                "refusing to record a reading that cannot reproduce a known field")
        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------- recording
    def observe(self, problem, temperature_new, step_state):
        """Record one accepted solve, failing closed on any recorder error.

        Once explicitly enabled, these values are part of the preregistered
        score rather than best-effort telemetry.  Continuing after an error
        could make a truncated JSONL look like a valid response trace.
        """
        if self.disabled_reason is not None:
            raise RuntimeError(
                f"online observables previously failed: {self.disabled_reason}")
        try:
            self._observe(problem, temperature_new, step_state)
        except Exception as error:
            self.disabled_reason = f"{type(error).__name__}: {error}"
            self._close()
            raise RuntimeError(
                f"online observables failed after {self.rows_written} rows: "
                f"{self.disabled_reason}") from error

    def _observe(self, problem, temperature_new, step_state):
        # Absolute time is the running sum of the per-step dt, which is exactly
        # how path_used.csv assigns time to a step -- so the online series and
        # the VTU series share one clock.
        self.time_s += float(step_state.dt)
        self.step_index += 1
        t_now = self.time_s
        lo, hi = self.window_s
        if t_now < lo or t_now > hi:
            return
        if (self.step_index - 1) % self.every:
            return

        if self._geometry is None:
            self._build_geometry(problem)
            self._write_meta()
            self._handle = open(self.path, "w", encoding="utf-8")

        geometry = self._geometry
        temperature = np.asarray(temperature_new, dtype=np.float64).reshape(-1)
        cell_t = temperature[geometry["gauge_conn"]].mean(axis=1)

        hot = cell_t >= self.threshold_k
        n_hot = int(hot.sum())
        l1 = _spectral_radiance(cell_t, self.wavelengths_m[0])
        l2 = _spectral_radiance(cell_t, self.wavelengths_m[1])
        s1, s2 = float(l1.mean()), float(l2.mean())
        two_colour = (invert_two_colour_ratio(s1 / s2, self.wavelengths_m)
                      if s2 > 0.0 else None)

        # ---- spec 3.2 band accounting ------------------------------------
        # The spec requires the below-1000 / 1000-3000 / above-3000 degC cell
        # AND time weights plus the non-invertible count. Cell counts could be
        # recomputed afterwards, but the RADIANCE weights cannot: they depend on
        # the field, and the field is deliberately not written to disk. So they
        # have to be produced here or not at all. Time weighting is recoverable
        # downstream because every record carries its own dt_s.
        below = cell_t < self.threshold_k
        above = cell_t > self.range_max_k
        within = ~below & ~above
        total_l1 = float(l1.sum())
        total_l2 = float(l2.sum())

        def _weight(mask, channel, total):
            return float(channel[mask].sum() / total) if total > 0.0 else None

        centre = geometry["spot_center_m"]
        laser = np.asarray(step_state.laser_center, dtype=np.float64).reshape(-1)
        beam_distance = float(math.hypot(laser[0] - centre[0],
                                         laser[1] - centre[1]))

        row = {
            "step_index": self.step_index - 1,
            "global_step": int(step_state.global_step),
            "time_s": t_now,
            "dt_s": float(step_state.dt),
            "mode": str(step_state.mode),
            "laser_on": bool(float(step_state.laser_switch) > 0.5),
            "laser_center_m": [float(laser[0]), float(laser[1])],
            "beam_to_spot_centre_m": beam_distance,
            "beam_inside_spot": bool(beam_distance <= 0.5 * self.spot_diameter_m),
            # adopted Balbaa Sec 3.3 conditional average
            "n_hot": n_hot,
            "avg_K": float(cell_t[hot].mean()) if n_hot else None,
            "max_K": float(cell_t.max()),
            "n_over_range": int((hot & (cell_t > self.range_max_k)).sum()),
            # D-V2-24 two-colour synthetic instrument
            "two_colour_K": two_colour,
            "two_colour_S1": s1,
            "two_colour_S2": s2,
            "two_colour_over_range": bool(
                two_colour is not None and two_colour > self.range_max_k),
            "two_colour_inversion_failed": bool(two_colour is None),
            # spec 3.2: cell counts and RADIANCE weights per instrument band
            "band_cells": {"below_range": int(below.sum()),
                           "in_range": int(within.sum()),
                           "above_range": int(above.sum())},
            "band_radiance_weight_short": {
                "below_range": _weight(below, l1, total_l1),
                "in_range": _weight(within, l1, total_l1),
                "above_range": _weight(above, l1, total_l1)},
            "band_radiance_weight_long": {
                "below_range": _weight(below, l2, total_l2),
                "in_range": _weight(within, l2, total_l2),
                "above_range": _weight(above, l2, total_l2)},
            # D-V2-24 unthresholded diagnostic bound
            "full_spot_avg_K": float(cell_t.mean()),
        }
        if geometry["probe_conn"].size:
            # Spec 6.1: the probe value is its CELL's temperature, and cell
            # temperature is the arithmetic mean of the 8 nodes -- the same
            # definition the adopted reading uses, so the two legs cannot
            # silently disagree about what "temperature at a place" means.
            row["probe_K"] = [float(v) for v in
                              temperature[geometry["probe_conn"]].mean(axis=1)]
            # Spec 6.3.5: model-side gradient, trilinear at the containing
            # cell's centre. Reported as the signed vector; the spec asks for
            # |dT/dx| and |dT/dy|, and taking the absolute value downstream
            # keeps the sign auditable here. dT/dz comes along free and is
            # worth having -- it is the build-direction gradient.
            grads = []
            for nodes in geometry["probe_conn"]:
                gradient = trilinear_gradient_at_centre(
                    geometry["probe_points"][nodes], temperature[nodes])
                grads.append(None if gradient is None
                             else [float(v) for v in gradient])
            row["probe_grad_K_per_m"] = grads
        self._handle.write(json.dumps(row) + "\n")
        self.rows_written += 1

    def _close(self):
        if self._handle is not None:
            try:
                self._handle.flush()
                self._handle.close()
            finally:
                self._handle = None

    def finalize(self):
        """Flush and print a one-line summary.  Safe to call more than once."""
        self._close()
        if self.disabled_reason is not None:
            raise RuntimeError(
                f"online observables incomplete after {self.rows_written} rows: "
                f"{self.disabled_reason}")
        if self.rows_written == 0:
            print("online observables: enabled but no step fell inside the "
                  f"window {self.window_s} (last time {self.time_s:.6f} s) -- "
                  "nothing written")
            return
        print(f"online observables: {self.rows_written} rows over "
              f"t = {self.window_s[0]}..{self.window_s[1]} s "
              f"(every {self.every} step(s)) -> {self.path}")


def recorder_from_args(args):
    """Build a recorder from parsed CLI args, or None when the flag is absent.

    The ``getattr`` defaults mean an args namespace that predates these flags
    (older configs, the test fixtures) simply yields None instead of raising.
    """
    if not getattr(args, "online_observables", False):
        return None

    def _pair(text, default):
        if not text:
            return default
        return tuple(float(v) for v in str(text).split(","))

    probes = []
    raw_probes = getattr(args, "online_observables_probes", "") or ""
    for chunk in (c for c in raw_probes.split(";") if c.strip()):
        parts = [float(v) for v in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(
                "--online-observables-probes wants x,y,z triples separated by "
                f"';', got {chunk!r}")
        probes.append(parts)

    spot = getattr(args, "online_observables_spot_center", "") or ""
    window = _pair(getattr(args, "online_observables_window", None), (0.45, 0.90))
    if len(window) != 2 or not window[0] < window[1]:
        raise ValueError("--online-observables-window wants 't_lo,t_hi' with "
                         f"t_lo < t_hi, got {window}")
    return OnlineObservableRecorder(
        args.output_dir,
        spot_center_m=(_pair(spot, None) if spot else None),
        spot_diameter_m=getattr(args, "online_observables_spot_diameter", 2.0e-3),
        threshold_c=getattr(args, "online_observables_threshold_c", 1000.0),
        range_max_c=getattr(args, "online_observables_range_max_c", 3000.0),
        wavelengths_m=tuple(
            v * 1.0e-6 for v in _pair(
                getattr(args, "online_observables_wavelengths_um", "0.95,1.05"),
                (0.95, 1.05))),
        window_s=window,
        every=getattr(args, "online_observables_every", 1),
        probes_m=probes,
        run_id=getattr(args, "online_observables_run_id", None),
        layer_thickness_m=getattr(args, "layer_thickness", 40.0e-6),
        all_depths=getattr(args, "online_observables_all_depths", False),
    )
