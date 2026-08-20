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
                               (D-V2-27) -- no conditional average, no n_hot

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
                 every, probes_m, layer_top_z_m=None, layer_thickness_m=40.0e-6,
                 all_depths=False, filename="online_observables.jsonl"):
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
        self.probes_m = [tuple(float(c) for c in p) for p in (probes_m or [])]
        self.layer_top_z_m = layer_top_z_m
        self.layer_thickness_m = float(layer_thickness_m)
        self.all_depths = bool(all_depths)

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

        probe_nodes = []
        for probe in self.probes_m:
            d2 = ((points[:, 0] - probe[0]) ** 2
                  + (points[:, 1] - probe[1]) ** 2
                  + (points[:, 2] - probe[2]) ** 2)
            idx = int(np.argmin(d2))
            probe_nodes.append(
                {"requested_m": list(probe),
                 "node_index": idx,
                 "actual_m": [float(v) for v in points[idx]],
                 "offset_m": float(math.sqrt(float(d2[idx])))})

        self._geometry = {
            "gauge_conn": cells[gauge],
            "probe_node_index": np.asarray(
                [p["node_index"] for p in probe_nodes], dtype=np.int64),
            "spot_center_m": [float(cx), float(cy)],
            "layer_bottom_z_m": float(layer_bottom),
            "gauge_cells": int(gauge.sum()),
            "gauge_cells_in_circle": int(in_circle.sum()),
            "probes": probe_nodes,
        }
        return self._geometry

    def _write_meta(self):
        geometry = self._geometry
        ok, self_check = uniform_field_self_check(self.wavelengths_m)
        meta = {
            "schema_version": self.SCHEMA,
            "claim_level": "solver_step_observable_extraction_only",
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
        """Called once per accepted thermal solve.  Never raises into the run."""
        if self.disabled_reason is not None:
            return
        try:
            self._observe(problem, temperature_new, step_state)
        except Exception as error:                       # observability only
            self.disabled_reason = f"{type(error).__name__}: {error}"
            print(f"online observables DISABLED after {self.rows_written} rows: "
                  f"{self.disabled_reason}")
            self._close()

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
            # D-V2-24 unthresholded diagnostic bound
            "full_spot_avg_K": float(cell_t.mean()),
        }
        if geometry["probe_node_index"].size:
            row["probe_K"] = [float(v) for v in
                              temperature[geometry["probe_node_index"]]]
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
            print(f"online observables: DISABLED ({self.disabled_reason}); "
                  f"{self.rows_written} rows survive in {self.path}")
            return
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
        layer_thickness_m=getattr(args, "layer_thickness", 40.0e-6),
        all_depths=getattr(args, "online_observables_all_depths", False),
    )
