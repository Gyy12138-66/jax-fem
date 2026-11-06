"""Device gather of the v06 thermal energy-ledger terms (cudss-plus plan C, 2026-11-06).

``extract_solver_step`` copies ~15 full quadrature arrays (204,401 x 8 on 0119) plus JxW, the quadrature
points and two dof->quad interpolations to the host on every thermal step (~270 MB, ~180 ms per step, 2,180 s
of a 3.8 h build) to produce ~20 scalars. This module evaluates the SAME formulas on the device in one jitted
call and copies back one float64 vector:

* the quadrature arrays, the nodal solution, ``fe.convert_from_dof_to_quad`` of the new / previous solution and
  ``problem.compute_residual`` are the very arrays the NumPy gather converts (same JAX producers);
* volume / surface integrals, raw-residual sums, temperature bounds and the state-override integral mirror
  ``integrate_volume_terms`` / ``integrate_surface_exchange`` / ``check_temperature_invariants`` line by line,
  written separately from the solver's weak-form kernels, so the assembly identity stays an independent check;
* every input check of the NumPy gather is evaluated as a violation count; any violation, any non-finite
  output or any scalar outside its domain sends that step to the original ``extract_solver_step`` (which then
  raises its own error), so the fast path never accepts what the reference would reject;
* the balance, the tolerances and the row come from the shared ``thermal_ledger._assemble_row``: the row schema
  and every criterion are unchanged.

Only summation order differs from NumPy (~1e-16 x sqrt(N) relative). ``JAX_FEM_LEDGER_SHADOW=N`` (default 500)
re-runs the NumPy gather on the first three steps and every N-th step; flags, counts and labels must agree exactly
(otherwise the NumPy row is kept and the step is counted as a mismatch), float differences are recorded in
``thermal_energy_ledger_device.json`` beside the ledger.

Switch: ``JAX_FEM_LEDGER_DEVICE=1`` (default off).
"""

from __future__ import annotations

import atexit
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from . import thermal_ledger as tl

_RTOL_U, _ATOL_U = 1.0e-12, 1.0e-15        # thermal_ledger._uniform_scalar (np.allclose)
_RELATIVE_FIELDS = ("relative_balance_error", "source_capture_fraction")


def enabled():
    return os.environ.get("JAX_FEM_LEDGER_DEVICE", "") == "1"


def shadow_every():
    try:
        return int(os.environ.get("JAX_FEM_LEDGER_SHADOW", "500") or 0)
    except ValueError:
        return 500


# --------------------------------------------------------------------------- kernel
_KERNEL = {}


def _kernel(cfg):
    """(jitted fn, holder): holder["names"] is the order of the returned vector, recorded while tracing."""
    hit = _KERNEL.get(cfg)
    if hit is not None:
        return hit
    import jax
    import jax.numpy as jnp

    (latent_branch, source_model, cutoff_on, renorm, front_on, front_rad, has_env, has_prev,
     surf_cfg, build_axis, p0, p1) = cfg
    holder = {}

    def cnt(mask):
        return jnp.sum(mask.astype(jnp.float64))

    def nonuniform(x):
        v = x.reshape(-1)
        v0 = v[0]
        return cnt(~jnp.isfinite(v)) + cnt(~(jnp.abs(v - v0) <= _ATOL_U + _RTOL_U * jnp.abs(v0))), v0

    def binary_bad(x):
        return cnt(~((x == 0.0) | (x == 1.0)))

    def density(old, new, rho, cp, latent_cp, active, cooling, solidus, liquidus, latent_heat):
        dT = new - old
        if latent_branch:
            interval = liquidus - solidus
            f_old = jnp.clip((old - solidus) / interval, 0.0, 1.0)
            f_new = jnp.clip((new - solidus) / interval, 0.0, 1.0)
            support = jnp.maximum(active, cooling)
            return rho * (cp * dT + latent_heat * (f_new - f_old) * support)
        return rho * (cp + latent_cp) * dT

    def body(a, s):
        out = {}
        old, new = a["T_old"], a["T_new_q"]
        rho, cp, latent_cp, cond = a["rho"], a["cp"], a["latent_cp"], a["k"]
        active, cooling, jxw, pts = a["active"], a["cooling"], a["jxw"], a["pts"]
        sc = {}
        for name in ("dt", "power", "radius", "depth", "switch", "h_old") + (("env_amb",) if has_env else ()):
            out["nu_" + name], sc[name] = nonuniform(a[name])
            out[name] = sc[name]
        amb = sc["env_amb"] if has_env else s["problem_ambient"]
        support = rho > 0.0
        # extract_solver_step: conductivity contract
        out["bad_k"] = (cnt(~jnp.isfinite(cond)) + cnt(cond < 0.0) + cnt(support & (cond <= 0.0))
                        + cnt((~support) & (cond != 0.0)))
        # integrate_volume_terms: array contracts
        out["bad_jxw"] = cnt(~jnp.isfinite(jxw)) + cnt(jxw <= 0.0)
        out["bad_pts"] = cnt(~jnp.isfinite(pts))
        out["bad_fields"] = sum(cnt(~jnp.isfinite(x)) for x in (old, new, rho, cp, latent_cp, active, cooling))
        out["bad_neg"] = cnt(rho < 0.0) + cnt(cp < 0.0) + cnt(latent_cp < 0.0)
        out["bad_binary"] = binary_bad(active) + binary_bad(cooling)
        out["bad_support"] = (cnt(support & (cp <= 0.0)) + cnt((~support) & ((cp != 0.0) | (latent_cp != 0.0)))
                              + cnt(((active > 0.5) | (cooling > 0.5)) & (~support)))
        lc = a["laser_center"][0, 0]
        out["bad_lc"] = cnt(~jnp.isfinite(lc))
        solidus, liquidus, latent_heat = s["solidus"], s["liquidus"], s["latent_heat"]
        out["storage_j"] = jnp.sum(jxw * density(old, new, rho, cp, latent_cp, active, cooling, solidus,
                                                 liquidus, latent_heat))
        r0 = pts[..., p0] - lc[p0]
        r1 = pts[..., p1] - lc[p1]
        depth = s["build_sign"] * (lc[build_axis] - pts[..., build_axis])
        if source_model == "paper_hemispherical":
            radius = sc["radius"]
            q_shape = jnp.where(depth >= 0.0, jnp.exp(-3.0 * (r0 ** 2 + r1 ** 2 + depth ** 2) / radius ** 2), 0.0)
            q_laser = (6.0 * jnp.sqrt(3.0) * sc["power"] / (jnp.pi * jnp.sqrt(jnp.pi) * radius ** 3) * q_shape
                       * sc["switch"] * active)
        else:
            q_depth = jnp.where(depth >= 0.0, jnp.exp(-depth / sc["depth"]), 0.0)
            if cutoff_on:
                q_depth = jnp.where(depth <= s["cutoff"], q_depth, 0.0)
                if renorm:
                    q_depth = q_depth / (-jnp.expm1(-s["cutoff"] / sc["depth"]))
            q_laser = (2.0 * sc["power"] / (jnp.pi * sc["radius"] ** 2 * sc["depth"])
                       * jnp.exp(-2.0 * (r0 ** 2 + r1 ** 2) / sc["radius"] ** 2) * q_depth * sc["switch"] * active)
        if front_on:
            band = jnp.where(depth >= 0.0, jnp.exp(-(depth / s["front_thick"]) ** 2), 0.0) * active
            q_front = s["front_h"] / s["front_thick"] * (old - amb) * band
            if front_rad:
                q_front = q_front + (s["emissivity"] * s["sigma"] / s["front_thick"] * (old ** 4 - amb ** 4)
                                     * band)
        else:
            q_front = jnp.zeros_like(old)
        q_old = sc["h_old"] * cooling * (old - amb)
        dt = sc["dt"]
        out["laser_deposited_j"] = dt * jnp.sum(jxw * q_laser)
        out["front_loss_j"] = dt * jnp.sum(jxw * q_front)
        out["old_layer_loss_j"] = dt * jnp.sum(jxw * q_old)
        # surfaces (_surface_exchange_from_problem / integrate_surface_exchange)
        T_nodal = a["T_new"]
        total = jnp.zeros((), jnp.float64)
        bad_s = jnp.zeros((), jnp.float64)
        nu_s = jnp.zeros((), jnp.float64)
        for k, has_amb in enumerate(surf_cfg):
            sf = a["surf"][k]
            nodal = T_nodal[a["cells"][sf["owner"]]]
            tf = jnp.einsum("fn,fqn->fq", nodal, sf["shape_vals"])
            sj, sa = sf["jxw"], sf["active"]
            bad_s = (bad_s + cnt(~jnp.isfinite(tf)) + cnt(~jnp.isfinite(sj)) + cnt(sj < 0.0)
                     + cnt(~jnp.isfinite(sa)) + binary_bad(sa))
            if has_amb:
                n_bad, s_amb = nonuniform(sf["ambient"])
                nu_s = nu_s + n_bad
            else:
                s_amb = s["problem_ambient"]
            outward = sa * (s["conv_h"] * (tf - s_amb) + s["emissivity"] * s["sigma"] * (tf ** 4 - s_amb ** 4))
            total = total + dt * jnp.sum(sj * outward)
        out["surface_loss_j"] = total
        out["bad_surf"] = bad_s
        out["nu_surf"] = nu_s
        # raw residual
        raw, cons = a["raw"], a["constrained"]
        out["bad_raw"] = cnt(~jnp.isfinite(raw))
        free = jnp.where(cons, 0.0, raw)
        out["dirichlet_exchange_j"] = dt * jnp.sum(jnp.where(cons, raw, 0.0))
        out["free_signed_j"] = dt * jnp.sum(free)
        out["residual_total_j"] = dt * jnp.sum(raw)
        out["free_l1_j"] = dt * jnp.sum(jnp.abs(free))
        out["free_l2_j"] = dt * jnp.linalg.norm(free)
        # check_temperature_invariants (T_old = quadrature old temperature, T_new = nodal solution)
        out["bad_T_old"] = cnt(~jnp.isfinite(old))
        finite_new = jnp.isfinite(T_nodal)
        out["nonfinite_new"] = cnt(~finite_new)
        lower = jnp.minimum(jnp.minimum(jnp.min(old), s["dir_min"]), amb)
        upper = jnp.maximum(jnp.maximum(jnp.max(old), s["dir_max"]), amb)
        out["lower"], out["upper"], out["amb"] = lower, upper, amb
        out["lower_count"] = cnt(finite_new & (T_nodal < lower - s["atol"]))
        out["upper_count"] = cnt(finite_new & (T_nodal > upper + s["atol"]))
        if has_prev:
            out["state_override"] = jnp.sum(jxw * density(a["prev_q"], old, rho, cp, latent_cp, active, cooling,
                                                          solidus, liquidus, latent_heat))
            out["bad_prev"] = cnt(~jnp.isfinite(a["prev_q"]))
        names = sorted(out)
        holder["names"] = names
        return jnp.stack([jnp.asarray(out[n], jnp.float64) for n in names])

    hit = (jax.jit(body), holder)
    _KERNEL[cfg] = hit
    return hit


# --------------------------------------------------------------------------- host side
class _State:
    def __init__(self):
        self.cache = {}                # key -> (source object, device array); the source keeps id() unique
        self.bc = None
        self.stats = dict(device_rows=0, fallback_rows=0, shadow_rows=0, shadow_mismatch=0, max_abs_diff_rel={},
                          max_rel_diff={}, fallback_reasons={}, device_seconds=0.0, shadow_seconds=0.0)
        self.path = None
        self.last_write = 0.0


_S = _State()


def _dev(x, key=None):
    """device float64 view of x; host arrays are uploaded once per object"""
    import jax.numpy as jnp
    if hasattr(x, "devices"):
        return x if x.dtype == jnp.float64 else x.astype(jnp.float64)
    k = (key, id(x))
    hit = _S.cache.get(k)
    if hit is None or hit[0] is not x:
        hit = (x, jnp.asarray(np.asarray(x, dtype=np.float64)))
        _S.cache[k] = hit
    return hit[1]


def _to_dev(x):
    """per-step arrays (never cached: host temporaries would pile up in the cache)"""
    import jax.numpy as jnp
    if hasattr(x, "devices"):
        return x if x.dtype == jnp.float64 else x.astype(jnp.float64)
    return jnp.asarray(np.asarray(x, dtype=np.float64))


def _dev_int(x, key):
    import jax.numpy as jnp
    if hasattr(x, "devices"):
        return x
    k = (key, id(x))
    hit = _S.cache.get(k)
    if hit is None or hit[0] is not x:
        hit = (x, jnp.asarray(np.asarray(x, dtype=np.int64)))
        _S.cache[k] = hit
    return hit[1]


def _dirichlet(fe, num_nodes):
    """(sig, constrained (device bool), dir_min, dir_max, free count, values finite); once per BC version"""
    import jax.numpy as jnp
    lists = (getattr(fe, "node_inds_list", []), getattr(fe, "vec_inds_list", []), getattr(fe, "vals_list", []))
    sig = (num_nodes, tuple((id(a), id(b), id(c), int(np.size(a)), int(np.size(c))) for a, b, c in zip(*lists)))
    if _S.bc is not None and _S.bc[0] == sig:
        return _S.bc
    mask, values = tl._dirichlet_mask_and_values(fe, num_nodes)      # raises exactly as the NumPy gather
    ok = bool(np.all(np.isfinite(values)))
    dmin = float(values.min()) if values.size else math.inf
    dmax = float(values.max()) if values.size else -math.inf
    _S.bc = (sig, jnp.asarray(mask), dmin, dmax, int(np.count_nonzero(~mask)), ok, lists)
    return _S.bc


def _scalars_ok(o, problem, source_model, cutoff, renorm, latent_heat, solidus, liquidus):
    vals = [o["dt"], o["power"], o["radius"], o["depth"], o["switch"], o["h_old"], problem.front_surface_loss_h,
            problem.front_surface_loss_thickness, problem.emissivity, problem.stefan_boltzmann, latent_heat,
            problem.build_sign, cutoff, problem.convection_h, o["amb"], solidus, liquidus]
    if not all(math.isfinite(float(v)) for v in vals):
        return "nonfinite scalar"
    if source_model not in ("legacy", "paper_hemispherical"):
        return "source model"
    if o["dt"] <= 0.0 or o["radius"] <= 0.0 or (source_model == "legacy" and o["depth"] <= 0.0):
        return "dt/radius/depth"
    if cutoff < 0.0 or (cutoff > 0.0 and source_model != "legacy") or (renorm and cutoff <= 0.0):
        return "cutoff"
    if o["power"] < 0.0 or latent_heat < 0.0 or not 0.0 <= o["switch"] <= 1.0:
        return "power/latent/switch"
    if o["h_old"] < 0.0 or problem.front_surface_loss_h < 0.0:
        return "loss coefficients"
    if problem.front_surface_loss_h > 0.0 and problem.front_surface_loss_thickness <= 0.0:
        return "front thickness"
    if not 0.0 <= problem.emissivity <= 1.0 or problem.stefan_boltzmann < 0.0 or problem.convection_h < 0.0:
        return "radiation/convection"
    if latent_heat > 0.0 and liquidus <= solidus:
        return "melt interval"
    return None


def stats():
    return _S.stats


def _write_stats():
    if _S.path is None:
        return
    try:
        _S.path.write_text(json.dumps(_S.stats, indent=1, default=str), encoding="utf-8")
    except OSError:
        pass


atexit.register(_write_stats)


def compare_rows(dev_row, ref_row, record=True):
    """(largest float difference / balance scale, [mismatching exact fields]).

    Flags, counts, labels and None must be identical; floats are compared relative to the step's balance scale
    (relative_balance_error / source_capture_fraction relative to themselves)."""
    scale = max(abs(float(ref_row.get("balance_scale_j") or 0.0)), 1e-300)
    worst = [0.0]
    mismatch = []

    def walk(a, b, key):
        if isinstance(b, dict):
            if not isinstance(a, dict) or set(a) != set(b):
                mismatch.append(key)
                return
            for k in b:
                walk(a[k], b[k], key + "." + k)
            return
        if isinstance(b, float) and isinstance(a, float):
            name = key.rsplit(".", 1)[-1]
            if name in _RELATIVE_FIELDS:
                d = abs(a - b) / max(abs(b), 1e-300)
                table = "max_rel_diff"
            else:
                d = abs(a - b) / scale
                table = "max_abs_diff_rel"
                worst[0] = max(worst[0], d)
            if record:
                _S.stats[table][name] = max(_S.stats[table].get(name, 0.0), d)
            return
        if a != b or type(a) is not type(b):
            mismatch.append(key)

    walk(dev_row, ref_row, "row")
    return worst[0], mismatch


def extract_solver_step_device(problem, solution, *, step_index, step_state, absorptivity, previous_solution,
                               temperature_atol_k, solver_residual_tolerance_w=1.0e-6,
                               relative_balance_tolerance=1.0e-6, output_dir=None):
    """Drop-in for ``thermal_ledger.extract_solver_step`` (same arguments, same row)."""
    kw = dict(step_index=step_index, step_state=step_state, absorptivity=absorptivity,
              previous_solution=previous_solution, temperature_atol_k=temperature_atol_k,
              solver_residual_tolerance_w=solver_residual_tolerance_w,
              relative_balance_tolerance=relative_balance_tolerance)
    if output_dir is not None and _S.path is None:
        _S.path = Path(output_dir) / "thermal_energy_ledger_device.json"
    t0 = time.perf_counter()
    row, reason = _device_row(problem, solution, **kw)
    _S.stats["device_seconds"] += time.perf_counter() - t0
    if row is None:
        _S.stats["fallback_rows"] += 1
        _S.stats["fallback_reasons"][reason] = _S.stats["fallback_reasons"].get(reason, 0) + 1
        _write_stats()
        return tl.extract_solver_step(problem, solution, **kw)
    _S.stats["device_rows"] += 1
    every = shadow_every()
    if every > 0 and (step_index < 3 or step_index % every == 0):
        t1 = time.perf_counter()
        ref = tl.extract_solver_step(problem, solution, **kw)
        _S.stats["shadow_seconds"] += time.perf_counter() - t1
        _S.stats["shadow_rows"] += 1
        _, mism = compare_rows(row, ref)
        if mism:
            _S.stats["shadow_mismatch"] += 1
            ex = _S.stats.setdefault("mismatch_examples", [])
            ex.append(dict(step=int(step_index), fields=mism[:10]))
            del ex[:-20]
            row = ref                              # the reference decides when they disagree
    if time.perf_counter() - _S.last_write > 60.0:
        _S.last_write = time.perf_counter()
        _write_stats()
    return row


def _device_row(problem, solution, *, step_index, step_state, absorptivity, previous_solution, temperature_atol_k,
                solver_residual_tolerance_w, relative_balance_tolerance):
    """(row, None) or (None, reason): reason sends the step to the NumPy gather."""
    import jax.numpy as jnp
    if not isinstance(solution, (list, tuple)) or len(solution) != 1:
        return None, "solution contract"
    T_new = solution[0]
    if getattr(T_new, "ndim", 0) != 2 or T_new.shape[1] != 1:
        return None, "solution shape"
    internal = list(problem.internal_vars)
    if len(internal) not in (14, 15):
        return None, "internal contract"
    fe = problem.fes[0]
    jxw = _dev(fe.JxW, "jxw")
    if jxw.ndim != 2:
        return None, "jxw shape"
    shape = tuple(jxw.shape)
    pts = _dev(problem.physical_quad_points, "pts")
    if tuple(pts.shape) != shape + (3,):
        return None, "points shape"
    f = [_to_dev(x) for x in internal]
    a = {}
    for name, idx in (("T_old", 0), ("active", 7), ("rho", 8), ("cp", 9), ("k", 10), ("latent_cp", 11),
                      ("cooling", 12)):
        x = f[idx]
        if x.ndim != 3 or tuple(x.shape) != shape + (1,):
            return None, "field shape %s" % name
        a[name] = x[..., 0]
    if f[2].ndim != 3 or tuple(f[2].shape[:2]) != shape or f[2].shape[2] != 3:
        return None, "laser center shape"
    a.update(jxw=jxw, pts=pts, laser_center=f[2], dt=f[1], power=f[3], radius=f[4], depth=f[5], switch=f[6],
             h_old=f[13])
    has_env = len(internal) == 15
    if has_env:
        a["env_amb"] = f[14]
    T_new_d = _to_dev(T_new)
    T_new_q = _to_dev(fe.convert_from_dof_to_quad(T_new))
    if tuple(T_new_q.shape) != shape + (1,):
        return None, "T_new quad shape"
    a["T_new_q"] = T_new_q[..., 0]
    a["T_new"] = T_new_d[:, 0]
    num_nodes = int(T_new.shape[0])
    raw = _to_dev(problem.compute_residual([solution[0]])[0]).reshape(-1)
    if raw.shape[0] != num_nodes:
        return None, "residual length"
    a["raw"] = raw
    try:
        bc = _dirichlet(fe, num_nodes)
    except ValueError:
        return None, "dirichlet contract"
    a["constrained"] = bc[1]
    has_prev = previous_solution is not None
    if has_prev:
        pq = _to_dev(fe.convert_from_dof_to_quad(previous_solution))
        if tuple(pq.shape) != shape + (1,):
            return None, "previous quad shape"
        a["prev_q"] = pq[..., 0]
    surf_cfg = []
    a["surf"] = []
    if any(len(b) for b in problem.boundary_inds_list):
        a["cells"] = _dev_int(fe.cells, "cells")
    for k, binds in enumerate(problem.boundary_inds_list):
        if not len(binds):
            continue
        vars_k = problem.internal_vars_surfaces[k]
        if not vars_k:
            return None, "surface active mask missing"
        hit = _S.cache.get(("owner", id(binds)))
        if hit is None or hit[0] is not binds:
            hit = (binds, jnp.asarray(np.asarray(binds, dtype=np.int64)[:, 0]))
            _S.cache[("owner", id(binds))] = hit
        sv = _dev(problem.selected_face_shape_vals[k], "sv%d" % k)
        ns = _dev(problem.nanson_scale[k], "ns%d" % k)
        if ns.ndim != 3 or sv.ndim != 3:
            return None, "surface shapes"
        s = dict(owner=hit[1], shape_vals=sv, jxw=ns[:, 0, :], active=_to_dev(vars_k[0])[..., 0])
        nf_k = int(hit[1].shape[0])
        if (sv.shape[0] != nf_k or tuple(s["jxw"].shape) != tuple(sv.shape[:2])
                or tuple(s["active"].shape) != tuple(sv.shape[:2])):
            return None, "surface shapes"
        has_amb = len(vars_k) > 1
        if has_amb:
            s["ambient"] = _to_dev(vars_k[1])
        surf_cfg.append(bool(has_amb))
        a["surf"].append(s)
    source_model = getattr(problem, "source_model", "legacy")
    cutoff = float(getattr(problem, "source_depth_cutoff", 0.0))
    renorm = bool(getattr(problem, "source_cutoff_renormalize", False))
    latent_heat = float(getattr(problem, "latent_heat", 0.0))
    solidus = getattr(problem, "solidus_temperature", 0.0)
    liquidus = getattr(problem, "liquidus_temperature", 0.0)
    solidus = 0.0 if solidus is None else float(solidus)
    liquidus = 0.0 if liquidus is None else float(liquidus)
    plane = (int(problem.plane_axis0_id), int(problem.plane_axis1_id))
    build_axis = int(problem.build_axis_id)
    if sorted((*plane, build_axis)) != [0, 1, 2]:
        return None, "axes"
    cfg = (bool(latent_heat > 0.0 and liquidus > solidus), source_model, bool(cutoff > 0.0), renorm,
           bool(float(problem.front_surface_loss_h) > 0.0), bool(problem.front_surface_loss_radiation), has_env,
           has_prev, tuple(surf_cfg), build_axis, plane[0], plane[1])
    s = {k: jnp.asarray(float(v), jnp.float64) for k, v in dict(
        problem_ambient=float(problem.ambient), solidus=solidus, liquidus=liquidus, latent_heat=latent_heat,
        build_sign=float(problem.build_sign), cutoff=cutoff, front_h=float(problem.front_surface_loss_h),
        front_thick=float(problem.front_surface_loss_thickness), emissivity=float(problem.emissivity),
        sigma=float(problem.stefan_boltzmann), conv_h=float(problem.convection_h), dir_min=bc[2], dir_max=bc[3],
        atol=float(temperature_atol_k)).items()}
    fn, holder = _kernel(cfg)
    vec = np.asarray(fn(a, s))                     # the one device->host copy of the step
    o = dict(zip(holder["names"], (float(v) for v in vec)))
    bad = sorted(k for k in o if (k.startswith("bad_") or k.startswith("nu_")) and o[k] != 0)
    if bad:
        return None, "violation " + ",".join(bad)
    if not all(math.isfinite(v) for v in o.values()) or not bc[5]:
        return None, "nonfinite output"
    why = _scalars_ok(o, problem, source_model, cutoff, renorm, latent_heat, solidus, liquidus)
    if why:
        return None, why
    if not math.isfinite(float(temperature_atol_k)) or temperature_atol_k < 0.0 or num_nodes == 0:
        return None, "invariant inputs"
    volume = {k: o[k] for k in ("storage_j", "laser_deposited_j", "front_loss_j", "old_layer_loss_j")}
    if volume["laser_deposited_j"] < 0.0:
        return None, "negative deposited source"
    residual_stats = {k: o[k] for k in ("dirichlet_exchange_j", "free_signed_j", "residual_total_j", "free_l1_j",
                                        "free_l2_j")}
    residual_stats["free_node_count"] = bc[4]
    lower, upper, atol = o["lower"], o["upper"], float(temperature_atol_k)
    deposited = volume["laser_deposited_j"]
    all_finite = o["nonfinite_new"] == 0

    def invariants():
        source_free = deposited <= np.finfo(np.float64).eps
        lower_count = int(o["lower_count"])
        upper_count = int(o["upper_count"]) if source_free else None
        valid = bool(True and all_finite and lower_count == 0 and (upper_count in (None, 0)))
        return {
            "claim_level": "physical_temperature_invariant_diagnostic",
            "valid": valid,
            "coefficient_preconditions_valid": True,
            "all_new_temperatures_finite": bool(all_finite),
            "source_free": source_free,
            "lower_bound_k": lower,
            "upper_bound_k": upper if source_free else None,
            "lower_violation_count": lower_count,
            "upper_violation_count": upper_count,
            "atol_k": atol,
        }

    def state_override():
        return o["state_override"] if has_prev else None

    try:
        row = tl._assemble_row(volume=volume, surface_loss=o["surface_loss_j"], residual_stats=residual_stats,
                               dt_s=o["dt"], effective_power=o["power"], laser_switch=o["switch"],
                               absorptivity=absorptivity, step_state=step_state, step_index=step_index,
                               solver_residual_tolerance_w=solver_residual_tolerance_w,
                               relative_balance_tolerance=relative_balance_tolerance, invariants=invariants,
                               state_override=state_override)
    except ValueError as exc:
        return None, "assemble: %s" % str(exc)[:60]
    return row, None
