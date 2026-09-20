# coding: utf-8
"""Per-physics linear-solver registry for the jax_fem_am runner.

Every Newton solve in the stepper (thermal increment, build-step mechanics,
raft-release mechanics) ends in ``jax_fem.solver.linear_solver``, which picks
its backend from the single-key ``linear`` block of the Newton options::

    {"jax_solver": {...}} | {"custom_solver": callable} | {"spsolve_solver": {}}
    | {"petsc_solver": {...}} | {"amgx_solver": {...}}

This module is the one place that turns a human-facing *spec* into such a
block, so the thermal and the mechanics problems are configured
independently (``--thermal-linear-solver`` / ``--mechanics-linear-solver`` on
the CLI, ``runner.linear_solver.thermal`` / ``.mechanics`` in a case JSON).

Spec forms accepted by :func:`parse_linear_solver_spec`::

    None | "" | "keep"                     -> None: keep the block the stepper wrote
    "pardiso" | "spsolve"                  -> backend only
    "jax:cg:jacobi" | "jax:bicgstab:none"  -> backend:method:precond shorthand
    "pardiso:phase23"                      -> pardiso:mode shorthand
    "pyamg:cg:rigid_body"                  -> pyamg:method:near_nullspace shorthand
    '{"backend": "jax", "method": "cg", "precond": "jacobi", "tol": 1e-6}'
    {"backend": ..., ...}                  -> dict with the same keys as the JSON form

Backends and their keys:

    jax      method (cg|bicgstab|gmres|spsolve), precond (jacobi|none), tol,
             atol, maxiter, restart, solve_method, check_residual, check_factor
    pardiso  mode (base|nocmp|cache-idx|phase23|fp32ir)
    spsolve  (no keys)
    petsc    ksp_type, pc_type, gpu
    amgx     cfg_path
    pyamg    method (cg|bicgstab), near_nullspace (rigid_body|constant),
             scaled, tol, maxiter, max_coarse, rebuild (pattern|always),
             fallback (pardiso|none), device (cpu|gpu), smoother
             (jacobi|block_gauss_seidel), smoother_sweeps, smoother_omega
             -- see solvers/amg.py (cpu = pyamg reference, gpu = jax V-cycle)

Defaults (:data:`DEFAULT_SCOPED_SPECS`, selected by ``--xla-linear-solver
auto``): thermal -> jax CG + Jacobi (tol/atol 1e-6, relative post-solve
residual check on: true residual <= check_factor x max(tol*|b|, atol),
RuntimeError -> fallback), mechanics -> MKL PARDISO phase23. The v159 group-A/B diagnostics
(2026-09-03) are the evidence: the thermal system has Jacobi-scaled kappa
12-42 independent of build height, while the mechanics tangent has kappa
6e4-4e6 and every Jacobi-type Krylov solve stalls on it.

Custom-solver protocol (``custom_solver`` values): ``x = solver(A_petsc, b,
x0, linear_options)``; ``__deepcopy__`` must return ``self`` because the
wrapper deep-copies the Newton options on every solve; optional
``bind_problem(problem)`` (called before each solve) and
``stats_snapshot()`` (profile report) hooks; a ``label`` attribute for logs.

Pure-Python: importable without jax / petsc / pypardiso installed; the
pardiso and pyamg backends import their dependencies lazily.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Mapping, Optional

BACKENDS = ("jax", "pardiso", "spsolve", "petsc", "amgx", "pyamg")
SCOPES = ("thermal", "mechanics")

JAX_METHODS = ("cg", "bicgstab", "gmres", "spsolve")
JAX_SOLVE_METHODS = ("batched", "incremental")
PRECONDS = ("jacobi", "none")
PARDISO_MODES = ("base", "nocmp", "cache-idx", "phase23", "fp32ir")
PYAMG_METHODS = ("cg", "bicgstab")
PYAMG_NEAR_NULLSPACES = ("rigid_body", "constant")
PYAMG_REBUILD = ("pattern", "always")
FALLBACK_SOLVERS = ("spsolve", "pardiso", "none")

#: ``linear`` block keys that mean "iterative Krylov backend" -- the release
#: solve (``problem.prefer_direct_linear_solver``) is re-routed to PARDISO
#: whenever the active block carries one of these.
ITERATIVE_LINEAR_KEYS = frozenset(
    {
        "jax_solver",
        "petsc_solver",
        "amgx_solver",
        "cg_solver",
        "bicgstab_solver",
        "gmres_solver",
    }
)

#: What ``--xla-linear-solver auto`` means per physics.
DEFAULT_SCOPED_SPECS: Dict[str, Dict[str, Any]] = {
    "thermal": {
        "backend": "jax",
        "method": "cg",
        "precond": "jacobi",
        "tol": 1e-6,
        "atol": 1e-6,
        "maxiter": 10000,
        "check_residual": True,
    },
    "mechanics": {"backend": "pardiso", "mode": "phase23"},
}

_KEEP_WORDS = frozenset({"", "keep", "preserve", "none", "null", "base"})

_ALLOWED_KEYS: Dict[str, frozenset] = {
    "jax": frozenset(
        {"backend", "method", "precond", "tol", "atol", "maxiter", "restart",
         "solve_method", "check_residual", "check_factor"}
    ),
    "pardiso": frozenset({"backend", "mode"}),
    "spsolve": frozenset({"backend"}),
    "petsc": frozenset({"backend", "ksp_type", "pc_type", "gpu"}),
    "amgx": frozenset({"backend", "cfg_path"}),
    "pyamg": frozenset(
        {"backend", "method", "near_nullspace", "scaled", "tol", "maxiter",
         "max_coarse", "rebuild", "fallback", "verbose", "device", "smoother",
         "smoother_sweeps", "smoother_omega", "rebuild_iter_factor",
         "clear_jax_caches_on_pattern", "shape_mode", "fixed_levels", "bucket_ratio",
         "scale_policy"}
    ),
}

PYAMG_SHAPE_MODES = ("auto", "free", "full", "bucket")
PYAMG_SCALE_POLICIES = ("current", "frozen")


def _as_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
    raise ValueError(f"linear-solver spec key {key!r} must be a boolean, got {value!r}")


def _check_choice(value: Any, choices, key: str) -> str:
    text = str(value).strip().lower()
    if text not in choices:
        raise ValueError(
            f"linear-solver spec key {key!r} must be one of {list(choices)}, got {value!r}"
        )
    return text


def normalize_linear_solver_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and canonicalise a spec dict (lower-case choices, typed numbers).

    Raises ``ValueError`` on an unknown backend, an unknown key for that
    backend or a value outside the documented choices, so a typo in a case
    JSON fails at preflight instead of silently running the wrong solver.
    """
    if not isinstance(spec, Mapping):
        raise TypeError(f"linear-solver spec must be a mapping, got {type(spec).__name__}")
    out: Dict[str, Any] = dict(spec)
    backend = _check_choice(out.get("backend", ""), BACKENDS, "backend")
    out["backend"] = backend
    unknown = set(out) - _ALLOWED_KEYS[backend]
    if unknown:
        raise ValueError(
            f"linear-solver backend {backend!r} does not accept keys {sorted(unknown)}; "
            f"allowed: {sorted(_ALLOWED_KEYS[backend] - {'backend'})}"
        )
    if backend == "jax":
        out["method"] = _check_choice(out.get("method", "bicgstab"), JAX_METHODS, "method")
        precond = out.get("precond", "jacobi")
        if isinstance(precond, bool):
            precond = "jacobi" if precond else "none"
        out["precond"] = _check_choice(precond, PRECONDS, "precond")
        for key in ("tol", "atol"):
            if out.get(key) is not None:
                out[key] = float(out[key])
        for key in ("maxiter", "restart"):
            if out.get(key) is not None:
                out[key] = int(out[key])
        if "solve_method" in out and out["solve_method"] is not None:
            out["solve_method"] = _check_choice(
                out["solve_method"], JAX_SOLVE_METHODS, "solve_method"
            )
        if "check_residual" in out and out["check_residual"] is not None:
            out["check_residual"] = _as_bool(out["check_residual"], "check_residual")
        if out.get("check_factor") is not None:
            out["check_factor"] = float(out["check_factor"])
    elif backend == "pardiso":
        mode = out.get("mode")
        if mode is not None:
            out["mode"] = _check_choice(mode, PARDISO_MODES, "mode")
    elif backend == "petsc":
        out["ksp_type"] = str(out.get("ksp_type", "gmres"))
        out["pc_type"] = str(out.get("pc_type", "jacobi"))
        out["gpu"] = _as_bool(out.get("gpu", False), "gpu")
    elif backend == "amgx":
        if out.get("cfg_path") is not None:
            out["cfg_path"] = str(out["cfg_path"])
    elif backend == "pyamg":
        out["method"] = _check_choice(out.get("method", "cg"), PYAMG_METHODS, "method")
        out["near_nullspace"] = _check_choice(
            out.get("near_nullspace", "rigid_body"), PYAMG_NEAR_NULLSPACES, "near_nullspace"
        )
        out["scaled"] = _as_bool(out.get("scaled", True), "scaled")
        out["tol"] = float(out.get("tol", 1e-6))
        out["maxiter"] = int(out.get("maxiter", 500))
        # dofs of the coarsest level; 3000 keeps the v159 hierarchy at 3 levels
        # (level 2 tops out at 1746 dofs), 1000 produced a 4th level from layer
        # 71 on and tripled the CG iteration count (BUG_FIX.md section 7).
        out["max_coarse"] = int(out.get("max_coarse", 3000))
        out["rebuild"] = _check_choice(out.get("rebuild", "pattern"), PYAMG_REBUILD, "rebuild")
        out["fallback"] = _check_choice(out.get("fallback", "pardiso"), ("pardiso", "none"), "fallback")
        out["verbose"] = _as_bool(out.get("verbose", False), "verbose")
        out["clear_jax_caches_on_pattern"] = _as_bool(
            out.get("clear_jax_caches_on_pattern", False), "clear_jax_caches_on_pattern"
        )
        out["device"] = _check_choice(out.get("device", "cpu"), ("cpu", "gpu", "jax"), "device")
        out["smoother"] = _check_choice(
            out.get("smoother", "jacobi"), ("jacobi", "block_gauss_seidel"), "smoother"
        )
        out["smoother_sweeps"] = int(out.get("smoother_sweeps", 2))
        out["smoother_omega"] = float(out.get("smoother_omega", 4.0 / 3.0))
        out["rebuild_iter_factor"] = float(out.get("rebuild_iter_factor", 3.0))
        # Shape policy of the jit path: "auto" = full on gpu/jax, free on cpu.
        out["shape_mode"] = _check_choice(out.get("shape_mode", "auto"), PYAMG_SHAPE_MODES, "shape_mode")
        out["fixed_levels"] = int(out.get("fixed_levels", 4))
        out["bucket_ratio"] = float(out.get("bucket_ratio", 1.25))
        # Jacobi scaling while a hierarchy is reused: "frozen" keeps the scaling the
        # prolongator was built with (BUG_FIX.md section 11); "current" is the
        # behaviour of every run up to E0/E1b.
        out["scale_policy"] = _check_choice(
            out.get("scale_policy", "current"), PYAMG_SCALE_POLICIES, "scale_policy"
        )
    return out


def parse_linear_solver_spec(value: Any) -> Optional[Dict[str, Any]]:
    """Turn a CLI/JSON value into a normalised spec dict, or ``None`` for "keep"."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return normalize_linear_solver_spec(value)
    if not isinstance(value, str):
        raise TypeError(
            f"linear-solver spec must be a string or mapping, got {type(value).__name__}"
        )
    text = value.strip()
    if text.lower() in _KEEP_WORDS:
        return None
    if text.startswith("{"):
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"linear-solver JSON spec must be an object, got {text!r}")
        return normalize_linear_solver_spec(loaded)
    parts = [part.strip() for part in text.split(":")]
    backend = parts[0].lower()
    spec: Dict[str, Any] = {"backend": backend}
    extra = [part for part in parts[1:]]
    if backend == "jax":
        if len(extra) > 2:
            raise ValueError(f"jax shorthand is backend[:method[:precond]], got {text!r}")
        if extra and extra[0]:
            spec["method"] = extra[0]
        if len(extra) > 1 and extra[1]:
            spec["precond"] = extra[1]
    elif backend == "pardiso":
        if len(extra) > 1:
            raise ValueError(f"pardiso shorthand is pardiso[:mode], got {text!r}")
        if extra and extra[0]:
            spec["mode"] = extra[0]
    elif backend == "pyamg":
        if len(extra) > 2:
            raise ValueError(f"pyamg shorthand is pyamg[:method[:near_nullspace]], got {text!r}")
        if extra and extra[0]:
            spec["method"] = extra[0]
        if len(extra) > 1 and extra[1]:
            spec["near_nullspace"] = extra[1]
    elif extra:
        raise ValueError(f"backend {backend!r} takes no shorthand arguments, got {text!r}")
    return normalize_linear_solver_spec(spec)


# ---------------------------------------------------------------------------
# shared direct-solver instance
# ---------------------------------------------------------------------------
_SHARED_PARDISO: Dict[Optional[str], Any] = {}


def shared_pardiso_solver(mode: Optional[str] = None):
    """One PARDISO adapter per mode for the whole process.

    The adapter keeps its symbolic/numeric factorisations keyed by sparsity
    pattern, so sharing it between the mechanics block, the release re-route
    and the fallback path means the release solve reuses a live handle instead
    of paying a fresh analysis.
    """
    from jax_fem_am.solvers.pardiso import PardisoCustomSolver

    key = mode or None
    solver = _SHARED_PARDISO.get(key)
    if solver is None:
        solver = PardisoCustomSolver(key)
        _SHARED_PARDISO[key] = solver
    return solver


def reset_shared_solvers() -> None:
    _SHARED_PARDISO.clear()


def release_shared_solvers() -> int:
    """Release the factorisations held by every shared PARDISO adapter without
    forgetting the adapters (the next solve re-analyses). Returns the number
    of PARDISO handles released. Used after a solve that is known to be the
    last one on its pattern (the raft release) so an ~8 GB factorisation does
    not stay resident next to the iterative lane's working set."""
    return sum(solver.release() for solver in _SHARED_PARDISO.values())


# ---------------------------------------------------------------------------
# spec -> jax-fem ``linear`` block
# ---------------------------------------------------------------------------
def build_linear_block(
    spec: Optional[Mapping[str, Any]],
    *,
    pardiso_mode_default: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the single-key ``linear`` block for ``spec`` (``None`` -> ``None``)."""
    if spec is None:
        return None
    spec = (
        parse_linear_solver_spec(spec)
        if isinstance(spec, str)
        else normalize_linear_solver_spec(spec)
    )
    if spec is None:
        return None
    backend = spec["backend"]
    if backend == "jax":
        inner: Dict[str, Any] = {
            "precond": spec["precond"] == "jacobi",
            "method": spec["method"],
        }
        for key in ("tol", "atol", "maxiter", "restart", "solve_method"):
            if spec.get(key) is not None:
                inner[key] = spec[key]
        if spec.get("check_residual") is False:
            inner["check_residual"] = False
        if spec.get("check_factor") is not None:
            inner["check_factor"] = spec["check_factor"]
        return {"jax_solver": inner}
    if backend == "pardiso":
        mode = spec.get("mode", pardiso_mode_default)
        if mode == "base":
            mode = None
        return {"custom_solver": shared_pardiso_solver(mode)}
    if backend == "spsolve":
        return {"spsolve_solver": {}}
    if backend == "petsc":
        inner = {"ksp_type": spec["ksp_type"], "pc_type": spec["pc_type"]}
        if spec.get("gpu"):
            inner["mat_type"] = "aijcusparse"
            inner["vec_type"] = "cuda"
        return {"petsc_solver": inner}
    if backend == "amgx":
        inner = {"persistent_resources": True}
        if spec.get("cfg_path"):
            inner["cfg_path"] = spec["cfg_path"]
        return {"amgx_solver": inner}
    if backend == "pyamg":
        from jax_fem_am.solvers.amg import PyamgKrylovSolver

        kwargs = {key: value for key, value in spec.items() if key != "backend"}
        return {
            "custom_solver": PyamgKrylovSolver(
                pardiso_mode=pardiso_mode_default or "phase23", **kwargs
            )
        }
    raise ValueError(f"unknown linear-solver backend {backend!r}")


def build_fallback_linear_block(
    name: Optional[str],
    *,
    pardiso_mode_default: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """``--xla-fallback-solver`` value -> block used to retry a failed solve."""
    if name is None:
        return {"spsolve_solver": {}}
    choice = _check_choice(name, FALLBACK_SOLVERS, "fallback")
    if choice == "none":
        return None
    if choice == "spsolve":
        return {"spsolve_solver": {}}
    return {"custom_solver": shared_pardiso_solver(pardiso_mode_default or "phase23")}


# ---------------------------------------------------------------------------
# argparse namespace -> per-scope blocks
# ---------------------------------------------------------------------------
def scoped_specs_from_args(args: Any) -> Dict[str, Optional[Dict[str, Any]]]:
    """Resolve the thermal / mechanics specs from the CLI namespace.

    Precedence per scope: explicit ``--<scope>-linear-solver`` > ``--xla-linear-solver
    auto`` defaults > ``--mechanics-direct-solver`` (mechanics only, alias for
    ``pardiso``) > ``None`` (fall through to the global ``--xla-linear-solver``
    rewrite, or the stepper's own block when that is ``keep``).
    """
    choice = getattr(args, "xla_linear_solver", "keep") or "keep"
    pardiso_mode = getattr(args, "xla_pardiso_mode", None)
    specs: Dict[str, Optional[Dict[str, Any]]] = {}
    for scope in SCOPES:
        spec = parse_linear_solver_spec(getattr(args, f"{scope}_linear_solver", None))
        if spec is None and choice == "auto":
            spec = copy.deepcopy(DEFAULT_SCOPED_SPECS[scope])
        if (
            spec is None
            and scope == "mechanics"
            and getattr(args, "mechanics_direct_solver", False)
        ):
            spec = {"backend": "pardiso", "mode": pardiso_mode or "phase23"}
        specs[scope] = spec
    return specs


def scoped_linear_options_from_args(args: Any) -> Dict[str, Optional[Dict[str, Any]]]:
    pardiso_mode = getattr(args, "xla_pardiso_mode", None)
    return {
        scope: build_linear_block(spec, pardiso_mode_default=pardiso_mode)
        for scope, spec in scoped_specs_from_args(args).items()
    }


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def linear_block_label(linear_options: Optional[Mapping[str, Any]]) -> str:
    """Human-readable one-liner for a ``linear`` block (used in logs and profile meta)."""
    if linear_options is None:
        return "preserve original solver_options"
    if "jax_solver" in linear_options:
        opts = linear_options["jax_solver"]
        parts = [
            f"method={opts.get('method', 'bicgstab')}",
            f"precond={opts.get('precond', False)}",
        ]
        if "tol" in opts:
            parts.append(f"tol={opts['tol']}")
        if "atol" in opts:
            parts.append(f"atol={opts['atol']}")
        if "maxiter" in opts:
            parts.append(f"maxiter={opts['maxiter']}")
        if opts.get("check_residual") is False:
            parts.append("check_residual=False")
        return f"jax_solver({', '.join(parts)})"
    if "amgx_solver" in linear_options:
        cfg_path = linear_options["amgx_solver"].get("cfg_path")
        return f"amgx_solver(cfg_path={cfg_path or 'built-in'})"
    if "petsc_solver" in linear_options:
        opts = linear_options["petsc_solver"]
        return (
            f"petsc_solver(ksp_type={opts.get('ksp_type')}, "
            f"pc_type={opts.get('pc_type')})"
        )
    if "spsolve_solver" in linear_options:
        return "spsolve_solver(cpu scipy baseline)"
    if "custom_solver" in linear_options:
        custom = linear_options["custom_solver"]
        return getattr(custom, "label", f"custom_solver({custom!r})")
    return str(linear_options)


def same_linear_backend(
    left: Optional[Mapping[str, Any]], right: Optional[Mapping[str, Any]]
) -> bool:
    """True when two blocks would run the same backend (retrying is pointless)."""
    if not left or not right:
        return False
    left_key = next(iter(left))
    right_key = next(iter(right))
    if left_key != right_key:
        return False
    if left_key != "custom_solver":
        return True
    lc, rc = left[left_key], right[right_key]
    return lc is rc or getattr(lc, "label", None) == getattr(rc, "label", object())


def is_iterative_linear_block(block: Optional[Mapping[str, Any]]) -> bool:
    """True for Krylov backends (jax/petsc/amgx keys, or a custom solver that
    declares ``iterative = True`` such as the pyamg backend)."""
    if not block:
        return False
    if ITERATIVE_LINEAR_KEYS.intersection(block):
        return True
    return bool(getattr(block.get("custom_solver"), "iterative", False))


def json_safe_spec(spec: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None
    return {
        key: (value if isinstance(value, (str, int, float, bool, type(None))) else str(value))
        for key, value in spec.items()
    }
