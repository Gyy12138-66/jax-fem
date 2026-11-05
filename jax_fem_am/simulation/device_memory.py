"""Device-memory helpers for the coupled thermal / mechanics problems (cudss-plus, 2026-11-05).

The thermal and the mechanics problem are built on the same mesh, element type and quadrature, so their per-cell
geometry (``shape_grads``, ``JxW``, ``v_grads_JxW``, ``physical_quad_points``) is the same values computed twice:
about 0.6 GB of device memory on the 0119 part (E feasibility prior Q6, 2026-11-05). :func:`share_cell_geometry`
points the second problem at the first one's arrays where they are bitwise identical; arrays that differ (another
quadrature, a modified B) are left alone, so the results are unchanged. ``JAX_FEM_SHARE_GEOMETRY=0`` turns it off.
"""
from __future__ import annotations

import os

GEOMETRY_ATTRS = ("shape_grads", "JxW", "v_grads_JxW", "physical_quad_points")
FE_GEOMETRY_ATTRS = ("shape_grads", "JxW", "v_grads_JxW")


def share_enabled() -> bool:
    return os.environ.get("JAX_FEM_SHARE_GEOMETRY", "1").strip().lower() not in ("0", "false", "no", "off")


def _identical(a, b) -> bool:
    if a is b:
        return True
    if getattr(a, "shape", None) != getattr(b, "shape", None) or getattr(a, "dtype", None) != getattr(b, "dtype", None):
        return False
    import jax.numpy as jnp

    return bool(jnp.array_equal(a, b))


def share_cell_geometry(src, dst) -> dict:
    """Point ``dst``'s per-cell geometry arrays (problem and FE level) at ``src``'s where bitwise identical.

    Returns ``{name: bytes}`` of the arrays of ``dst`` that were replaced (each distinct buffer counted once, so
    the total is what can be freed once nothing else holds them).
    """
    if not share_enabled() or src is dst:
        return {}
    pairs = [(src, dst, name, name) for name in GEOMETRY_ATTRS]
    if len(src.fes) == len(dst.fes):
        for i, (fs, fd) in enumerate(zip(src.fes, dst.fes)):
            pairs += [(fs, fd, name, f"fes[{i}].{name}") for name in FE_GEOMETRY_ATTRS]
    shared, released = {}, set()
    for s, d, name, label in pairs:
        a, b = getattr(s, name, None), getattr(d, name, None)
        if a is None or b is None or a is b or not _identical(a, b):
            continue
        setattr(d, name, a)
        shared[label] = 0 if id(b) in released else int(getattr(b, "nbytes", 0))
        released.add(id(b))
    return shared
