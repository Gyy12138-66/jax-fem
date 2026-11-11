#!/usr/bin/env python3
"""E8 / E9 mesh-scaling configs (2026-09-23): the E0j (pyamg-GPU, frozen, tjit) and E1j (hybrid PARDISO, tjit)
configs with ONLY the mesh-dependent keys changed, so the solver arms are identical to the paper baseline.

Mesh-dependent keys, each re-derived with the rule that produced the 0.5 mm values:
  mesh.inp / raft_height_m / cell_height_m        from mesh/out_voxel_<s>/..._report.json
  layer_schedule.activation_slab_thickness_m      = cell height
  layer_schedule.physical_layers_per_slab         = 5 flashes per slab (energy per flash per cell mass =
                                                    validated 0.2 mm reading -> real layers per flash = h / 0.2 mm)
  runner.source_depth_cutoff_m                    = h - 1 um
  runner.source_depth_m                           d with h/2 = d (1 - exp(-cutoff/d)), i.e. discrete vertex-rule
                                                    band integral 1 (reproduces 125.935 um @ 0.2 mm, 314.182 um @ 0.5 mm)

    python make_scaling_configs.py        # writes inputs/0119-flash-voxel-<s>mm-fast-{pyamg-frozen,hybrid}-tjit.json
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from scipy.optimize import brentq

CASE = Path(__file__).resolve().parents[1]
INPUTS = CASE / "inputs"
BASES = {
    "pyamg-frozen-tjit": INPUTS / "0119-flash-voxel-fast-pyamg-frozen-tjit.json",   # E0j
    "hybrid-tjit": INPUTS / "0119-flash-voxel-fast-hybrid-tjit.json",               # E1j
}
MESHES = {"1p0": 1.0e-3, "0p4": 4.0e-4}
FLASHES_PER_SLAB = 5


def depth_for(slab: float) -> tuple[float, float]:
    cutoff = slab - 1e-6
    d = brentq(lambda d: 0.5 * slab - d * (1.0 - math.exp(-cutoff / d)), 0.1 * slab, 10 * slab, xtol=1e-18)
    return d, cutoff


def main() -> None:
    for tag, h in MESHES.items():
        rel = f"cases/159_simulation/mesh/out_voxel_{tag}/0119_hm_voxel_{tag}mm"
        rep = json.loads((CASE.parents[1] / f"{rel}_report.json").read_text())
        if not math.isclose(rep["voxel_m"], h, rel_tol=1e-9):
            raise SystemExit(f"{tag}: report voxel {rep['voxel_m']} != {h}")
        d, cutoff = depth_for(h)
        real_per_slab = h / 4e-5
        for lane, base_path in BASES.items():
            cfg = copy.deepcopy(json.loads(base_path.read_text()))
            mm = f"{h * 1e3:.1f} mm"
            cfg["scope"] = cfg["scope"].replace("0.5 mm voxel", f"{mm} voxel") + f" -- E8/E9 mesh scaling ({mm})"
            cfg["purpose"] = (f"E8/E9 mesh-scaling arm: {base_path.name} (the paper baseline lane) with only the mesh-dependent keys "
                              f"re-derived for the {mm} HyperMesh voxel mesh (model/make_scaling_configs.py). " + cfg["purpose"])
            cfg["provenance"]["scaling_base"] = f"cases/159_simulation/inputs/{base_path.name}"
            m = cfg["mesh"]
            m["inp"] = f"{rel}.inp"
            m["raft_height_m"] = round(rep["raft_height_m"], 9)
            m["cell_height_m"] = h
            m["note"] = (f"HyperMesh voxel lattice {mm} (mode 3, hm/voxel_sizes.tcl) filtered to centre-inside voxels "
                         f"(convert_hm_voxel.py), rotated to +Z, raft = {round(rep['raft_height_m'] / h)} voxel layer(s); "
                         f"{rep['part_cells']} part + {rep['raft_cells']} raft cells, {rep['mechanics_dof']} mechanics DOF")
            ls = cfg["layer_schedule"]
            ls["activation_slab_thickness_m"] = h
            ls["physical_layers_per_slab"] = FLASHES_PER_SLAB
            ls["activation_rule"] = (f"activate a {mm} slab when its first flash starts; {FLASHES_PER_SLAB} flashes per slab, each carrying "
                                     f"{real_per_slab / FLASHES_PER_SLAB:g} real (40 um) layers of energy and clock ({real_per_slab:g} real layers per slab)")
            r = cfg["runner"]
            r["source_depth_m"] = d
            r["source_depth_cutoff_m"] = cutoff
            r["heat_source_note"] = (f"depth band re-derived for {mm} cells under the vertex rule: cutoff {cutoff * 1e6:.0f} um (< slab), "
                                     f"d = {d * 1e6:.3f} um, discrete band integral 1.000000")
            out = INPUTS / f"0119-flash-voxel-{tag}mm-fast-{lane}.json"
            out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"wrote {out.name}: slab {mm}, d {d * 1e6:.3f} um, cutoff {cutoff * 1e6:.0f} um, "
                  f"{real_per_slab / FLASHES_PER_SLAB:g} real layers/flash, {rep['mechanics_dof']} DOF")


if __name__ == "__main__":
    main()
