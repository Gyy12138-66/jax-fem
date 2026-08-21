#!/usr/bin/env python3
"""Structured C3D8 mesh for the V1 single-track model (Balbaa2022 parity).

Geometry (inputs/balbaa-model.json single_track_model + D-V1-16):
  X (scan)  : 1.0 mm
  Y (width) : 0.48 mm  -- Balbaa states 1 mm x 240 um x 300 um WITH an XZ
              symmetry plane; 240 um is read as the modeled half-width
              (D-V1-16). We mesh the FULL width (2 x 240) and skip the
              symmetry BC: numerically equivalent for a centered track.
  Z (build) : 0.30 mm = 0.28 mm substrate + 0.02 mm powder layer
              (validation variant, Balbaa Sec 3.2: 20 um layer).
Mesh: uniform 10 um hexes (Balbaa: min 10 um in the powder layer,
substrate grading unspecified -> D-V1-14 uniform choice).
100 x 48 x 30 = 144,000 elements. Units: meters, build axis z.

Defaults reproduce the historical mesh byte-for-byte. The overrides exist
for the WP2 convergence ladder (--cell-size 20/10/5 um) and the V1-E
bare-plate domain sweep (--lx/--ly/--lz); every dimension must divide
evenly by the cell size.
"""
import argparse
from pathlib import Path

LAYER_THICKNESS = 20.0e-6

ap = argparse.ArgumentParser()
ap.add_argument("--cell-size", type=float, default=10.0e-6,
                help="uniform hex edge in metres (default 10 um)")
ap.add_argument("--lx", type=float, default=1.0e-3)
ap.add_argument("--ly", type=float, default=0.48e-3)
ap.add_argument("--lz", type=float, default=0.30e-3,
                help="total depth incl. the 20 um top layer")
ap.add_argument("--output", type=Path,
                default=Path(__file__).parent / "v1_single_track_c3d8.inp")
args = ap.parse_args()

DX = DY = DZ = args.cell_size


def _count(length, name):
    n = round(length / args.cell_size)
    if abs(n * args.cell_size - length) > 1e-12:
        raise SystemExit(
            f"{name}={length} is not an integer multiple of "
            f"cell size {args.cell_size}"
        )
    return n


NX, NY, NZ = _count(args.lx, "lx"), _count(args.ly, "ly"), _count(args.lz, "lz")
if round(LAYER_THICKNESS / args.cell_size) * args.cell_size != LAYER_THICKNESS:
    raise SystemExit(
        f"cell size {args.cell_size} does not tile the 20 um layer"
    )
LX, LY, LZ = NX * DX, NY * DY, NZ * DZ
substrate_top = LZ - LAYER_THICKNESS

out = args.output

def nid(i, j, k):
    return 1 + i + j * (NX + 1) + k * (NX + 1) * (NY + 1)

with open(out, "w", newline="\n") as f:
    f.write("*HEADING\n")
    f.write("V1 single-track Balbaa2022-parity structured C3D8 mesh "
            "(units: m, build axis z, 20um powder layer on 280um substrate)\n")
    f.write("*NODE\n")
    for k in range(NZ + 1):
        for j in range(NY + 1):
            for i in range(NX + 1):
                f.write(f"{nid(i,j,k)}, {i*DX:.10e}, {j*DY:.10e}, {k*DZ:.10e}\n")
    f.write("*ELEMENT, TYPE=C3D8\n")
    eid = 0
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                eid += 1
                n0 = nid(i, j, k)
                n1 = nid(i + 1, j, k)
                n2 = nid(i + 1, j + 1, k)
                n3 = nid(i, j + 1, k)
                f.write(f"{eid}, {n0}, {n1}, {n2}, {n3}, "
                        f"{n0 + (NX+1)*(NY+1)}, {n1 + (NX+1)*(NY+1)}, "
                        f"{n2 + (NX+1)*(NY+1)}, {n3 + (NX+1)*(NY+1)}\n")

print(f"wrote {out}: {(NX+1)*(NY+1)*(NZ+1)} nodes, {eid} elements")
print(f"domain: {LX*1e3:.2f} x {LY*1e3:.2f} x {LZ*1e3:.2f} mm")
print(f"substrate top z = {substrate_top*1e6:.0f} um; layer top z = {LZ*1e6:.0f} um")
