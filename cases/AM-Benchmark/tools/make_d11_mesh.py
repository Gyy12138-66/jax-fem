"""D-11 sub-domain mesh generator (PREREQUISITES.md D.7, decision D-11).

The D-11 matrix runs on the D-07 periodic cell TRUNCATED to one leg-group
period, built to the overhang-band top:

  x : [0, 14] mm    one leg-group period, substrate included
  y : [-10, 10] mm  the full D-07 20 mm periodic cell (part |y| <= 2.5)
  z : [-12.7, 7.0]  substrate below 0; build to layer 350 = overhang top

Why x = [0, 14] is the period (feature_id_map, verified against the STL):
  L1 thick  x 0.0 - 5.0     L2 thin  x 7.0 - 7.5     L3 med  x 9.5 - 12.0
  L4 thick  x 14.0 - ...    <- the pattern repeats, so 14 mm is the pitch.
Both cut faces are benign: x = 0 is the part's REAL free end (L1 is the
leftmost feature) and x = 14 falls in the powder gap between L3 and L4
through the whole legs band. The solver has no side-face symmetry BC, so
both cut faces are traction-free / adiabatic-by-default -- a registered
truncation assumption, identical for all 15 matrix members, which is what
the relative M1/M2 metrics require.

Only the BUILD z-discretisation changes with N (that IS the lumping study);
the in-plane grid and the substrate grading are held fixed across N so the
metric differences are attributable to the lumping alone.

Output: derived/meshes/amb_d11_N<N>.inp (NOT committed; regenerable) +
derived/meshes/amb_d11_N<N>_summary.json (committed).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from make_d07_mesh import DZ_PHYS, axis, graded, in_part  # noqa: E402

X_MIN, X_MAX = 0.0, 14.0          # one leg-group period
Y_MIN, Y_MAX = -10.0, 10.0        # full D-07 periodic cell
Y_DEPTH = 5.0                     # part depth
Z_SUB = 12.7                      # substrate thickness (A.3/D-07)
Z_TOP_DEFAULT = 7.0               # overhang-band top = physical layer 350

DX_PART = 0.25                    # thin 0.5 mm leg gets 2 elements across
DY_PART = 0.50


def build_mesh(N, dx_part=DX_PART, dy_part=DY_PART, z_top=Z_TOP_DEFAULT,
               out_dir=None):
    n_phys = round(z_top / DZ_PHYS)
    if n_phys % N:
        raise SystemExit(f'z_top {z_top} mm = {n_phys} physical layers is '
                         f'not divisible by N={N}')
    dz_comp = N * DZ_PHYS
    n_comp = n_phys // N

    xs = axis([(X_MIN, [dx_part] * round((X_MAX - X_MIN) / dx_part))])
    wing = graded(7.5, dy_part * 1.6, 2.0)
    ys = axis([(Y_MIN, list(reversed(wing))),
               (-Y_DEPTH / 2, [dy_part] * round(Y_DEPTH / dy_part)),
               (Y_DEPTH / 2, wing)])
    zs = axis([(-Z_SUB, list(reversed(graded(Z_SUB, 0.4, 2.0)))),
               (0.0, [dz_comp] * n_comp)])

    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)

    def nid(i, j, k):
        return 1 + i + j * (nx + 1) + k * (nx + 1) * (ny + 1)

    out_dir = out_dir or os.path.join(CASE, 'derived', 'meshes')
    os.makedirs(out_dir, exist_ok=True)
    inp = os.path.join(out_dir, f'amb_d11_N{N}.inp')

    sets = {'SUBSTRATE': [], 'PART': [], 'POWDER': []}
    with open(inp, 'w') as f:
        f.write('*HEADING\n')
        f.write(f'AMB2018-01 D-11 sub-domain (one leg period), N={N}, '
                f'units mm, build axis z, Dirichlet face NSET=BOTTOM\n')
        f.write('*NODE\n')
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    f.write(f'{nid(i, j, k)}, {xs[i]:.6f}, '
                            f'{ys[j]:.6f}, {zs[k]:.6f}\n')
        f.write('*ELEMENT, TYPE=C3D8\n')
        eid = 0
        off = (nx + 1) * (ny + 1)
        for k in range(nz):
            zc = 0.5 * (zs[k] + zs[k + 1])
            layer = None
            if zc > 0:
                layer = min(n_phys, int(zc / DZ_PHYS) + 1)
            for j in range(ny):
                yc = 0.5 * (ys[j] + ys[j + 1])
                for i in range(nx):
                    xc = 0.5 * (xs[i] + xs[i + 1])
                    eid += 1
                    n0, n1 = nid(i, j, k), nid(i + 1, j, k)
                    n2, n3 = nid(i + 1, j + 1, k), nid(i, j + 1, k)
                    f.write(f'{eid}, {n0}, {n1}, {n2}, {n3}, '
                            f'{n0 + off}, {n1 + off}, {n2 + off}, {n3 + off}\n')
                    if zc < 0:
                        sets['SUBSTRATE'].append(eid)
                    elif in_part(xc, yc, layer):
                        sets['PART'].append(eid)
                    else:
                        sets['POWDER'].append(eid)
        for name, ids in sets.items():
            f.write(f'*ELSET, ELSET={name}\n')
            for a in range(0, len(ids), 10):
                f.write(', '.join(map(str, ids[a:a + 10])) + '\n')
        f.write('*NSET, NSET=BOTTOM\n')
        bottom = [nid(i, j, 0) for j in range(ny + 1) for i in range(nx + 1)]
        for a in range(0, len(bottom), 10):
            f.write(', '.join(map(str, bottom[a:a + 10])) + '\n')
        for name, ii in (('XMIN', 0), ('XMAX', nx)):
            f.write(f'*NSET, NSET={name}\n')
            side = [nid(ii, j, k) for k in range(nz + 1)
                    for j in range(ny + 1)]
            for a in range(0, len(side), 10):
                f.write(', '.join(map(str, side[a:a + 10])) + '\n')
        for name, jj in (('YMIN', 0), ('YMAX', ny)):
            f.write(f'*NSET, NSET={name}\n')
            side = [nid(i, jj, k) for k in range(nz + 1)
                    for i in range(nx + 1)]
            for a in range(0, len(side), 10):
                f.write(', '.join(map(str, side[a:a + 10])) + '\n')

    summary = {
        'schema_version': 'ambench.d11-mesh/1',
        'N': N, 'physical_layers': n_phys, 'computational_layers': n_comp,
        'layer_thickness_mm': dz_comp,
        'dx_part_mm': dx_part, 'dy_part_mm': dy_part,
        'grid': [nx, ny, nz], 'nodes': n_nodes, 'elements': nx * ny * nz,
        'sets': {k: len(v) for k, v in sets.items()},
        'domain_mm': {'x': [X_MIN, X_MAX], 'y': [Y_MIN, Y_MAX],
                      'z': [-Z_SUB, z_top]},
        'legs_in_domain': {'L1_thick': [0.0, 5.0], 'L2_thin': [7.0, 7.5],
                           'L3_med': [9.5, 12.0]},
        'notes': [
            'one leg-group period of the D-07 cell; pitch 14 mm from '
            'feature_id_map (L1 at 0.0, L4 at 14.0)',
            'x=0 is the part real free end; x=14 cuts powder gap (legs band)',
            'cut faces traction-free / no side-face BC: registered '
            'truncation assumption, identical across all 15 matrix members',
            'only the build z-discretisation varies with N; in-plane grid '
            'and substrate grading held fixed',
            f'thin 0.5 mm leg: {round(0.5 / dx_part)} elements across',
        ],
    }
    sfile = os.path.join(out_dir, f'amb_d11_N{N}_summary.json')
    with open(sfile, 'w') as f:
        json.dump(summary, f, indent=1)
        f.write('\n')
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ('notes', 'legs_in_domain')}, indent=1))
    print(f'wrote {inp} ({os.path.getsize(inp) / 1e6:.1f} MB) and summary')
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('N', type=int, nargs='?', default=50,
                    help='physical layers lumped per computational layer')
    ap.add_argument('--dx-part', type=float, default=DX_PART)
    ap.add_argument('--dy-part', type=float, default=DY_PART)
    ap.add_argument('--z-top', type=float, default=Z_TOP_DEFAULT)
    a = ap.parse_args()
    build_mesh(a.N, dx_part=a.dx_part, dy_part=a.dy_part, z_top=a.z_top)
