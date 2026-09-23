"""D-11 per-run metric extraction (PREREQUISITES.md D.7 dual metrics).

Reads the final VTU of one matrix member and writes the raw quantities the
matrix report differences between adjacent N:

  M1 source : sigma_xx along the leg mid-plane vertical line, per leg.
              D.7: M1 = ||sxx_Ni(z) - sxx_Ni+1(z)||_2 / ||sxx_Ni+1(z)||_2
              with the fields interpolated to a common z grid (the report
              does the interpolation; this tool stores the raw profile).
  M2 source : Mroot = integral sigma_xx * (z - zbar) dA over the leg-root
              cross section -- the x-normal section at the leg mid-plane,
              lever arm about the section's area centroid. Reported over the
              legs band z in [0, 5] mm (the leg proper, "root" section) and,
              as a robustness variant, over the full built height.
  M3 source : top-surface u_z(x) profile. Pure-computation displacement-type
              third metric (HIGH-T-EXTRAPOLATION-NOTE section 5): the
              stress-type pair M1/M2 can converge while the height-integrated
              displacement has not. Touches no measured value.

Legs inside the sub-domain (mid-planes): L1 thick x=2.5, L2 thin x=7.25,
L3 medium x=10.75 mm.

Usage: python d11_metrics.py <output_dir> [--out <json>]
"""
import argparse
import glob
import json
import os
import re

import meshio
import numpy as np

LEGS = {'L1_thick': (0.0, 5.0), 'L2_thin': (7.0, 7.5), 'L3_med': (9.5, 12.0)}
LEGS_BAND_TOP_MM = 5.0          # top of the legs band (physical layer 250)
STATE_SOLID = 2


def cell_field(mesh, *names):
    for n in names:
        if n in mesh.cell_data:
            return np.concatenate([np.asarray(a) for a in mesh.cell_data[n]])
    return None


def quad_mean(mesh, base, suffix=''):
    """Cell-average of a per-quadrature-point cell field.

    io/vtu.py names quad fields by splicing the index into the word 'quad'
    (quad_field_name -> 'stress_quad0_xx', 'vm_quad0'), so the pattern is
    <base>{q}<suffix>, not <base><suffix>{q}.
    """
    names = [n for n in mesh.cell_data
             if re.fullmatch(rf'{re.escape(base)}\d+{re.escape(suffix)}', n)]
    if not names:                                  # single-quadrature run
        if base + suffix in mesh.cell_data:
            names = [base + suffix]
        else:
            raise SystemExit(f'{base}*{suffix} not present in the VTU')
    return np.mean(np.stack([cell_field(mesh, n) for n in
                             sorted(names)]), axis=0)


def nearest_plane(values, target):
    """The unique coordinate closest to target (lowest wins on a tie)."""
    uniq = np.unique(np.round(values, 6))
    d = np.abs(uniq - target)
    return float(uniq[np.argmin(d)])


def main(run_dir, out_path=None):
    vtus = sorted(glob.glob(os.path.join(run_dir, 'step_*.vtu')),
                  key=lambda f: int(re.search(r'step_(\d+)', f).group(1)))
    if not vtus:
        raise SystemExit(f'no step_*.vtu in {run_dir}')
    mesh = meshio.read(vtus[-1])

    case_file = os.path.join(run_dir, 'd11_case.json')
    case = json.load(open(case_file)) if os.path.exists(case_file) else {}

    cells = np.concatenate([b.data for b in mesh.cells
                            if b.type == 'hexahedron'])
    pts = mesh.points[cells]                       # (ncell, 8, 3) in metres
    xc, yc, zc = (pts[:, :, i].mean(axis=1) * 1e3 for i in range(3))
    dz = (pts[:, :, 2].max(axis=1) - pts[:, :, 2].min(axis=1)) * 1e3
    dy = (pts[:, :, 1].max(axis=1) - pts[:, :, 1].min(axis=1)) * 1e3

    state = cell_field(mesh, 'material_state')
    sxx = quad_mean(mesh, 'stress_quad', '_xx') / 1e6              # MPa
    vm = quad_mean(mesh, 'vm_quad') / 1e6
    eqp = cell_field(mesh, 'eq_plastic_strain')
    solid = (zc > 1e-6) & (state == STATE_SOLID)

    y_plane = nearest_plane(yc[solid], 0.0)
    on_mid_y = np.isclose(yc, y_plane, atol=1e-6)

    result = {
        'schema_version': 'ambench.d11-metrics/1',
        'run_dir': run_dir,
        'final_vtu': os.path.basename(vtus[-1]),
        'case': case,
        'mid_y_plane_mm': y_plane,
        'solid_cells': int(solid.sum()),
        'legs': {},
        'global': {
            'vm_solid_MPa': {'max': float(vm[solid].max()),
                             'mean': float(vm[solid].mean()),
                             'p95': float(np.percentile(vm[solid], 95))},
            'eqp_solid': {'max': float(eqp[solid].max()),
                          'mean': float(eqp[solid].mean())},
            'sxx_solid_MPa': {'min': float(sxx[solid].min()),
                              'max': float(sxx[solid].max())},
        },
    }

    for name, (x0, x1) in LEGS.items():
        x_mid = 0.5 * (x0 + x1)
        x_plane = nearest_plane(xc[solid], x_mid)
        on_sec = np.isclose(xc, x_plane, atol=1e-6) & solid

        line = on_sec & on_mid_y
        order = np.argsort(zc[line])
        prof_z = zc[line][order].tolist()
        prof_s = sxx[line][order].tolist()

        moments = {}
        for tag, ztop in (('root', LEGS_BAND_TOP_MM),
                          ('full', float(zc[on_sec].max()) if on_sec.any()
                           else 0.0)):
            sel = on_sec & (zc <= ztop + 1e-9)
            if not sel.any():
                moments[tag] = None
                continue
            area = dy[sel] * dz[sel]                      # mm^2
            zbar = float(np.sum(zc[sel] * area) / np.sum(area))
            # Mroot = integral sigma_xx * (z - zbar) dA   [MPa*mm^3 = N*mm]
            moments[tag] = {
                'M_N_mm': float(np.sum(sxx[sel] * (zc[sel] - zbar) * area)),
                'zbar_mm': zbar,
                'area_mm2': float(np.sum(area)),
                'cells': int(sel.sum()),
            }

        result['legs'][name] = {
            'x_mid_nominal_mm': x_mid,
            'x_plane_mm': x_plane,
            'profile_cells': int(line.sum()),
            'profile_z_mm': prof_z,
            'profile_sxx_MPa': prof_s,
            'moment': moments,
        }

    # ---- M3: top-surface u_z(x) over the part width -------------------------
    u = np.asarray(mesh.point_data['u'])
    P = mesh.points * 1e3
    z_top = float(P[:, 2].max())
    top = (np.abs(P[:, 2] - z_top) < 1e-6) & (np.abs(P[:, 1]) <= 2.5 + 1e-6)
    xs = np.unique(np.round(P[top, 0], 6))
    uz = [float(np.mean(u[top][np.isclose(P[top, 0], x, atol=1e-6), 2]))
          for x in xs]
    result['top_surface'] = {
        'z_top_mm': z_top,
        'x_mm': xs.tolist(),
        'uz_mm': [v * 1e3 for v in uz],
        'uz_max_abs_mm': float(np.max(np.abs(uz)) * 1e3),
        'note': 'displacement-type third metric; pure computation, no '
                'measured value touched (HIGH-T note section 5)',
    }

    out_path = out_path or os.path.join(run_dir, 'd11_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=1)
        f.write('\n')
    print(f'solid cells {result["solid_cells"]}, '
          f'vm max {result["global"]["vm_solid_MPa"]["max"]:.1f} MPa, '
          f'eqp max {result["global"]["eqp_solid"]["max"]:.4f}')
    for name, leg in result['legs'].items():
        m = leg['moment']['root']
        print(f'  {name:9s} x={leg["x_plane_mm"]:6.3f}  '
              f'{leg["profile_cells"]:3d} line cells  '
              f'Mroot={m["M_N_mm"] if m else float("nan"):+.4g} N*mm')
    print(f'  top-surface |uz|max {result["top_surface"]["uz_max_abs_mm"]:.4f} mm')
    print(f'wrote {out_path}')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    main(a.run_dir, a.out)
