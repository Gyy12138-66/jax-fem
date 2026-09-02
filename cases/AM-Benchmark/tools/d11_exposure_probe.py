"""D-11 diagnostic A, part 2: quantify how the thermal history differs with N.

Runs a THERMAL-ONLY case per N with dense VTU output and records, per cell:
  * cumulative time above T_cut (default 1273.15 K = the 1000 C member), and
  * post-activation peak temperature (the solver's own
    `max_temperature_history` cell field, read off the final step).

Deliberately does NOT try to reconstruct this from the stage-1 production VTUs:
those were written sparsely, so their time axis cannot carry an exposure
integral. Re-probing thermal-only is cheap (mechanics is ~90 % of the cost).

Peak temperature is reported BOTH ways, because the two conventions in the
thread differ by exactly the plate temperature:
    absolute T_peak [K]      and      T_peak - 347.05 K  (rise above plate)

Also reports thermal cycles per unit build height (= computational layers /
build height). That one is aggregation-intrinsic: 7/14/35 layers over the same
7 mm is what layer lumping MEANS, and no meshing or scan convention can remove
it.

Usage:
  python d11_exposure_probe.py [--n 50 25 10] [--dwell 8] [--mesh <inp>]
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess

import meshio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
PLATE_K = 347.05
T_CUT_K = 1273.15
BUILD_MM = 7.0
STATE_SOLID = 2


def run_probe(N, dwell, out_dir, mesh=None, threads=16):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    env = dict(os.environ, OUT_ROOT=out_dir,
               MKL_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
    if mesh:
        env['D11_MESH'] = mesh
    cmd = ['bash', os.path.join(HERE, 'run_d11_case.sh'), str(N), '1000',
           '--mechanics-every', '0', '--cooling-steps', '0',
           '--dwell-steps-between-layers', str(dwell),
           '--thermal-output-every', '1', '--summary-every', '50']
    log = out_dir + '.log'
    os.makedirs(os.path.dirname(log) or '.', exist_ok=True)
    with open(log, 'w') as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return rc, log


def analyse(out_dir, dt_s):
    vtus = sorted(glob.glob(os.path.join(out_dir, 'step_*.vtu')),
                  key=lambda f: int(re.search(r'step_(\d+)', f).group(1)))
    if not vtus:
        raise SystemExit(f'no VTUs in {out_dir}')
    last = meshio.read(vtus[-1])
    cells = np.concatenate([b.data for b in last.cells
                            if b.type == 'hexahedron'])
    zc = last.points[cells][:, :, 2].mean(axis=1) * 1e3
    state = np.concatenate([np.asarray(a)
                            for a in last.cell_data['material_state']])
    part = (zc > 1e-6) & (state == STATE_SOLID)
    maxT = np.concatenate([np.asarray(a) for a in
                           last.cell_data['max_temperature_history']])

    exposure = np.zeros(len(cells))
    for v in vtus:
        m = meshio.read(v)
        # point_data['T'] comes back as (npoints, 1); ravel or the cell mean
        # stays 2-D and the accumulate below broadcasts to (ncell, ncell)
        Tn = np.asarray(m.point_data['T']).reshape(-1)
        Tc = Tn[cells].mean(axis=1)
        exposure += np.where(Tc > T_CUT_K, dt_s, 0.0)

    ex_p = exposure[part]
    mx_p = maxT[part]
    hot = ex_p > 0
    return {
        'part_cells': int(part.sum()),
        'steps_sampled': len(vtus),
        'peak_T_abs_K': float(mx_p.max()),
        'peak_T_rise_K': float(mx_p.max() - PLATE_K),
        'mean_peak_T_abs_K': float(mx_p.mean()),
        'mean_peak_T_rise_K': float(mx_p.mean() - PLATE_K),
        'frac_cells_above_T_cut': float(hot.mean()),
        'exposure_above_T_cut_s': {
            'max': float(ex_p.max()), 'mean': float(ex_p.mean()),
            'mean_over_hot_cells': float(ex_p[hot].mean()) if hot.any() else 0.0,
            'total_cell_seconds': float(ex_p.sum())},
    }


def main(ns, dwell, mesh, out_dir, threads):
    res = {'schema_version': 'ambench.d11-exposure/1',
           'T_cut_K': T_CUT_K, 'plate_K': PLATE_K,
           'dwell_steps': dwell, 'shared_mesh': mesh,
           'note': 'thermal-only; dwell truncated, so exposure is a LOWER '
                   'bound on a full-dwell run (material keeps cooling past '
                   'the truncation). Comparison across N is like-for-like.',
           'per_N': {}}
    for N in ns:
        d = f'/tmp/d11_exposure_N{N}'
        rc, log = run_probe(N, dwell, d, mesh=mesh, threads=threads)
        if rc != 0:
            res['per_N'][str(N)] = {'error': f'probe rc={rc}, see {log}'}
            print(f'N={N}: FAILED rc={rc}')
            continue
        case = json.load(open(os.path.join(d, 'd11_case.json')))
        a = analyse(d, float(case['dt_s']))
        a['dt_s'] = float(case['dt_s'])
        a['computational_layers'] = int(case['computational_layers'])
        a['cycles_per_mm_build'] = int(case['computational_layers']) / BUILD_MM
        res['per_N'][str(N)] = a
        print(f'N={N:3d}  peak {a["peak_T_abs_K"]:.1f} K '
              f'(rise {a["peak_T_rise_K"]:.1f} K)  '
              f'hot cells {100 * a["frac_cells_above_T_cut"]:.1f} %  '
              f'exposure max {a["exposure_above_T_cut_s"]["max"]:.1f} s  '
              f'cycles/mm {a["cycles_per_mm_build"]:.2f}')
        shutil.rmtree(d, ignore_errors=True)

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, 'a-exposure.json')
    with open(p, 'w') as fh:
        json.dump(res, fh, indent=1)
        fh.write('\n')
    print(f'wrote {p}')
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, nargs='+', default=[50, 25, 10])
    ap.add_argument('--dwell', type=int, default=8)
    ap.add_argument('--mesh', default=None,
                    help='shared mesh .inp (sets D11_MESH)')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--out', default=os.path.join(CASE, 'derived', 'd11'))
    a = ap.parse_args()
    main(a.n, a.dwell, a.mesh, a.out, a.threads)
