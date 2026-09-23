"""D-11 diagnostic A: is the N non-convergence a metric-resolution artefact?

Stage 1 measured M1/M2 with the z sampling BOUND TO N -- the mesh z cell height
is the computational layer thickness, so the leg-root section is resolved by 5
elements at N=50, 10 at N=25, 25 at N=10. Part of the 26.7 % / 16.9 % is
therefore a pure numerical staircase effect, not aggregation physics.

This tool re-reads the stage-1 finals and re-computes M1/M2 after block
averaging every member onto the COARSEST member's z bands (N=50, 1.0 mm).

  Coarsened M1 : sigma_xx band-averaged on the leg mid-plane vertical line,
                 relative L2 difference between adjacent N over the 7 bands.
  Coarsened M2 : Mroot rebuilt from band-averaged sigma_xx over the legs band
                 (z in [0, 5] mm), lever arm about the band-set area centroid.

Two readings the report must carry (pre-registered, do not drop):
  * Block averaging is NOT hiding the difference -- it is the common ruler the
    coarsest member can actually resolve. A coarse member cannot answer a
    sub-cell question, so this is the correct like-for-like comparison.
  * But N=50 gives only 5 bands over the leg. That ruler is blunt, so a "pass"
    on it is WEAK evidence. Strong evidence needs deliverable B (every member
    on the same 0.1 mm mesh).

Deciding metric is the load-bearing thick leg L1 only (pre-registered
2026-08-04); L2/L3 are reported as ABSOLUTE differences because their signal is
5-20x smaller and relative metrics on them are noise.

Usage: python d11_coarsen_metrics.py <runs_root> [--out derived/d11]
"""
import argparse
import glob
import json
import os
import re

import meshio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)

N_ORDER = [50, 25, 10]
LEGS = {'L1_thick': (0.0, 5.0), 'L2_thin': (7.0, 7.5), 'L3_med': (9.5, 12.0)}
DECIDING_LEG = 'L1_thick'
LEGS_BAND_TOP_MM = 5.0
BAND_H_MM = 1.0                 # N=50 computational layer = coarsest ruler
Z_TOP_MM = 7.0
STATE_SOLID = 2
THRESH = 0.05
STAGE1 = {'M1': 0.2670, 'M2': 0.1690}    # N50 vs N25, L1, raw (stage 1)
GATE2_DROP = 0.50               # pre-registered "significant improvement"


def cell_field(mesh, name):
    return np.concatenate([np.asarray(a) for a in mesh.cell_data[name]])


def quad_mean(mesh, base, suffix=''):
    names = [n for n in mesh.cell_data
             if re.fullmatch(rf'{re.escape(base)}\d+{re.escape(suffix)}', n)]
    if not names:
        raise SystemExit(f'{base}*{suffix} missing')
    return np.mean(np.stack([cell_field(mesh, n) for n in sorted(names)]),
                   axis=0)


def final_vtu(run_dir):
    v = sorted(glob.glob(os.path.join(run_dir, 'step_*.vtu')),
               key=lambda f: int(re.search(r'step_(\d+)', f).group(1)))
    if not v:
        raise SystemExit(f'no step_*.vtu in {run_dir}')
    return v[-1]


def load_sections(run_dir):
    """Per leg: the x-normal section at the leg mid-plane, and the mid-y line."""
    mesh = meshio.read(final_vtu(run_dir))
    cells = np.concatenate([b.data for b in mesh.cells
                            if b.type == 'hexahedron'])
    P = mesh.points[cells] * 1e3
    xc, yc, zc = (P[:, :, i].mean(axis=1) for i in range(3))
    dz = P[:, :, 2].max(axis=1) - P[:, :, 2].min(axis=1)
    dy = P[:, :, 1].max(axis=1) - P[:, :, 1].min(axis=1)
    sxx = quad_mean(mesh, 'stress_quad', '_xx') / 1e6
    state = cell_field(mesh, 'material_state')
    solid = (zc > 1e-6) & (state == STATE_SOLID)

    uy = np.unique(np.round(yc[solid], 6))
    y_plane = float(uy[np.argmin(np.abs(uy))])

    out = {}
    for leg, (x0, x1) in LEGS.items():
        ux = np.unique(np.round(xc[solid], 6))
        xp = float(ux[np.argmin(np.abs(ux - 0.5 * (x0 + x1)))])
        sec = np.isclose(xc, xp, atol=1e-6) & solid
        line = sec & np.isclose(yc, y_plane, atol=1e-6)
        out[leg] = {
            'x_plane_mm': xp,
            'sec': {'z': zc[sec], 'sxx': sxx[sec],
                    'area': dy[sec] * dz[sec], 'dz': dz[sec]},
            'line': {'z': zc[line], 'sxx': sxx[line], 'dz': dz[line]},
        }
    return out


def bands(z_top=Z_TOP_MM, h=BAND_H_MM):
    edges = np.arange(0.0, z_top + 1e-9, h)
    return list(zip(edges[:-1], edges[1:]))


def band_average(z, v, w, band_list):
    """Weight-averaged v per band; NaN where a band holds nothing."""
    out = []
    for lo, hi in band_list:
        m = (z > lo - 1e-9) & (z <= hi + 1e-9)
        out.append(float(np.sum(v[m] * w[m]) / np.sum(w[m]))
                   if m.any() and np.sum(w[m]) > 0 else np.nan)
    return np.array(out)


def rel_l2(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any() or np.linalg.norm(b[m]) == 0:
        return None
    return float(np.linalg.norm(a[m] - b[m]) / np.linalg.norm(b[m]))


def coarse_moment(sec, band_list):
    """Mroot rebuilt from band-averaged sigma_xx over the legs band."""
    z, sxx, area = sec['z'], sec['sxx'], sec['area']
    keep = [b for b in band_list if b[1] <= LEGS_BAND_TOP_MM + 1e-9]
    sb = band_average(z, sxx, area, keep)
    ab, zb = [], []
    for lo, hi in keep:
        m = (z > lo - 1e-9) & (z <= hi + 1e-9)
        ab.append(float(np.sum(area[m])))
        zb.append(float(np.sum(z[m] * area[m]) / np.sum(area[m]))
                  if m.any() and np.sum(area[m]) > 0 else np.nan)
    ab, zb = np.array(ab), np.array(zb)
    ok = np.isfinite(sb) & np.isfinite(zb) & (ab > 0)
    if not ok.any():
        return None
    zbar = float(np.sum(zb[ok] * ab[ok]) / np.sum(ab[ok]))
    return float(np.sum(sb[ok] * (zb[ok] - zbar) * ab[ok]))


def raw_moment(sec):
    z, sxx, area = sec['z'], sec['sxx'], sec['area']
    m = z <= LEGS_BAND_TOP_MM + 1e-9
    if not m.any():
        return None
    zbar = float(np.sum(z[m] * area[m]) / np.sum(area[m]))
    return float(np.sum(sxx[m] * (z[m] - zbar) * area[m]))


def main(runs_root, out_dir):
    band_list = bands()
    data = {}
    for N in N_ORDER:
        d = os.path.join(runs_root, f'N{N}_T1000')
        if not os.path.isdir(d):
            raise SystemExit(f'missing stage-1 member {d}')
        data[N] = load_sections(d)

    report = {
        'schema_version': 'ambench.d11-coarsen/1',
        'purpose': 'diagnostic A: separate metric-resolution staircase from '
                   'aggregation physics',
        'band_height_mm': BAND_H_MM,
        'bands': [[float(a), float(b)] for a, b in band_list],
        'deciding_leg': DECIDING_LEG,
        'threshold': THRESH,
        'pre_registered': {
            'deciding_metric': 'L1 only; L2/L3 reported as absolute '
                               'differences, not relative, not deciding',
            'gate2': f'coarsened L1 M1 AND M2 must both fall >= '
                     f'{int(100 * GATE2_DROP)} % vs stage 1 '
                     f'({100 * STAGE1["M1"]:.1f} % / {100 * STAGE1["M2"]:.1f} %) '
                     f'to count as significant improvement',
        },
        'readings': [
            'Block averaging is NOT hiding the difference: it is the common '
            'ruler the coarsest member can actually resolve. A coarse member '
            'cannot answer a sub-cell question, so this is the correct '
            'like-for-like comparison.',
            'But N=50 gives only 5 bands over the leg. That ruler is blunt, '
            'so a PASS on it is WEAK evidence. Strong evidence requires '
            'deliverable B: every member on the same 0.1 mm mesh.',
        ],
        'per_member': {}, 'pairs': {},
    }

    for N in N_ORDER:
        report['per_member'][str(N)] = {
            leg: {'x_plane_mm': data[N][leg]['x_plane_mm'],
                  'section_cells': int(len(data[N][leg]['sec']['z'])),
                  'line_cells': int(len(data[N][leg]['line']['z'])),
                  'raw_Mroot_N_mm': raw_moment(data[N][leg]['sec']),
                  'coarse_Mroot_N_mm': coarse_moment(data[N][leg]['sec'],
                                                     band_list)}
            for leg in LEGS}

    for i in range(len(N_ORDER) - 1):
        c, f = N_ORDER[i], N_ORDER[i + 1]
        key = f'N{c}_vs_N{f}'
        entry = {}
        for leg in LEGS:
            sc, sf = data[c][leg], data[f][leg]
            pc = band_average(sc['line']['z'], sc['line']['sxx'],
                              sc['line']['dz'], band_list)
            pf = band_average(sf['line']['z'], sf['line']['sxx'],
                              sf['line']['dz'], band_list)
            mc = coarse_moment(sc['sec'], band_list)
            mf = coarse_moment(sf['sec'], band_list)
            rc, rf = raw_moment(sc['sec']), raw_moment(sf['sec'])
            entry[leg] = {
                'M1_coarse': rel_l2(pc, pf),
                'M2_coarse': (abs(mc - mf) / abs(mf)
                              if mc is not None and mf else None),
                'M2_raw': (abs(rc - rf) / abs(rf)
                           if rc is not None and rf else None),
                'abs_dsxx_band_MPa': float(np.nanmax(np.abs(pc - pf))),
                'abs_dMroot_N_mm': (abs(mc - mf)
                                    if mc is not None and mf is not None
                                    else None),
                'band_profile_coarse': [None if not np.isfinite(v) else v
                                        for v in pc],
                'band_profile_fine': [None if not np.isfinite(v) else v
                                      for v in pf],
            }
        L = entry[DECIDING_LEG]
        entry['verdict'] = {
            'M1_coarse': L['M1_coarse'], 'M2_coarse': L['M2_coarse'],
            'passes_5pct': bool(L['M1_coarse'] is not None
                                and L['M2_coarse'] is not None
                                and L['M1_coarse'] < THRESH
                                and L['M2_coarse'] < THRESH),
        }
        report['pairs'][key] = entry

    ref = report['pairs'].get('N50_vs_N25', {}).get('verdict', {})
    if ref.get('M1_coarse') is not None and ref.get('M2_coarse') is not None:
        d1 = 1.0 - ref['M1_coarse'] / STAGE1['M1']
        d2 = 1.0 - ref['M2_coarse'] / STAGE1['M2']
        report['gate2'] = {
            'M1_drop': d1, 'M2_drop': d2,
            'significant_improvement': bool(d1 >= GATE2_DROP
                                            and d2 >= GATE2_DROP),
        }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'a-diagnostics.json'), 'w') as fh:
        json.dump(report, fh, indent=1)
        fh.write('\n')

    def pct(v):
        return 'n/a' if v is None else f'{100 * v:.2f} %'

    L = []
    L.append('## D-11 diagnostic A: metric resolution vs aggregation physics')
    L.append('')
    L.append(f'All members block-averaged onto the coarsest ruler '
             f'({BAND_H_MM:.1f} mm z bands = the N=50 computational layer).')
    L.append('')
    for r in report['readings']:
        L.append(f'> {r}')
        L.append('')
    L.append(f'### Deciding leg {DECIDING_LEG} (pre-registered)')
    L.append('')
    L.append('| pair | M1 raw (stage 1) | M1 coarsened | M2 raw | M2 coarsened | passes 5 % |')
    L.append('|---|---|---|---|---|---|')
    raw1 = {'N50_vs_N25': STAGE1['M1'], 'N25_vs_N10': 0.4454}
    raw2 = {'N50_vs_N25': STAGE1['M2'], 'N25_vs_N10': 0.4648}
    for key, e in report['pairs'].items():
        v = e['verdict']
        L.append(f'| {key} | {pct(raw1.get(key))} | {pct(v["M1_coarse"])} '
                 f'| {pct(raw2.get(key))} | {pct(v["M2_coarse"])} '
                 f'| {"yes" if v["passes_5pct"] else "NO"} |')
    if 'gate2' in report:
        g = report['gate2']
        L.append('')
        L.append(f'### Gate 2 (pre-registered: both drop >= '
                 f'{int(100 * GATE2_DROP)} %)')
        L.append('')
        L.append(f'- M1 drop {100 * g["M1_drop"]:.1f} %, '
                 f'M2 drop {100 * g["M2_drop"]:.1f} %')
        L.append(f'- **significant improvement: '
                 f'{"YES" if g["significant_improvement"] else "NO"}**')
    L.append('')
    L.append('### L2 / L3 (absolute differences, not deciding)')
    L.append('')
    L.append('| pair | leg | max band |d sigma_xx| (MPa) | |d Mroot| (N*mm) |')
    L.append('|---|---|---|---|')
    for key, e in report['pairs'].items():
        for leg in ('L2_thin', 'L3_med'):
            m = e[leg]
            dm = ('n/a' if m['abs_dMroot_N_mm'] is None
                  else f'{m["abs_dMroot_N_mm"]:.3f}')
            L.append(f'| {key} | {leg} | {m["abs_dsxx_band_MPa"]:.2f} | {dm} |')
    L.append('')
    L.append('### Leg-root moment on the common ruler (N*mm)')
    L.append('')
    L.append('| N | raw Mroot | coarsened Mroot |')
    L.append('|---|---|---|')
    for N in N_ORDER:
        m = report['per_member'][str(N)][DECIDING_LEG]
        L.append(f'| {N} | {m["raw_Mroot_N_mm"]:.1f} | '
                 f'{m["coarse_Mroot_N_mm"]:.1f} |')

    # ---- thermal history section (a-exposure.json, if produced) ------------
    exp_path = os.path.join(out_dir, 'a-exposure.json')
    if os.path.exists(exp_path):
        ex = json.load(open(exp_path))
        report['exposure'] = ex
        L.append('')
        L.append('### Thermal history per N (exposure recorder)')
        L.append('')
        L.append('Peak temperature is given in BOTH conventions, because the '
                 'two used in the thread differ by exactly the plate '
                 'temperature (347.05 K). These are peaks over CONSOLIDATED '
                 'PART cells; the thermal-probe table quotes the global nodal '
                 'maximum, which is higher because it includes powder and '
                 'free-surface nodes.')
        L.append('')
        L.append('| N | peak T abs (K) | peak T rise (K) | cells ever > T_cut | '
                 'max exposure > T_cut (s) | comp layers | cycles / mm build |')
        L.append('|---|---|---|---|---|---|---|')
        for N in N_ORDER:
            e = ex['per_N'].get(str(N))
            if not e or 'error' in e:
                L.append(f'| {N} | n/a | n/a | n/a | n/a | n/a | n/a |')
                continue
            L.append(f'| {N} | {e["peak_T_abs_K"]:.1f} | '
                     f'{e["peak_T_rise_K"]:.1f} | '
                     f'{100 * e["frac_cells_above_T_cut"]:.2f} % | '
                     f'{e["exposure_above_T_cut_s"]["max"]:.1f} | '
                     f'{e["computational_layers"]} | '
                     f'{e["cycles_per_mm_build"]:.2f} |')
        L.append('')
        L.append(f'> {ex["note"]}')
    md = '\n'.join(L) + '\n'
    with open(os.path.join(out_dir, 'a-diagnostics.md'), 'w') as fh:
        fh.write(md)
    print(md)
    print(f'wrote {out_dir}/a-diagnostics.json and .md')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs_root')
    ap.add_argument('--out', default=os.path.join(CASE, 'derived', 'd11'))
    a = ap.parse_args()
    main(a.runs_root, a.out)
