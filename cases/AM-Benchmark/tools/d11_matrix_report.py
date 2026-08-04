"""D-11 matrix report: dual convergence metrics + freeze recommendation.

Consumes the per-run d11_metrics.json files and applies D.7 verbatim:

  M1 = ||sxx_Ni(z) - sxx_Ni+1(z)||_2 / ||sxx_Ni+1(z)||_2   (leg mid-plane
       vertical line, both profiles interpolated to a common z grid)
  M2 = |Mroot_Ni - Mroot_Ni+1| / |Mroot_Ni+1|              (leg-root section)
  Converged when M1 < 5 % AND M2 < 5 %  -- both required.

M1/M2 are reported per leg and the DECIDING value is the max over the three
legs in the sub-domain (the conservative reading of "both required": no leg
may be unconverged). M3 (top-surface u_z profile, same L2 form as M1) is
advisory only -- it is not part of the approved freeze rule, it is reported
because a stress-type pair can converge while the height-integrated
displacement has not.

Freeze rule (D.7): N is frozen by the dual metric. On the frozen N the T_cut
sensitivity band is reported; if it is below the propagated MaCTO yield
uncertainty, freeze T_cut = 1000 C and close the gap, otherwise T_cut is NOT
frozen and the endpoints go forward as a prediction band. The propagated
yield uncertainty is measured in the SAME metric units by rerunning the
frozen case on the m_low / m_high yield arms (--arm), so the comparison is
like-for-like rather than a units-mismatched guess.

Usage:
  python d11_matrix_report.py <runs_root> [--out derived/d11]
where <runs_root> holds one directory per member, each with d11_metrics.json.
"""
import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)

N_ORDER = [50, 25, 10]                      # coarse -> fine
TCUTS = ['800', '900', '1000', '1100', 'none']
THRESH = 0.05
GRID_PTS = 200


def common_grid(profiles):
    lo = max(min(p['z']) for p in profiles)
    hi = min(max(p['z']) for p in profiles)
    if hi <= lo:
        return None
    return np.linspace(lo, hi, GRID_PTS)


def rel_l2(coarse, fine):
    """||coarse - fine||_2 / ||fine||_2 on a common grid."""
    g = common_grid([coarse, fine])
    if g is None:
        return None
    a = np.interp(g, coarse['z'], coarse['v'])
    b = np.interp(g, fine['z'], fine['v'])
    nb = np.linalg.norm(b)
    if nb == 0:
        return None
    return float(np.linalg.norm(a - b) / nb)


def load(runs_root):
    runs = {}
    for f in sorted(glob.glob(os.path.join(runs_root, '*', 'd11_metrics.json'))
                    + glob.glob(os.path.join(runs_root, '*.json'))):
        try:
            d = json.load(open(f))
        except (ValueError, OSError):
            continue
        if d.get('schema_version') != 'ambench.d11-metrics/1':
            continue
        case = d.get('case') or {}
        try:
            key = (int(case['N']), str(case['T_cut_C']),
                   case.get('arm', 'mean'))
        except (KeyError, TypeError, ValueError):
            print(f'skipping {f}: no usable d11_case.json provenance')
            continue
        runs[key] = d
    return runs


def profile(run, leg):
    return {'z': run['legs'][leg]['profile_z_mm'],
            'v': run['legs'][leg]['profile_sxx_MPa']}


def top_profile(run):
    return {'z': run['top_surface']['x_mm'], 'v': run['top_surface']['uz_mm']}


def pair_metrics(coarse, fine):
    legs = sorted(set(coarse['legs']) & set(fine['legs']))
    m1 = {leg: rel_l2(profile(coarse, leg), profile(fine, leg))
          for leg in legs}
    m2 = {}
    for leg in legs:
        a = coarse['legs'][leg]['moment']['root']
        b = fine['legs'][leg]['moment']['root']
        if not a or not b or b['M_N_mm'] == 0:
            m2[leg] = None
            continue
        m2[leg] = abs(a['M_N_mm'] - b['M_N_mm']) / abs(b['M_N_mm'])
    ok1 = [v for v in m1.values() if v is not None]
    ok2 = [v for v in m2.values() if v is not None]
    return {
        'M1_per_leg': m1, 'M2_per_leg': m2,
        'M1': max(ok1) if ok1 else None,
        'M2': max(ok2) if ok2 else None,
        'M3_advisory': rel_l2(top_profile(coarse), top_profile(fine)),
    }


def fmt(v):
    return 'n/a' if v is None else f'{100 * v:.2f} %'


def main(runs_root, out_dir):
    runs = load(runs_root)
    if not runs:
        raise SystemExit(f'no d11_metrics.json found under {runs_root}')
    present = sorted(runs)
    report = {
        'schema_version': 'ambench.d11-matrix-report/1',
        'decision': 'D-11 / PREREQUISITES.md D.7',
        'threshold': THRESH,
        'members_present': [f'N{n}_T{t}_{a}' for n, t, a in present],
        'members_expected': len(N_ORDER) * len(TCUTS),
        'convergence': {}, 'tcut_band': {}, 'yield_uncertainty': {},
    }

    # ---- N convergence, per T_cut column ----------------------------------
    frozen_N = None
    for t in TCUTS:
        col = {}
        for i in range(len(N_ORDER) - 1):
            coarse, fine = N_ORDER[i], N_ORDER[i + 1]
            a, b = runs.get((coarse, t, 'mean')), runs.get((fine, t, 'mean'))
            if not a or not b:
                continue
            m = pair_metrics(a, b)
            m['converged'] = (m['M1'] is not None and m['M2'] is not None
                              and m['M1'] < THRESH and m['M2'] < THRESH)
            col[f'N{coarse}_vs_N{fine}'] = m
        if col:
            report['convergence'][t] = col

    ref = report['convergence'].get('1000', {})
    if ref.get('N50_vs_N25', {}).get('converged'):
        frozen_N = 50
    elif ref.get('N25_vs_N10', {}).get('converged'):
        frozen_N = 25
    elif ref:
        frozen_N = 10                       # not converged at the finest pair
    report['frozen_N'] = frozen_N
    report['frozen_N_basis'] = (
        'dual metric on the T_cut = 1000 C column; the coarsest N whose '
        'pair against the next finer N satisfies M1 < 5 % AND M2 < 5 %. '
        'If no pair converges, N = 10 is reported as NOT converged and the '
        'matrix does not license a freeze.')
    report['frozen_N_converged'] = bool(
        frozen_N and frozen_N != 10
        or (frozen_N == 10 and ref.get('N25_vs_N10', {}).get('converged')))

    # ---- T_cut band on the frozen N ---------------------------------------
    if frozen_N:
        base = runs.get((frozen_N, '1000', 'mean'))
        if base:
            band = {}
            for t in TCUTS:
                if t == '1000':
                    continue
                other = runs.get((frozen_N, t, 'mean'))
                if not other:
                    continue
                band[t] = pair_metrics(other, base)
            report['tcut_band'][str(frozen_N)] = band
            vals1 = [v['M1'] for v in band.values() if v['M1'] is not None]
            vals2 = [v['M2'] for v in band.values() if v['M2'] is not None]
            report['tcut_band_max'] = {
                'M1': max(vals1) if vals1 else None,
                'M2': max(vals2) if vals2 else None}
            # zero-exposure argument (HIGH-T note section 1): the 800 C member
            # has zero extrapolation exposure for sigma_y form, E and alpha.
            z = band.get('800')
            if z:
                report['zero_exposure_800C'] = {
                    'M1': z['M1'], 'M2': z['M2'],
                    'interpretation':
                        'T_cut = 800 C exposes NO extrapolated sigma_y form '
                        '(window [773,1073] only), no E extrapolation '
                        '(data ends 1144 K) and no alpha extrapolation '
                        '(data ends 1200 K). If its band vs 1000 C is inside '
                        'the threshold, the sigma_y high-T form, the E '
                        'extrapolation and the alpha extrapolation are all '
                        'shown irrelevant to the result in one stroke.'}

        # ---- propagated MaCTO yield uncertainty, same metric units --------
        for arm in ('mlow', 'mhigh', 'bd', 'td'):
            other = runs.get((frozen_N, '1000', arm))
            if other and base:
                report['yield_uncertainty'][arm] = pair_metrics(other, base)
        u1 = [v['M1'] for v in report['yield_uncertainty'].values()
              if v['M1'] is not None]
        u2 = [v['M2'] for v in report['yield_uncertainty'].values()
              if v['M2'] is not None]
        report['yield_uncertainty_max'] = {
            'M1': max(u1) if u1 else None, 'M2': max(u2) if u2 else None}

    # ---- freeze recommendation --------------------------------------------
    band = report.get('tcut_band_max') or {}
    unc = report.get('yield_uncertainty_max') or {}
    rec = {'N': frozen_N}
    if band.get('M1') is None:
        rec['T_cut'] = 'undetermined'
        rec['reason'] = 'T_cut column on the frozen N is incomplete'
    elif unc.get('M1') is None:
        rec['T_cut'] = 'undetermined'
        rec['reason'] = ('propagated MaCTO yield uncertainty not measured; '
                         'rerun the frozen case with D11_ARM=mlow and '
                         'D11_ARM=mhigh to close the comparison')
        rec['tcut_band'] = band
    elif (band['M1'] <= unc['M1']) and (band.get('M2') is not None
                                        and unc.get('M2') is not None
                                        and band['M2'] <= unc['M2']):
        rec['T_cut'] = 1000
        rec['reason'] = ('T_cut band is inside the propagated MaCTO yield '
                         'uncertainty on both metrics: freeze T_cut = 1000 C '
                         '(Balbaa precedent) and close the gap')
    else:
        rec['T_cut'] = 'not frozen'
        rec['reason'] = ('T_cut band exceeds the propagated MaCTO yield '
                         'uncertainty: report the sweep endpoints as a '
                         'prediction band with the sensitivity as open '
                         'model uncertainty (D.7); decision sits with the '
                         'acceptance owner')
    rec['decided_by'] = 'proposal only - D.7 gives the call to the reviewer'
    report['recommendation'] = rec

    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, 'matrix-report.json')
    with open(jpath, 'w') as f:
        json.dump(report, f, indent=1)
        f.write('\n')

    lines = ['## D-11 matrix: dual convergence metrics', '',
             f'members present: {len(present)} / '
             f'{report["members_expected"]} (+ yield-uncertainty arms)', '',
             '### N convergence (max over legs; M1 AND M2 both < 5 %)', '',
             '| T_cut | pair | M1 | M2 | M3 (advisory) | converged |',
             '|---|---|---|---|---|---|']
    for t, col in report['convergence'].items():
        for pair, m in col.items():
            lines.append(f'| {t} | {pair} | {fmt(m["M1"])} | {fmt(m["M2"])} '
                         f'| {fmt(m["M3_advisory"])} | '
                         f'{"yes" if m["converged"] else "NO"} |')
    if report['tcut_band']:
        lines += ['', f'### T_cut band on N = {frozen_N} (vs T_cut = 1000 C)',
                  '', '| T_cut | M1 | M2 | M3 (advisory) |', '|---|---|---|---|']
        for t, m in report['tcut_band'][str(frozen_N)].items():
            lines.append(f'| {t} | {fmt(m["M1"])} | {fmt(m["M2"])} | '
                         f'{fmt(m["M3_advisory"])} |')
    if report['yield_uncertainty']:
        lines += ['', '### Propagated MaCTO yield uncertainty (same units)',
                  '', '| arm | M1 | M2 |', '|---|---|---|']
        for arm, m in report['yield_uncertainty'].items():
            lines.append(f'| {arm} | {fmt(m["M1"])} | {fmt(m["M2"])} |')
    lines += ['', '### Recommendation (proposal, not a freeze)', '',
              f'- N = {rec["N"]}', f'- T_cut = {rec["T_cut"]}',
              f'- {rec["reason"]}']
    mpath = os.path.join(out_dir, 'matrix-report.md')
    with open(mpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    print(f'\nwrote {jpath} and {mpath}')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs_root')
    ap.add_argument('--out', default=os.path.join(CASE, 'derived', 'd11'))
    a = ap.parse_args()
    main(a.runs_root, a.out)
