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
DECIDING_LEG = 'L1_thick'                   # pre-registered 2026-08-04
BAND_REF_TCUT = '1000'                      # Balbaa precedent = band centre
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
        # DECIDING values are the load-bearing thick leg L1 (pre-registered
        # 2026-08-04). The max-over-legs values are kept as background only:
        # L2/L3 carry 5-20x less signal, so their small denominators inflate
        # the relative metric (B produced 212-5262 % that way).
        'M1_L1': m1.get(DECIDING_LEG), 'M2_L1': m2.get(DECIDING_LEG),
        'M1': max(ok1) if ok1 else None,
        'M2': max(ok2) if ok2 else None,
        'M3_advisory': rel_l2(top_profile(coarse), top_profile(fine)),
    }


def mroot(run, leg=None):
    """Leg-root moment [N*mm] for the deciding leg."""
    m = run['legs'].get(leg or DECIDING_LEG, {}).get('moment', {}).get('root')
    return None if not m else m['M_N_mm']


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
            # verdict on the deciding leg only (pre-registered 2026-08-04)
            m['converged'] = (m['M1_L1'] is not None and m['M2_L1'] is not None
                              and m['M1_L1'] < THRESH
                              and m['M2_L1'] < THRESH)
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

    # ---- T_cut band on the reference N ------------------------------------
    # B closed with "true aggregation non-convergence", so there is no frozen
    # N. The band is therefore reported on a REFERENCE member (N=10, closest to
    # real layer-by-layer physics), which is a reporting choice, not a freeze.
    band_N = frozen_N if report.get('frozen_N_converged') else 10
    report['band_N'] = band_N
    report['band_N_basis'] = (
        'frozen N when the dual metric licenses one; otherwise the reference '
        'member N=10 (finest lumping = closest to real layer-by-layer '
        'physics). Reporting on a reference N is NOT a freeze.')

    base = runs.get((band_N, BAND_REF_TCUT, 'mean'))
    if base:
        band = {}
        for t in TCUTS:
            if t == BAND_REF_TCUT:
                continue
            other = runs.get((band_N, t, 'mean'))
            if not other:
                continue
            band[t] = pair_metrics(other, base)
        report['tcut_band'][str(band_N)] = band

        # ---- pre-registered band formulas (2026-08-07) --------------------
        #   band_M1 = max_i M1(member i vs T1000)         [L1 only]
        #   band_M2 = (max Mroot - min Mroot) / |Mroot(T1000)|
        # band_M2 is a SPREAD over the available members, not a max of
        # pairwise differences -- so a missing member (e.g. a diverged `none`)
        # narrows the band rather than corrupting it.
        m_ref = mroot(base)
        moments = {BAND_REF_TCUT: m_ref}
        for t, run in ((t, runs.get((band_N, t, 'mean'))) for t in TCUTS):
            if run is not None and t != BAND_REF_TCUT:
                moments[t] = mroot(run)
        moments = {k: v for k, v in moments.items() if v is not None}
        b1 = [v['M1_L1'] for v in band.values() if v['M1_L1'] is not None]
        # A spread needs at least two members. With one member the formula is
        # trivially 0, which would read as "no T_cut sensitivity" when it
        # actually means "not measured yet" -- report None instead.
        band_m2 = None
        if len(moments) >= 2 and m_ref:
            band_m2 = ((max(moments.values()) - min(moments.values()))
                       / abs(m_ref))
        report['tcut_band_preregistered'] = {
            'formula_M1': 'max_i M1(member i vs T1000), L1 only',
            'formula_M2': '(max Mroot - min Mroot) / |Mroot(T1000)|, L1',
            'members_used': sorted(moments),
            'members_missing': [t for t in TCUTS if t not in moments],
            'Mroot_N_mm': moments,
            'band_M1': max(b1) if b1 else None,
            'band_M2': band_m2,
            'band_complete': not [t for t in TCUTS if t not in moments],
        }
        # background only: the max-over-legs reading
        vals1 = [v['M1'] for v in band.values() if v['M1'] is not None]
        vals2 = [v['M2'] for v in band.values() if v['M2'] is not None]
        report['tcut_band_max_over_legs_background'] = {
            'M1': max(vals1) if vals1 else None,
            'M2': max(vals2) if vals2 else None}

        # zero-exposure argument (HIGH-T note section 1): the 800 C member
        # has zero extrapolation exposure for sigma_y form, E and alpha.
        z = band.get('800')
        if z:
            report['zero_exposure_800C'] = {
                'M1_L1': z['M1_L1'], 'M2_L1': z['M2_L1'],
                'interpretation':
                    'T_cut = 800 C exposes NO extrapolated sigma_y form '
                    '(window [773,1073] only), no E extrapolation '
                    '(data ends 1144 K) and no alpha extrapolation '
                    '(data ends 1200 K). If its band vs 1000 C is inside '
                    'the threshold, the sigma_y high-T form, the E '
                    'extrapolation and the alpha extrapolation are all '
                    'shown irrelevant to the result in one stroke.'}

    # ---- propagated MaCTO yield uncertainty, same metric units ------------
    # Pre-registered isomorphic form: the arms must be run on the SAME shared
    # mesh, same N, same T_cut as the band. Stage-1 arm values are on the old
    # per-N meshes and MUST NOT be compared across that convention change.
    for arm in ('mlow', 'mhigh', 'bd', 'td'):
        other = runs.get((band_N, BAND_REF_TCUT, arm))
        if other and base:
            report['yield_uncertainty'][arm] = pair_metrics(other, base)
    u1 = [v['M1_L1'] for v in report['yield_uncertainty'].values()
          if v['M1_L1'] is not None]
    m_lo = runs.get((band_N, BAND_REF_TCUT, 'mlow'))
    m_hi = runs.get((band_N, BAND_REF_TCUT, 'mhigh'))
    arm_band_M2 = None
    if m_lo and m_hi and base and mroot(base):
        a, b = mroot(m_lo), mroot(m_hi)
        if a is not None and b is not None:
            arm_band_M2 = abs(b - a) / abs(mroot(base))
    report['yield_uncertainty_preregistered'] = {
        'formula_M1': 'spread of M1(arm vs mean), L1 only',
        'formula_M2': '|Mroot(mhigh) - Mroot(mlow)| / |Mroot(mean)|, L1',
        'band_M1': max(u1) if u1 else None,
        'band_M2': arm_band_M2,
        'both_arms_present': bool(m_lo and m_hi),
        'note': 'freeze/interval decision REQUIRES both arms on this same '
                'mesh; until then the comparison has no like-for-like '
                'denominator and no freeze may be recorded.',
    }
    report['yield_uncertainty_max'] = {
        'M1': max(u1) if u1 else None, 'M2': arm_band_M2}

    # ---- freeze recommendation --------------------------------------------
    band = report.get('tcut_band_preregistered') or {}
    unc = report.get('yield_uncertainty_preregistered') or {}
    rec = {'N': frozen_N, 'band_N': band_N}
    if band.get('band_M1') is None:
        rec['T_cut'] = 'undetermined'
        rec['reason'] = 'T_cut column on the reference N is incomplete'
    elif not unc.get('both_arms_present'):
        rec['T_cut'] = 'undetermined'
        rec['reason'] = ('band measured, but the freeze/interval decision is '
                         'pre-registered as AWAITING BOTH ARMS: m_low and '
                         'm_high must run on this same shared mesh at '
                         f'N={band_N}, T_cut={BAND_REF_TCUT}. Stage-1 arm '
                         'values are on the old per-N meshes and must not be '
                         'compared across that convention change.')
        rec['tcut_band'] = {k: band.get(k) for k in ('band_M1', 'band_M2')}
    elif (band['band_M1'] <= unc['band_M1']
          and band.get('band_M2') is not None
          and unc.get('band_M2') is not None
          and band['band_M2'] <= unc['band_M2']):
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
             f'### N convergence (deciding leg {DECIDING_LEG}; '
             'M1 AND M2 both < 5 %)', '',
             '| T_cut | pair | M1 (L1) | M2 (L1) | M3 (advisory) | '
             'converged | M1/M2 max-over-legs (background) |',
             '|---|---|---|---|---|---|---|']
    for t, col in report['convergence'].items():
        for pair, m in col.items():
            lines.append(f'| {t} | {pair} | {fmt(m["M1_L1"])} '
                         f'| {fmt(m["M2_L1"])} | {fmt(m["M3_advisory"])} | '
                         f'{"yes" if m["converged"] else "NO"} | '
                         f'{fmt(m["M1"])} / {fmt(m["M2"])} |')
    if report['tcut_band']:
        pre = report.get('tcut_band_preregistered', {})
        lines += ['', f'### T_cut band on N = {band_N} '
                      f'(vs T_cut = {BAND_REF_TCUT} C)', '',
                  f'Reference member basis: {report["band_N_basis"]}', '',
                  '| T_cut | M1 (L1) | M2 (L1) | M3 (advisory) |',
                  '|---|---|---|---|']
        for t, m in report['tcut_band'][str(band_N)].items():
            lines.append(f'| {t} | {fmt(m["M1_L1"])} | {fmt(m["M2_L1"])} | '
                         f'{fmt(m["M3_advisory"])} |')
        lines += ['', '**Pre-registered band (2026-08-07)**', '',
                  f'- `band_M1 = max_i M1(member i vs T{BAND_REF_TCUT})` '
                  f'= {fmt(pre.get("band_M1"))}',
                  f'- `band_M2 = (max Mroot - min Mroot) / '
                  f'|Mroot(T{BAND_REF_TCUT})|` = {fmt(pre.get("band_M2"))}',
                  f'- members used: {", ".join(pre.get("members_used", []))}'
                  + (f'; MISSING: {", ".join(pre["members_missing"])}'
                     if pre.get('members_missing') else ''),
                  f'- Mroot (N*mm): ' + ', '.join(
                      f'{k}={v:.1f}' for k, v in
                      sorted((pre.get('Mroot_N_mm') or {}).items()))]
        bg = report.get('tcut_band_max_over_legs_background', {})
        lines.append(f'- background only, max over legs: M1 '
                     f'{fmt(bg.get("M1"))}, M2 {fmt(bg.get("M2"))}')
    if report['yield_uncertainty']:
        up = report.get('yield_uncertainty_preregistered', {})
        lines += ['', '### Propagated MaCTO yield uncertainty (same units)',
                  '', '| arm | M1 (L1) | M2 (L1) |', '|---|---|---|']
        for arm, m in report['yield_uncertainty'].items():
            lines.append(f'| {arm} | {fmt(m["M1_L1"])} | {fmt(m["M2_L1"])} |')
        lines += ['',
                  f'- `band_M2 = |Mroot(mhigh) - Mroot(mlow)| / '
                  f'|Mroot(mean)|` = {fmt(up.get("band_M2"))}',
                  f'- both arms present: {up.get("both_arms_present")}']
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
