"""D-11 sigma_y(T) / hardening(T) closure from the MaCTO data (decision D.7).

Replaces the PROVISIONAL-L0 yield/hardening ramps (which say verbatim
"D-11 MaCTO tables replace this") with the form D.7 approved:

  T <= 773 K : MaCTO three-temperature 0.2 % offset yield interpolated as-is
               (298 / 523 / 773 K), BD/TD anisotropy kept as a bracket.
  T >  773 K : J-C-type power softening anchored at the 773 K value
               sigma_y(T) = sigma_y(773) * [1 - ((T-773)/(T_zero-773))^m],
               T_zero = solidus 1552 K.

m is fitted to the MaCTO three-point trend (no new free parameter): the same
one-parameter family anchored at the 298 K point,
`sigma_y(T) = sigma_y(298) * [1 - ((T-298)/(T_zero-298))^m]`, is least-squares
fitted to the two remaining measured points (523, 773 K); that m is then
reused in the 773-anchored form above 773 K. Reported alongside:

  m_low  = m that reproduces the 523 K point exactly
  m_high = m that reproduces the 773 K point exactly

which is the m bracket the high-T note (verification/HIGH-T-EXTRAPOLATION-NOTE
section 2b) asks for. The solver's J2 model is isotropic and cannot carry the
BD/TD anisotropy, so the ISOTROPIC table is the BD/TD mean and the BD/TD
spread is carried as the propagated MaCTO yield uncertainty -- which is
exactly the quantity D.7's freeze rule compares the T_cut band against.

Hardening: D.7 says "hardening modulus scaled proportionally to yield".
H(T <= 773) is the true-stress chord modulus between 2 % and 10 % true strain
(elastic part removed with the L0 E(T) table) at each measured temperature;
above 773 K, H(T) = H(773) * sigma_y(T) / sigma_y(773). The 2-10 % chord is
deliberately long: the V2 flow-curve post-mortem (D-V2-22) showed that a short
chord off the knee produces a tangent above E and a singular return map.

Outputs (derived/d11/):
  yield_table_d11.csv, hardening_table_d11.csv          (isotropic, mean)
  yield_table_d11_mlow.csv, yield_table_d11_mhigh.csv   (m bracket)
  yield_table_d11_bd.csv, yield_table_d11_td.csv        (anisotropy bracket)
  macto-closure.json                                    (provenance + fit)
"""
import csv
import json
import math
import os

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
OUT = os.path.join(CASE, 'derived', 'd11')
XLSX = os.path.join(CASE, 'references', 'material', 'amb2022-04-macto',
                    'CHAL-AMB2022-04-MaCTO_submission_ANSWERS.xlsx')

T_ZERO = 1552.0          # solidus (D-08); J-C zero-strength temperature
T_KNEE = 773.15          # highest MaCTO temperature = anchor of the J-C arm
SIGMA_FLOOR = 5.0e6      # Pa, same floor the L0 table used at the solidus
H_FLOOR = 5.0e7          # Pa
# L0 E(T) table (derived/material/E_table.csv) - used only to strip the
# elastic part off the MaCTO total-strain columns.
E_TABLE = [(294.15, 207.5e9), (1144.15, 147.5e9), (1552.0, 20.0e9)]


def interp(table, T):
    xs = [t for t, _ in table]
    ys = [v for _, v in table]
    if T <= xs[0]:
        return ys[0]
    if T >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= T <= xs[i + 1]:
            f = (T - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + f * (ys[i + 1] - ys[i])
    raise AssertionError


def read_macto():
    """{'BD'|'TD': {T_K: {'yield': MPa, 'flow': {strain: MPa}}}} from the
    ANSWERS workbook (true macroscopic stress sheet)."""
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    header = rows[1]
    # column 1 is the 0.2 % offset yield; columns 2.. are strain levels
    strains = []
    for c in header[2:]:
        if c is None:
            break
        strains.append(abs(float(str(c).replace('% strain', '')
                                  .replace('%strain', '').replace('%', '')
                                  .strip())) / 100.0)
    out = {'BD': {}, 'TD': {}}
    for row in rows:
        label = row[0]
        if not isinstance(label, str) or '-' not in label:
            continue
        direction, temp = label.split('-')
        if direction not in out or not temp.endswith('K'):
            continue
        T = float(temp[:-1])
        flow = {}
        for s, v in zip(strains, row[2:2 + len(strains)]):
            try:
                flow[s] = float(v)          # 'N/A' cells: test not run
            except (TypeError, ValueError):
                continue
        out[direction][T] = {'yield': float(row[1]), 'flow': flow}
    for d in out:
        assert sorted(out[d]) == [298.0, 523.0, 773.0], out[d].keys()
    return out, strains


def chord_hardening(sigma_y_MPa, flow, T_K):
    """True-stress chord modulus over 2 %-10 % strain, elastic part removed."""
    E = interp(E_TABLE, T_K) / 1e6            # MPa
    s2, s10 = flow[0.02], flow[0.10]
    ep2 = 0.02 - s2 / E
    ep10 = 0.10 - s10 / E
    return (s10 - s2) / (ep10 - ep2)          # MPa


def jc_at(m, T, sigma_anchor, T_anchor):
    x = (T - T_anchor) / (T_ZERO - T_anchor)
    x = min(max(x, 0.0), 1.0)
    return sigma_anchor * (1.0 - x ** m)


def fit_m(points, sigma_298, T_298):
    """Least-squares m for the 298-anchored one-parameter family."""
    def sse(m):
        return sum((jc_at(m, T, sigma_298, T_298) - s) ** 2
                   for T, s in points)
    lo, hi = 0.05, 6.0
    for _ in range(200):                       # golden-section
        a = hi - (hi - lo) * 0.6180339887
        b = lo + (hi - lo) * 0.6180339887
        if sse(a) < sse(b):
            hi = b
        else:
            lo = a
    m = 0.5 * (lo + hi)
    return m, sse(m)


def exact_m(T, sigma, sigma_298, T_298):
    """m that reproduces one measured point exactly."""
    x = (T - T_298) / (T_ZERO - T_298)
    return math.log(1.0 - sigma / sigma_298) / math.log(x)


def build_curve(sigma_lowT, H_lowT, m, tag):
    """[(T_K, sigma_Pa, H_Pa, source)] over the full solver range."""
    rows = []
    Ts = sorted(sigma_lowT)
    rows.append((293.15, sigma_lowT[Ts[0]], H_lowT[Ts[0]],
                 'MaCTO 298 K value held to 293.15 K (table floor)'))
    for T in Ts:
        rows.append((T + 0.15, sigma_lowT[T], H_lowT[T],
                     f'MaCTO {tag} 0.2 % offset yield at {T:.0f} K, as-is'))
    s773, h773 = sigma_lowT[773.0], H_lowT[773.0]
    T = 798.15
    while T < T_ZERO:
        s = max(jc_at(m, T, s773, T_KNEE), SIGMA_FLOOR / 1e6)
        h = max(h773 * s / s773, H_FLOOR / 1e6)
        rows.append((T, s, h, f'J-C power softening m={m:.4f}, '
                              f'T_zero={T_ZERO:.0f} K (D.7)'))
        T += 25.0
    rows.append((T_ZERO, SIGMA_FLOOR / 1e6, H_FLOOR / 1e6,
                 'floor at solidus'))
    return rows


def write_pair(name, rows, kind):
    os.makedirs(OUT, exist_ok=True)
    idx = 1 if kind == 'yield' else 2
    path = os.path.join(OUT, name)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['T', 'value', 'source'])
        for r in rows:
            w.writerow([round(r[0], 2), round(r[idx] * 1e6, 1), r[3]])
    return path


def main():
    macto, strains = read_macto()
    T_298 = 298.15

    directions = {}
    for d in ('BD', 'TD'):
        sy = {T: macto[d][T]['yield'] for T in macto[d]}
        H = {T: chord_hardening(sy[T], macto[d][T]['flow'], T + 0.15)
             for T in macto[d]}
        directions[d] = {'yield_MPa': sy, 'hardening_MPa': H}

    mean_sy = {T: 0.5 * (directions['BD']['yield_MPa'][T]
                         + directions['TD']['yield_MPa'][T])
               for T in (298.0, 523.0, 773.0)}
    mean_H = {T: 0.5 * (directions['BD']['hardening_MPa'][T]
                        + directions['TD']['hardening_MPa'][T])
              for T in (298.0, 523.0, 773.0)}

    pts = [(523.15, mean_sy[523.0]), (773.15, mean_sy[773.0])]
    m, sse = fit_m(pts, mean_sy[298.0], T_298)
    m_low = exact_m(523.15, mean_sy[523.0], mean_sy[298.0], T_298)
    m_high = exact_m(773.15, mean_sy[773.0], mean_sy[298.0], T_298)

    paths = {}
    rows = build_curve(mean_sy, mean_H, m, 'BD/TD mean')
    paths['yield'] = write_pair('yield_table_d11.csv', rows, 'yield')
    paths['hardening'] = write_pair('hardening_table_d11.csv', rows, 'hard')
    for tag, mm in (('mlow', m_low), ('mhigh', m_high)):
        r = build_curve(mean_sy, mean_H, mm, 'BD/TD mean')
        paths[f'yield_{tag}'] = write_pair(f'yield_table_d11_{tag}.csv',
                                           r, 'yield')
        paths[f'hardening_{tag}'] = write_pair(
            f'hardening_table_d11_{tag}.csv', r, 'hard')
    for d in ('BD', 'TD'):
        r = build_curve(directions[d]['yield_MPa'],
                        directions[d]['hardening_MPa'], m, d)
        paths[f'yield_{d.lower()}'] = write_pair(
            f'yield_table_d11_{d.lower()}.csv', r, 'yield')
        paths[f'hardening_{d.lower()}'] = write_pair(
            f'hardening_table_d11_{d.lower()}.csv', r, 'hard')

    anis = {str(T): round(abs(directions['BD']['yield_MPa'][T]
                              - directions['TD']['yield_MPa'][T])
                          / mean_sy[T], 4) for T in mean_sy}
    prov = {
        'schema_version': 'ambench.d11-macto-closure/1',
        'decision': 'D-11 / PREREQUISITES.md D.7',
        'source': {
            'file': os.path.relpath(XLSX, CASE),
            'sheet': 'true macroscopic stress, 0.2 % offset yield column',
            'note': 'AMB2022-04 specimens machined from a reserved '
                    'AMB2018-01 bridge (PREREQUISITES.md note on MaCTO)'},
        'measured_MPa': {d: {str(T): directions[d]['yield_MPa'][T]
                             for T in (298.0, 523.0, 773.0)}
                         for d in ('BD', 'TD')},
        'isotropic_mean_MPa': {str(T): mean_sy[T] for T in mean_sy},
        'hardening_chord_MPa': {
            d: {str(T): round(directions[d]['hardening_MPa'][T], 1)
                for T in (298.0, 523.0, 773.0)} for d in ('BD', 'TD')},
        'hardening_definition': 'true-stress chord over 2-10 % strain, '
                                'elastic part removed with the L0 E(T); '
                                'long chord avoids the D-V2-22 H > E trap',
        'jc_fit': {
            'form_below_773K': 'MaCTO three points interpolated as-is',
            'form_above_773K': 'sigma_y(773) * [1 - ((T-773)/(T_zero-773))^m]',
            'T_zero_K': T_ZERO,
            'm_fitted': round(m, 4),
            'm_low_523K_exact': round(m_low, 4),
            'm_high_773K_exact': round(m_high, 4),
            'residual_sse_MPa2': round(sse, 2),
            'fit_procedure': 'same one-parameter family anchored at the '
                             '298 K point, least-squares over the 523 and '
                             '773 K points; no new free parameter'},
        'propagated_yield_uncertainty': {
            'bd_td_relative_spread': anis,
            'm_bracket': [round(m_low, 4), round(m_high, 4)],
            'sigma_y_at_1273K_MPa': {
                'm_fitted': round(jc_at(m, 1273.15, mean_sy[773.0], T_KNEE), 1),
                'm_low': round(jc_at(m_low, 1273.15, mean_sy[773.0], T_KNEE), 1),
                'm_high': round(jc_at(m_high, 1273.15, mean_sy[773.0],
                                      T_KNEE), 1)},
            'note': 'the run-level propagated uncertainty is measured by '
                    'rerunning the frozen case on the m_low / m_high arms '
                    'and reading M1/M2 (HIGH-T-EXTRAPOLATION-NOTE 2b)'},
        'replaces': 'derived/material/yield_table.csv + hardening_table.csv '
                    '(PROVISIONAL-L0, labelled "D-11 MaCTO tables replace '
                    'this")',
        'outputs': {k: os.path.relpath(v, CASE) for k, v in paths.items()},
    }
    with open(os.path.join(OUT, 'macto-closure.json'), 'w') as f:
        json.dump(prov, f, indent=1)
        f.write('\n')

    print(f'MaCTO yield (MPa)  BD {directions["BD"]["yield_MPa"]}')
    print(f'                   TD {directions["TD"]["yield_MPa"]}')
    print(f'isotropic mean     {mean_sy}')
    print(f'hardening chord    mean {mean_H}')
    print(f'm fitted {m:.4f}  (bracket {m_low:.4f} .. {m_high:.4f}), '
          f'sse {sse:.1f} MPa^2')
    for T in (873.15, 1073.15, 1173.15, 1273.15, 1373.15):
        print(f'  sigma_y({T - 273.15:.0f} C) = '
              f'{jc_at(m, T, mean_sy[773.0], T_KNEE):7.1f} MPa   '
              f'[m_low {jc_at(m_low, T, mean_sy[773.0], T_KNEE):6.1f} / '
              f'm_high {jc_at(m_high, T, mean_sy[773.0], T_KNEE):6.1f}]')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
