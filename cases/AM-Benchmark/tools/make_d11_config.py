"""D-11 material config: the L0 wiring with the MaCTO sigma_y closure swapped in.

Everything except yield_table / hardening_table is copied verbatim from
derived/material/amb_material_config_L0.json, so the D-11 matrix differs from
the validated L0 pipeline in exactly one registered respect: the D.7 sigma_y
high-temperature closure replaces the PROVISIONAL-L0 linear ramp.

Table paths in the L0 config are absolute and point at whatever checkout
generated them; they are re-resolved here against THIS checkout so the config
is portable between runtimes.

Usage: python make_d11_config.py [--arm mean|mlow|mhigh|bd|td]
Output: derived/d11/amb_material_config_D11_<arm>.json
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
MAT = os.path.join(CASE, 'derived', 'material')
D11 = os.path.join(CASE, 'derived', 'd11')

ARMS = {
    'mean': ('yield_table_d11.csv', 'hardening_table_d11.csv'),
    'mlow': ('yield_table_d11_mlow.csv', 'hardening_table_d11_mlow.csv'),
    'mhigh': ('yield_table_d11_mhigh.csv', 'hardening_table_d11_mhigh.csv'),
    'bd': ('yield_table_d11_bd.csv', 'hardening_table_d11_bd.csv'),
    'td': ('yield_table_d11_td.csv', 'hardening_table_d11_td.csv'),
}


def main(arm):
    src = os.path.join(MAT, 'amb_material_config_L0.json')
    cfg = json.load(open(src))

    # re-resolve the L0 table paths against this checkout
    for key in ('k_table_solid', 'cp_table_solid', 'k_table_powder',
                'cp_table_powder', 'E_table', 'alpha_table',
                'poisson_table'):
        cfg[key] = os.path.join(MAT, os.path.basename(cfg[key]))
        if not os.path.exists(cfg[key]):
            raise SystemExit(f'missing L0 table {cfg[key]}; run '
                             f'tools/make_material_tables.py first')

    y, h = ARMS[arm]
    cfg['yield_table'] = os.path.join(D11, y)
    cfg['hardening_table'] = os.path.join(D11, h)
    for p in (cfg['yield_table'], cfg['hardening_table']):
        if not os.path.exists(p):
            raise SystemExit(f'missing {p}; run '
                             f'tools/make_d11_yield_tables.py first')

    cfg['_comment'] = (
        'AMB2018-01 D-11 matrix config. Identical to the L0 wiring except '
        f'yield/hardening, which use the D.7 MaCTO closure (arm: {arm}). '
        'E/alpha/poisson remain the L0 PROVISIONAL tables: D-11 freezes the '
        'sigma_y high-T treatment only; E/alpha/nu extrapolation is D-V2-17 '
        '(see verification/HIGH-T-EXTRAPOLATION-NOTE.md).')
    cfg['_d11_arm'] = arm
    cfg['_d11_yield_provenance'] = 'derived/d11/macto-closure.json'

    os.makedirs(D11, exist_ok=True)
    dst = os.path.join(D11, f'amb_material_config_D11_{arm}.json')
    with open(dst, 'w') as f:
        json.dump(cfg, f, indent=1)
        f.write('\n')
    print(f'wrote {dst}')
    return dst


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=sorted(ARMS), default='mean')
    main(ap.parse_args().arm)
