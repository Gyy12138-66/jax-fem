"""D-11 T_cut band: failure classifier for the `none` member (two-tier).

The `none` member runs with the relaxation flag omitted entirely, so mechanical
history is never wiped above T_cut. On the 0.1 mm shared mesh that may diverge
-- and if it does, THAT IS EVIDENCE, not an accident: it answers "is the memory
reset mechanism necessary at all". No parameter is ever tuned to rescue it.

But divergence is only evidence if the run actually got somewhere first. A
member that dies before it has experienced real activation and thermal cycling
tells us about our wiring, not about physics. Publishing a bug as a physical
result is worse than the run failing. Hence two tiers (pre-registered
2026-08-07):

  wiring-suspect        ledger <= --min-ledger steps (default 12), i.e. death
                        before real activation/thermal cycling. Report, fix,
                        re-run. MUST NOT be registered as "infeasible".
  infeasible-candidate  divergence after the build has progressed, with Newton
                        cutback / acceptance violations. Only this tier may be
                        registered, and only with evidence on disk: tail
                        residuals, Newton cutback history, ledger tail.

Usage:
  python d11_none_guard.py <member_dir> [--rc N] [--min-ledger 12]
Writes <member_dir>/none_guard.json and none_guard_evidence.txt.
Exit code 0 always -- this is a classifier, not a gate.
"""
import argparse
import json
import os
import re

LEDGER = 'thermal_energy_ledger.jsonl'
CUTBACK_RE = re.compile(r'mechanics cutback:')
BAD_RE = re.compile(r'nan|inf|diverg|failed|singular|not converge', re.I)
TAIL_LOG_LINES = 80
TAIL_LEDGER_LINES = 20


def read_tail(path, n):
    if not os.path.exists(path):
        return []
    with open(path, 'r', errors='replace') as f:
        return f.readlines()[-n:]


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, 'r', errors='replace') as f:
        return sum(1 for _ in f)


def main(member_dir, rc, min_ledger):
    log = os.path.join(member_dir, 'run.log')
    ledger = os.path.join(member_dir, LEDGER)

    ledger_steps = count_lines(ledger)
    last_step = None
    cutbacks, bad = [], []
    if os.path.exists(log):
        with open(log, 'r', errors='replace') as f:
            for line in f:
                m = re.match(r'global_step=(\d+)', line)
                if m:
                    last_step = int(m.group(1))
                if CUTBACK_RE.search(line):
                    cutbacks.append(line.rstrip())
                elif BAD_RE.search(line) and not line.startswith('global_step'):
                    bad.append(line.rstrip())

    if rc == 0:
        cls = 'ok'
        guidance = 'member completed; no registration needed'
    elif ledger_steps <= min_ledger:
        cls = 'wiring-suspect'
        guidance = (f'died at ledger step {ledger_steps} <= {min_ledger}, '
                    f'before real activation/thermal cycling. This is a '
                    f'wiring/config suspicion, NOT a physics result. Report '
                    f'and investigate; MUST NOT be registered as '
                    f'"infeasible on the fine mesh".')
    else:
        cls = 'infeasible-candidate'
        guidance = (f'diverged after real build progress (ledger step '
                    f'{ledger_steps}, last global_step {last_step}), with '
                    f'{len(cutbacks)} Newton cutback events. Eligible to be '
                    f'registered as "T_cut=none infeasible on the 0.1 mm '
                    f'shared mesh" -- which is itself evidence that the '
                    f'memory-reset mechanism is necessary. Evidence is '
                    f'written alongside; registration must cite it. Do NOT '
                    f'tune any parameter to rescue this member.')

    result = {
        'schema_version': 'ambench.d11-none-guard/1',
        'member_dir': member_dir,
        'run_rc': rc,
        'classification': cls,
        'min_ledger_steps': min_ledger,
        'ledger_steps': ledger_steps,
        'last_global_step': last_step,
        'newton_cutback_events': len(cutbacks),
        'suspicious_log_lines': len(bad),
        'guidance': guidance,
        'evidence_file': ('none_guard_evidence.txt'
                          if cls == 'infeasible-candidate' else None),
        'pre_registered': '2026-08-07 two-tier none guardrail; a member that '
                          'dies before real activation is a bug suspicion, '
                          'not a physics finding',
    }

    if cls == 'infeasible-candidate':
        ev = os.path.join(member_dir, 'none_guard_evidence.txt')
        with open(ev, 'w') as f:
            f.write(f'# D-11 none-member divergence evidence\n')
            f.write(f'# classification: {cls}\n')
            f.write(f'# ledger steps: {ledger_steps}, '
                    f'last global_step: {last_step}, rc: {rc}\n\n')
            f.write(f'== Newton cutback history ({len(cutbacks)} events) ==\n')
            f.write('\n'.join(cutbacks[-200:]) + '\n\n')
            f.write(f'== suspicious log lines ({len(bad)}) ==\n')
            f.write('\n'.join(bad[-100:]) + '\n\n')
            f.write(f'== run.log tail ({TAIL_LOG_LINES} lines) ==\n')
            f.writelines(read_tail(log, TAIL_LOG_LINES))
            f.write(f'\n== ledger tail ({TAIL_LEDGER_LINES} lines) ==\n')
            f.writelines(read_tail(ledger, TAIL_LEDGER_LINES))
        print(f'wrote {ev}')

    out = os.path.join(member_dir, 'none_guard.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=1)
        f.write('\n')
    print(f'[{cls}] ledger_steps={ledger_steps} last_step={last_step} '
          f'cutbacks={len(cutbacks)} rc={rc}')
    print(guidance)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('member_dir')
    ap.add_argument('--rc', type=int, default=1)
    ap.add_argument('--min-ledger', type=int, default=12)
    a = ap.parse_args()
    raise SystemExit(main(a.member_dir, a.rc, a.min_ledger))
