# D-11 campaign B archives (shared-mesh N sweep)

Primary evidence for the 2026-08-07 campaign-B verdict (issue IET-7): N does
NOT converge on the shared mesh — L1 M1/M2 = 68.5 % / 36.0 % (N50 vs N25) and
77.5 % / 61.3 % (N25 vs N10), both far above the pre-registered 5 % dual gate
→ true aggregation non-convergence → D.5-style interval publication; N is not
frozen.

Provenance:

- Members: N ∈ {50, 25, 10} × T_cut = 1000 °C, arm = mean, `fixed_power`,
  all on the SHARED 0.1 mm z mesh `derived/meshes/amb_d11_N5.inp` (90 720
  elements; N only controls activation grouping: 10/5/2 rows per
  computational layer). NOT comparable to `runs/` (stage 1, per-N meshes).
- Execution: git worktree at commit `55d94ce`, serial via
  `tools/d11_run_b.sh`, 22→26 threads (hot switch validated), campaign
  finished 2026-08-06T12:41:35Z, all rc=0 (`B.done.txt`).
- `<tag>_metrics.json` / `<tag>_case.json` are the untouched
  `d11_metrics.json` / `d11_case.json` from `/home/user/work/output/d11_B/`.
- `matrix-report.{json,md}` is the harvest-time report (pre-e854aa4 tool:
  deciding column = max over legs; the 212–5262 % entries are the thin-leg
  small-denominator artefact). The verdict was read on the pre-registered
  L1-only convention; the campaign-end rerun after the T_cut band uses the
  e854aa4 tool with L1-only deciding columns.
