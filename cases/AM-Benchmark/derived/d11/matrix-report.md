## D-11 matrix: dual convergence metrics

members present: 8 / 15 (+ yield-uncertainty arms)

### N convergence (deciding leg L1_thick; M1 AND M2 both < 5 %)

| T_cut | pair | M1 (L1) | M2 (L1) | M3 (advisory) | converged | M1/M2 max-over-legs (background) |
|---|---|---|---|---|---|---|
| 1000 | N50_vs_N25 | 66.63 % | 36.02 % | 230.82 % | NO | 212.38 % / 202.69 % |
| 1000 | N25_vs_N10 | 69.39 % | 61.33 % | 222.23 % | NO | 387.08 % / 5261.95 % |

### T_cut band on N = 10 (vs T_cut = 1000 C)

Reference member basis: frozen N when the dual metric licenses one; otherwise the reference member N=10 (finest lumping = closest to real layer-by-layer physics). Reporting on a reference N is NOT a freeze.

| T_cut | M1 (L1) | M2 (L1) | M3 (advisory) |
|---|---|---|---|
| 800 | 0.43 % | 0.77 % | 238.62 % |
| 900 | 0.22 % | 0.08 % | 19.76 % |
| 1100 | 0.33 % | 0.15 % | 48.65 % |
| none | 0.41 % | 0.33 % | 55.16 % |

**Pre-registered band (2026-08-07)**

- `band_M1 = max_i M1(member i vs T1000)` = 0.43 %
- `band_M2 = (max Mroot - min Mroot) / |Mroot(T1000)|` = 0.84 %
- members used: 1000, 1100, 800, 900, none
- Mroot (N*mm): 1000=-683.5, 1100=-682.5, 800=-678.3, 900=-684.0, none=-681.3
- background only, max over legs: M1 14.51 %, M2 356.04 %

### Propagated MaCTO yield uncertainty (same units)

| arm | M1 (L1) | M2 (L1) |
|---|---|---|
| mlow | 0.00 % | 0.00 % |

- `band_M2 = |Mroot(mhigh) - Mroot(mlow)| / |Mroot(mean)|` = n/a
- both arms present: False

### Recommendation (proposal, not a freeze)

- N = 10
- T_cut = undetermined
- band measured, but the freeze/interval decision is pre-registered as AWAITING BOTH ARMS: m_low and m_high must run on this same shared mesh at N=10, T_cut=1000. Stage-1 arm values are on the old per-N meshes and must not be compared across that convention change.
