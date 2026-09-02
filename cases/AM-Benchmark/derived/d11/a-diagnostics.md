## D-11 diagnostic A: metric resolution vs aggregation physics

All members block-averaged onto the coarsest ruler (1.0 mm z bands = the N=50 computational layer).

> Block averaging is NOT hiding the difference: it is the common ruler the coarsest member can actually resolve. A coarse member cannot answer a sub-cell question, so this is the correct like-for-like comparison.

> But N=50 gives only 5 bands over the leg. That ruler is blunt, so a PASS on it is WEAK evidence. Strong evidence requires deliverable B: every member on the same 0.1 mm mesh.

### Deciding leg L1_thick (pre-registered)

| pair | M1 raw (stage 1) | M1 coarsened | M2 raw | M2 coarsened | passes 5 % |
|---|---|---|---|---|---|
| N50_vs_N25 | 26.70 % | 27.45 % | 16.90 % | 21.32 % | NO |
| N25_vs_N10 | 44.54 % | 43.32 % | 46.48 % | 47.86 % | NO |

### Gate 2 (pre-registered: both drop >= 50 %)

- M1 drop -2.8 %, M2 drop -26.2 %
- **significant improvement: NO**

### L2 / L3 (absolute differences, not deciding)

| pair | leg | max band |d sigma_xx| (MPa) | |d Mroot| (N*mm) |
|---|---|---|---|
| N50_vs_N25 | L2_thin | 6.89 | 24.354 |
| N50_vs_N25 | L3_med | 1.27 | 3.974 |
| N25_vs_N10 | L2_thin | 2.23 | 16.453 |
| N25_vs_N10 | L3_med | 2.39 | 2.541 |

### Leg-root moment on the common ruler (N*mm)

| N | raw Mroot | coarsened Mroot |
|---|---|---|
| 50 | -1141.0 | -1141.0 |
| 25 | -976.0 | -940.5 |
| 10 | -666.3 | -636.1 |

### Thermal history per N (exposure recorder)

Peak temperature is given in BOTH conventions, because the two used in the thread differ by exactly the plate temperature (347.05 K). These are peaks over CONSOLIDATED PART cells; the thermal-probe table quotes the global nodal maximum, which is higher because it includes powder and free-surface nodes.

| N | peak T abs (K) | peak T rise (K) | cells ever > T_cut | max exposure > T_cut (s) | comp layers | cycles / mm build |
|---|---|---|---|---|---|---|
| 50 | 1644.7 | 1297.7 | 0.56 % | 59.7 | 7 | 1.00 |
| 25 | 1633.2 | 1286.1 | 0.54 % | 44.8 | 14 | 2.00 |
| 10 | 1432.6 | 1085.6 | 0.15 % | 23.9 | 35 | 5.00 |

> thermal-only; dwell truncated, so exposure is a LOWER bound on a full-dwell run (material keeps cooling past the truncation). Comparison across N is like-for-like.
