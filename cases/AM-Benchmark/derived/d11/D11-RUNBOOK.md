# D-11 matrix runbook — 15 runs, T_cut × N

Status: **implementation + smoke debug complete; production runs NOT executed.**
Handoff per the 2026-08-04 division-of-labour update (Opus5 implements and
smoke-debugs, the results-bearing runs are executed after code review).

Decision of record: `PREREQUISITES.md` D.7 (D-11, approved). Two-cycle gate
passed 2026-08-03 (`tools/d11_two_cycle_gate.py`, `derived/d11/gate-test.json`),
so the matrix is unblocked.

Two things were found during implementation that the decision owner should
settle before the campaign starts. Both are in §0 and §3a. Everything else is
mechanical.

---

## 0. Finding 1 — the mainline yield table was still the L0 placeholder

D.7 specifies the matrix runs on the D-11 σ_y closure, but
`derived/material/yield_table.csv` was still the L0 placeholder — it says
verbatim `PROVISIONAL-L0 (D-11 MaCTO tables replace this)` and decays linearly
to 50 MPa at 1273 K. Running the T_cut sweep on that table would have measured
the sensitivity of a placeholder, not of the approved model, so the closure was
built first (`tools/make_d11_yield_tables.py`, provenance in
`macto-closure.json`).

Effect at the sweep point that matters most: **σ_y(1000 °C) = 277 MPa** under
the D.7 model versus **50 MPa** under the L0 placeholder. That is not a tuning
change — it is the model D.7 approved — but it is a factor of 5.5 and it is the
first thing to check on review.

Fit result: **m = 1.400**, bracket [0.969 (523 K exact), 1.683 (773 K exact)].
The residual is large (SSE 7548 MPa²) because IN625 has a genuine yield dip at
523 K that a monotone J-C form cannot represent. That is data, not a coding
error, and it is why the m bracket is carried rather than a point value.

Everything else in the run wiring is copied from the validated L0 pipeline.

## 1. Files

| File | Role |
|---|---|
| `tools/make_d11_yield_tables.py` | MaCTO σ_y(T) + hardening closure (D.7), m fit, m/BD-TD brackets |
| `tools/make_d11_mesh.py` | one-leg-period sub-domain mesh, per N |
| `tools/make_d11_config.py` | L0 material config with the D-11 σ_y swapped in |
| `tools/run_d11_case.sh` | one matrix member |
| `tools/d11_thermal_probe.sh` | **pre-flight**: does the model reach T_cut? |
| `tools/d11_metrics.py` | per-run M1/M2/M3 raw quantities from the final VTU |
| `tools/d11_matrix_report.py` | dual metric, T_cut band, freeze recommendation |
| `derived/d11/macto-closure.json` | σ_y provenance + fit residuals (committed) |
| `derived/d11/yield_table_d11*.csv` | generated tables (committed) |
| `derived/meshes/amb_d11_N*_summary.json` | mesh summaries (committed; `.inp` regenerable) |

## 2. Run it

```bash
cd cases/AM-Benchmark
PY=/home/user/miniconda3/envs/jax-fem-gpu/bin/python

# one-time inputs
$PY tools/make_d11_yield_tables.py          # σ_y closure + brackets
$PY tools/make_d11_config.py --arm mean     # material config
for N in 50 25 10; do $PY tools/make_d11_mesh.py $N; done

# PRE-FLIGHT (minutes) - confirm the T_cut axis is live before committing days
bash tools/d11_thermal_probe.sh 50 25 10

# the 15 members  (T_cut in °C; "none" omits the relaxation flag entirely)
for N in 50 25 10; do
  for T in 800 900 1000 1100 none; do
    OUT_ROOT=/home/user/work/output/d11/N${N}_T${T} \
      bash tools/run_d11_case.sh $N $T
    $PY tools/d11_metrics.py /home/user/work/output/d11/N${N}_T${T}
  done
done

$PY tools/d11_matrix_report.py /home/user/work/output/d11
# -> derived/d11/matrix-report.{json,md}
```

Environment: WSL conda `jax-fem-gpu`, `XLA_PLATFORM=gpu` is the default. The
linear solve is MKL **pardiso on CPU** (87 % of L0 wall time), so the GPU is not
the bottleneck — `MKL_NUM_THREADS` is. Set it explicitly when running members
concurrently.

**Do not** rely on `nohup` for long runs (PREREQUISITES.md ops note); clean up
by exact PID, never a broad `pkill`.

## 3a. Finding 2 — the scan-to-sub-domain mapping decides whether T_cut does anything

T_cut has **no effect unless material crosses it on cooling** (the relaxation
mask is literally `T_quad >= T_cut`). So if peak temperature stays below the
sweep ladder {1073, 1173, 1273, 1373} K, the entire T_cut axis collapses to a
single answer and the matrix measures nothing on it.

The sub-domain is 14 mm of a 75 mm part, so L0's aggregated scan cannot be
carried over unchanged: conserving *energy* per layer and conserving
*instantaneous power* are no longer the same choice, and they differ by ~6×.
Absorbed energy per computational layer is `N × 60.45 J` under every convention
below (power is always back-solved from it) — they differ only in how that
energy is spread in time, hence in peak temperature.

Measured, thermal-only probe over the full build height
(`tools/d11_thermal_probe.sh`; the validated L0 run peaks at 2630 K):

| convention | N=50 | N=25 | N=10 | power | ladder covered |
|---|---|---|---|---|---|
| `energy` | 616 K | — | 421 K | ~7.6 W | **none, at any N** |
| `power` | 2571 K | 1677 K | 916 K | 61/30/12 W | none at N=10 |
| `l0_density` | 1904 K | 2505 K | 1321 K | 41/61/24 W | N=10 misses 1100 °C |
| **`fixed_power`** (default) | 2052 K | 2001 K | 1707 K | **40.8 W, constant** | **all, at every N** |

`fixed_power` refines dt with N so the scan phase lands exactly on the slice's
share of the scan work (`640 s × N/50 × 14/75`). Consequences: laser power is
constant across the sweep and inside L0's own band, peak T is nearly constant
across N, and steps per computational layer is constant (87) — so the step
count scales with the number of computational layers, i.e. **refining N
genuinely costs more**, which is honest but expensive (see §3).

The first convention implemented was `energy`, and it is verifiably degenerate.
All four are kept switchable via `D11_SCAN_CONVENTION` so the reviewer can
reproduce the table rather than take it on trust.

**This is a modelling choice, not a bug fix.** It was not specified in D.7
because D.7 did not anticipate that truncating to a sub-domain forces it. It
belongs to the decision owner.

## 3. Runtime — measured, and it is large

Smoke measurement (N=50 sub-domain, 20 160 cells, 84 steps, 21 mechanics
solves, `MKL_NUM_THREADS=8`, **with another agent's GPU job competing for
CPU**): 484 s wall.

Extrapolated per production member under `fixed_power`, scaling pardiso as
n^1.5 and counting linear iterations:

| N | cells | build steps | est. wall / member | × 5 T_cut |
|---|---|---|---|---|
| 50 | 20 160 | 609 | ~0.8 h | ~4 h |
| 25 | 28 000 | 1 218 | ~2.4 h | ~12 h |
| 10 | 51 520 | 3 045 | ~14 h | ~70 h |

**~86 h sequential.** Order-of-magnitude only — one measured point, and it was
measured under CPU contention; a quiet machine with more MKL threads should be
materially faster. The `energy` convention is ~3× cheaper but degenerate, so
the cost is the price of a live T_cut axis.

Suggested order so a truncated campaign still decides something:

1. **N convergence** — `T_cut=1000 × N∈{50,25,10}` (3 runs, ~17 h). This alone
   freezes N. The expensive N=10 member is unavoidable here: proving N=25 is
   converged requires differencing it against N=10.
2. **T_cut band on the frozen N** (4 runs). This alone decides the freeze rule.
3. **Yield-uncertainty arms** — `D11_ARM=mlow` and `mhigh` at the frozen
   (N, T_cut=1000) (2 runs). Without these the freeze rule has no like-for-like
   denominator and `d11_matrix_report.py` returns `T_cut: undetermined` **by
   design** rather than guessing.
4. The remaining 8 members complete the matrix.

Members are independent — run them concurrently if RAM allows (~31 GB total on
this box, one L0-scale member peaked ~6 GB).

## 3b. PRE-REGISTERED CONVENTIONS (agreed 2026-08-04, binding before B)

Registered before the B production runs so they cannot be chosen after seeing
results:

1. **Deciding metric is the load-bearing thick leg L1 only** (M1 and M2 both
   < 5 %, unchanged). L2/L3 are reported as **absolute** differences
   (MPa / N·mm), never relative, and do not enter the verdict — their signal is
   5-20x smaller so relative metrics on them are noise (stage 1 produced 2320 %
   on L3 from a moment of 1.4 N·mm).
2. **Gate 2 quantified**: against stage 1 (L1: M1 26.7 %, M2 16.9 %), the
   coarsened common-ruler metrics must **both fall >= 50 %** to count as
   "significant improvement".
3. **If B still fails the 5 % gate on both pairs** -> true aggregation
   non-convergence; go straight to D.5-style interval publication
   ("aggregation coarser than X gives moment deviation Y %"). Option C is
   discarded: thermal cycles per unit build height (7/14/35 over the same
   7 mm) is what layer lumping IS, and no mesh or scan convention removes it.
4. **`fixed_power` stays**. Re-enabling any other convention is a convention
   re-approval and needs its own record.

**Convention change registered for D.7**: the N sweep moved from "mesh z
resolution varies with N" to "one fixed 0.1 mm mesh, N controls only the
deposition grouping". This makes M1/M2 differences attributable to aggregation
alone. Side effect: absolute quantities (moment, displacement) from B are
shifted relative to the stage-1 values and are **not** directly comparable.

### The shared mesh

`python tools/make_d11_mesh.py 5` -> `amb_d11_N5.inp`: 0.1 mm z rows, 70 build
rows, **90 720 elements / 98 154 nodes**. Select it with `D11_MESH=<path>`,
which overrides **only** the mesh; layer thickness, layer count, dt, power and
mechanics cadence still derive from N.

`run_d11_case.sh` asserts the alignment and refuses to run if the computational
layer is not an integer number of mesh rows. Verified: 0.1 mm mesh gives
**10 / 5 / 2 rows** for N = 50/25/10, and the 0.2 mm mesh at N=25 gives 2.5
rows and is **rejected with exit 3** — that is exactly the [3,2,3,2,…] uneven
grouping the 0.2 mm shared mesh would have produced.

## 4. What the metrics compute

D.7 verbatim, with the implementation choices stated:

- **M1** `‖σxx_Ni(z) − σxx_Ni+1(z)‖₂ / ‖σxx_Ni+1(z)‖₂` on the leg mid-plane
  vertical line, both profiles interpolated to a common 200-point z grid.
- **M2** `|Mroot_Ni − Mroot_Ni+1| / |Mroot_Ni+1|` with
  `Mroot = ∫ σxx (z − z̄) dA` — the x-normal section at the leg mid-plane,
  lever arm about the section area centroid, integrated over the legs band
  z ∈ [0, 5] mm (the leg proper). A full-height variant is also stored.
- **Converged when M1 < 5 % AND M2 < 5 %**, both required.
- Three legs live in the sub-domain (L1 thick, L2 thin, L3 medium). D.7 says
  "the leg mid-plane" without naming one, so the tool reports all three and the
  **deciding value is the max over legs** — the conservative reading. Per-leg
  values are in the JSON if the reviewer prefers a different convention.
- **M3** (top-surface u_z profile, same L2 form) is **advisory, not part of the
  freeze rule**. It exists because M1/M2 are both stress-type and a stress field
  can converge while the height-integrated displacement has not
  (`verification/HIGH-T-EXTRAPOLATION-NOTE.md` §5). Pure computation — it
  touches no measured value.

## 5. Assumptions this implementation makes — all need review

1. **Sub-domain truncation.** x ∈ [0, 14] mm is one leg-group period
   (`feature_id_map`: L1 at 0.0, L4 at 14.0). Both cut faces happen to be
   benign — x = 0 is the part's real free end and x = 14 sits in the powder gap
   between L3 and L4 through the legs band — but the solver has **no side-face
   symmetry BC**, so both are traction-free and adiabatic. Identical for all 15
   members, which is what relative metrics need; it does mean absolute stresses
   here are not the full-part values.
2. **Uniform layer clock.** One computational layer = N × 52 s (D-10 legs-band
   time) for every band. Same uniform-clock deviation L0 registered.
3. **Energy.** Absorbed energy per physical layer for this 14 mm slice is
   60.45 J, derived from the real scan (leg area / hatch / speed × 195 W ×
   0.62). Cross-checks against L0 to 3 significant figures via the
   cross-section ratio 40/255, and is conserved exactly at every N and under
   every scan convention (verified: 3022.5 / 1511.3 / 604.5 J per
   computational layer for N = 50/25/10).
4. **Mechanics cadence.** A fixed **6 solves per computational layer** for
   every N (`--mechanics-every` derived, override `D11_MECH_PER_LAYER`). A
   fixed step interval would give coarse-N runs more solves per deposited layer
   than fine-N runs and confound the sweep; a fixed wall-clock interval breaks
   once dt varies with N under `fixed_power`.
5. **σ_y anisotropy.** The solver's J2 model is isotropic and cannot carry the
   BD/TD anisotropy D.7 asks to keep. The isotropic table is the BD/TD **mean**
   and the BD/TD spread is carried as part of the propagated yield uncertainty
   (`--arm bd` / `--arm td` reproduce either limb). This is the one place the
   implementation could not follow D.7 literally.
6. **m fit.** D.7 says m is "fitted to the MaCTO three-point trend (no new free
   parameter)" without fixing the procedure. Implemented as: the same
   one-parameter family anchored at the 298 K point, least-squares over the two
   remaining measured points. See §0.
7. **E/α/ν stay the L0 PROVISIONAL tables.** D-11 covers σ_y only; the E/α/ν
   high-T extrapolation is D-V2-17. Note α is currently a *constant* 1.28e-5,
   so the mainline does **not** have the D-V2-18 secant-vs-instantaneous bug (a
   constant secant coefficient is its own instantaneous value) — but it also has
   no temperature dependence at all, which will move the absolute T_cut band
   once the real curve lands.

## 6. Free interpretation the reviewer gets

`HIGH-T-EXTRAPOLATION-NOTE.md` §1: the T_cut = 800 °C member has **zero**
extrapolation exposure — no extrapolated σ_y form above 1073 K, no E
extrapolation (data ends 1144 K), no α extrapolation (data ends 1200 K). So if
the 800 °C vs 1000 °C band lands inside the threshold, the σ_y high-T form, the
E extrapolation and the α extrapolation are all shown irrelevant to the result
in one stroke. `d11_matrix_report.py` reports this comparison explicitly under
`zero_exposure_800C`.

## 7. Proposed D.7 addendum (ledger edit deliberately NOT made here)

For the decision owner to land, if the implementation is accepted:

- σ_y closure realised as `derived/d11/yield_table_d11.csv`, provenance in
  `macto-closure.json`; m = 1.400 (bracket 0.969–1.683); isotropic BD/TD mean
  with the spread carried as propagated yield uncertainty.
- **Scan convention for sub-domain runs** (§3a) — D.7 needs a sentence, since
  the choice decides whether the T_cut axis exists at all.
- Propagated MaCTO yield uncertainty is **measured in metric units** by
  rerunning the frozen case on the m_low / m_high arms, so the freeze rule's
  "band smaller than the yield uncertainty" test is like-for-like.
- M1/M2 deciding value = max over the three legs in the sub-domain.
- Sub-domain cut faces are traction-free (no symmetry BC available).

## 8. Smoke evidence

Three short members ran end-to-end on GPU: `N50/T_cut=1000`, `N50/none`
(exercises the no-relaxation branch — the flag is omitted, not set to a
sentinel) and `N25/T_cut=1000`. All exit 0, all wrote VTUs carrying the fields
the metrics tool consumes, and `d11_metrics.py` / `d11_matrix_report.py` both
ran to completion on the results and produced the report tables.

Verified during smoke:
- T_cut is genuinely plumbed: `stress_free_temperature` = 1273.15 K on the
  `T_cut=1000` arm versus the activation temperature (347–561 K) on the `none`
  arm.
- The two arms produced *identical stresses*, which is **correct**: `eps_ref` is
  captured at activation so material is born stress-free, and T_cut therefore
  only bites on a cooling crossing. In the short smoke nothing got near 1073 K.
  This is what led to Finding 2 — chase it if you see it again in production,
  because in production it would mean the axis is dead.
- Smoke members are **short** (2 scan steps/line, 4 dwell steps) — they prove
  wiring, not physics. No production numbers exist yet.

Note for whoever shortens a run: `--layers` does **not** shorten it. The runner
derives the layer count from the mesh build height ÷ layer thickness (7 for
N=50, 14 for N=25, 35 for N=10). Use `--scan-steps-per-layer` /
`--dwell-steps-between-layers` / `--cooling-steps`, or `--max-print-layers`.
