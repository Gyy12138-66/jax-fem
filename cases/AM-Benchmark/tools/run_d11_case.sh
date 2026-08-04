#!/usr/bin/env bash
# ============================================================================
# D-11 matrix member runner (PREREQUISITES.md D.7, decision D-11).
#
#   bash run_d11_case.sh <N> <T_cut_C|none> [extra runner args...]
#
# e.g.  bash run_d11_case.sh 25 1000
#       bash run_d11_case.sh 10 none
#
# Env knobs: PYTHON_BIN, XLA_PLATFORM (default gpu), OUT_ROOT, RUN_ID,
#            D11_ARM (mean|mlow|mhigh|bd|td), MKL_NUM_THREADS.
#
# ---------------------------------------------------------------------------
# WIRING RATIONALE - every number below is derived, not tuned.
#
# CLOCK.  One computational layer = N physical layers = N * 52 s (D-10
#   legs-band layer clock, uniform across bands; the same uniform-clock
#   deviation L0 registered). dt = 20 s, so steps/layer = N * 2.6 and the
#   TOTAL number of build steps is 910 for EVERY N -- the sub-domain's
#   simulated build time (350 * 52 s) does not depend on the lumping. That
#   is what makes the N-sweep a clean lumping study: same physical clock,
#   same step count, same mechanics cadence; only the z-discretisation and
#   the deposition granularity change.
#
# ENERGY. Real absorbed energy per physical layer for THIS 14 mm slice:
#   laser-on time = leg area / (hatch * speed) = (5.0+0.5+2.5) mm * 5.0 mm
#   / (0.10 mm * 800 mm/s) = 0.5 s;  E_phys = 195 W * 0.62 * 0.5 s = 60.45 J.
#   Per computational layer: N * 60.45 J, delivered over the scan phase, so
#   laser_power = N * 60.45 / (absorptivity * t_scan). Cross-check against
#   L0: the slice/full-part cross-section ratio is 40/255 = 0.157 and
#   0.157 * 19.34 kJ = 3.04 kJ = 50 * 60.45 J. Consistent by construction.
#
# SCAN PHASE. 4 serpentine hatch lines over the slice footprint (x 0-14 mm,
#   y +-2.5 mm), S steps per line, S = N/6.25 rounded to 8/4/2 for N =
#   50/25/10 -> t_scan = 640/320/160 s (~25 % of the layer clock, matching
#   L0's 640/2660). Instantaneous power is an aggregation construct at
#   dt = 20 s lumping (peaks are sub-grid at part scale; consolidation is
#   activation-driven), exactly as registered for L0 rev 3.
#
# MECHANICS. --mechanics-every 13 = one mechanics solve per 260 s of
#   simulated time for every N (76 solves per run, identical cost and
#   identical physical cadence across the sweep). Acceptance quartet is the
#   Kaess-golden-validated one L0 rev 3 wired (rel-tol 5e-5 / abaqus / 50 /
#   line search) - the default 1e-9/1e-11 is the known j2 stagnation trap.
#
# T_cut. --stress-relaxation-temperature <T_cut_K>; the "none" arm OMITS the
#   flag (schema default None => relaxation disabled), which is the correct
#   encoding of "no high-temperature history reset at all".
# ============================================================================
set -euo pipefail

N="${1:?usage: run_d11_case.sh <N> <T_cut_C|none> [extra args]}"
TCUT_C="${2:?usage: run_d11_case.sh <N> <T_cut_C|none> [extra args]}"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CASE_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/miniconda3/envs/jax-fem-gpu/bin/python}"
D11_ARM="${D11_ARM:-mean}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
TAG="N${N}_T${TCUT_C}"
OUT_ROOT="${OUT_ROOT:-/home/user/work/output/d11_${TAG}_${RUN_ID}}"
MESH="${CASE_DIR}/derived/meshes/amb_d11_N${N}.inp"
CONFIG="${CASE_DIR}/derived/d11/amb_material_config_D11_${D11_ARM}.json"

[[ -f "${MESH}" ]] || { echo "d11: missing mesh ${MESH}; run: python ${SCRIPT_DIR}/make_d11_mesh.py ${N}" >&2; exit 2; }
[[ -f "${CONFIG}" ]] || { echo "d11: missing config ${CONFIG}; run: python ${SCRIPT_DIR}/make_d11_config.py --arm ${D11_ARM}" >&2; exit 2; }

# ---- derived schedule (see rationale above) --------------------------------
PHYS_LAYERS=350                 # build to the overhang-band top (z = 7.0 mm)
LAYER_CLOCK_S=52                # D-10 legs-band layer time
HATCH_LINES=4
SCAN_LEN=0.014                  # m, one leg-group period
ABSORPTIVITY=0.62
E_PHYS_J=60.45                  # absorbed J per physical layer, this slice
L0_SCAN_LEN=0.075               # m, the full-part scan line L0 aggregates over
L0_SCAN_PHASE_S=640             # s, L0's scan phase at N = 50
#
# SCAN CONVENTION -- OPEN QUESTION, see derived/d11/D11-RUNBOOK.md section 3a.
# Absorbed energy per computational layer is N * E_PHYS_J under EVERY
# convention (laser power is always back-solved from it), so they differ only
# in how that energy is spread in time -- i.e. in the INSTANTANEOUS power, and
# therefore in peak temperature. That matters because T_cut only bites on a
# cooling crossing: if the model never reaches T_cut, the whole T_cut axis is
# degenerate. Measured: L0 peaks at 2630 K; the 'energy' convention peaks at
# ~690 K, below every T_cut in the sweep {1073, 1173, 1273, 1373} K.
#
#   energy      scan phase = 25 % of the layer clock (L0's 640/2660 ratio).
#               Constant ~7.6 W. Cheapest (910 build steps). VERIFIED
#               DEGENERATE on the T_cut axis -- kept only for reproducibility.
#   power       scan phase pinned at 4 lines x 1 step x 20 s = 80 s for every
#               N; power then scales with N (60.9 / 30.5 / 12.2 W). Same cost
#               as 'energy'. Reaches T_cut at N = 50 but not at N = 10, so the
#               T_cut axis goes degenerate as N refines -- confounds the sweep.
#   l0_density  scan phase tracks the slice's real share of the scan work
#               (L0_SCAN_PHASE_S * N/50 * SCAN_LEN/L0_SCAN_LEN) at dt = 10 s.
#               Integer-step rounding makes power jitter 24-61 W across the
#               sweep and N = 10 still misses the 1100 C member.
#   fixed_power dt refined with N so the scan phase is exact -> constant
#               40.8 W, peak T 2052/2001/1707 K for N = 50/25/10, so every
#               T_cut in the ladder is crossed at every N. DEFAULT.
#
# Measured peak T (thermal-only probe over the full build height,
# tools/d11_thermal_probe.sh), T_cut ladder = {1073, 1173, 1273, 1373} K:
#   convention   N=50    N=25    N=10    ladder covered
#   energy        616 K   --      421 K  none at any N
#   power        2571 K  1677 K   916 K  none at N=10
#   l0_density   1904 K  2505 K  1321 K  N=10 misses 1100 C
#   fixed_power  2052 K  2001 K  1707 K  all, at every N
CONV="${D11_SCAN_CONVENTION:-fixed_power}"
case "${CONV}" in
  energy)
    DT="${D11_DT:-20.0}"
    SCAN_STEPS=$(awk -v n="${N}" 'BEGIN{s=int(n/6.25+0.5); if(s<1)s=1; printf "%d", s}')
    ;;
  power)
    DT="${D11_DT:-20.0}"
    SCAN_STEPS=1
    ;;
  l0_density)
    DT="${D11_DT:-10.0}"
    # ideal scan phase for this slice and this N, then round to whole steps
    SCAN_STEPS=$(awk -v n="${N}" -v p="${L0_SCAN_PHASE_S}" -v sl="${SCAN_LEN}" \
                     -v l0="${L0_SCAN_LEN}" -v h="${HATCH_LINES}" -v d="${DT}" \
                 'BEGIN{s=int(p*(n/50.0)*(sl/l0)/(h*d)+0.5); if(s<1)s=1; printf "%d", s}')
    ;;
  fixed_power)
    # Same as l0_density but WITHOUT the integer-step rounding error: dt is
    # refined with N so the scan phase lands exactly on the slice's share of
    # the scan work. Consequences: laser power is CONSTANT across the sweep
    # (~41 W, inside L0's own band) instead of jittering 24-61 W, peak T is
    # therefore ~constant across N, and steps/computational-layer is constant
    # (~87) so the step COUNT scales with the number of computational layers
    # -- refining N genuinely costs more, which is honest.
    SCAN_STEPS="${D11_SCAN_STEPS:-1}"
    DT=$(awk -v n="${N}" -v p="${L0_SCAN_PHASE_S}" -v sl="${SCAN_LEN}" \
             -v l0="${L0_SCAN_LEN}" -v h="${HATCH_LINES}" -v s="${SCAN_STEPS}" \
         'BEGIN{printf "%.6f", p*(n/50.0)*(sl/l0)/(h*s)}')
    ;;
  *) echo "d11: unknown D11_SCAN_CONVENTION=${CONV}" >&2; exit 2 ;;
esac

COMP_LAYERS=$(( PHYS_LAYERS / N ))
(( COMP_LAYERS * N == PHYS_LAYERS )) || { echo "d11: N=${N} does not divide ${PHYS_LAYERS}" >&2; exit 2; }
LAYER_THICK=$(awk -v n="${N}" 'BEGIN{printf "%.6e", n*20e-6}')
STEPS_PER_LAYER=$(awk -v n="${N}" -v c="${LAYER_CLOCK_S}" -v d="${DT}" 'BEGIN{printf "%d", n*c/d}')
DWELL_STEPS=$(( STEPS_PER_LAYER - HATCH_LINES * SCAN_STEPS ))
(( DWELL_STEPS > 0 )) || { echo "d11: no dwell left for N=${N}" >&2; exit 2; }
T_SCAN=$(awk -v h="${HATCH_LINES}" -v s="${SCAN_STEPS}" -v d="${DT}" 'BEGIN{printf "%.6f", h*s*d}')
# Mechanics cadence: a FIXED NUMBER OF SOLVES PER COMPUTATIONAL LAYER for
# every N (default 6). A fixed step interval would hand the coarse-N runs
# more solves per deposited layer than the fine-N runs and confound the
# lumping sweep; a fixed wall-clock interval breaks once dt varies with N
# (fixed_power). Per-layer is the control that stays constant under every
# convention.
MECH_PER_LAYER="${D11_MECH_PER_LAYER:-6}"
MECH_EVERY=$(awk -v s="${STEPS_PER_LAYER}" -v m="${MECH_PER_LAYER}" 'BEGIN{e=int(s/m+0.5); if(e<1)e=1; printf "%d", e}')
LASER_POWER=$(awk -v n="${N}" -v e="${E_PHYS_J}" -v a="${ABSORPTIVITY}" -v t="${T_SCAN}" 'BEGIN{printf "%.6f", n*e/(a*t)}')
SCAN_SPEED=$(awk -v l="${SCAN_LEN}" -v s="${SCAN_STEPS}" -v d="${DT}" 'BEGIN{printf "%.6e", l/(s*d)}')

RELAX_ARGS=()
if [[ "${TCUT_C}" != "none" ]]; then
  TCUT_K=$(awk -v c="${TCUT_C}" 'BEGIN{printf "%.2f", c+273.15}')
  RELAX_ARGS=(--stress-relaxation-temperature "${TCUT_K}")
else
  TCUT_K="none"
fi

PLATE_K=347.05                  # A.3/D-07 substrate underside Dirichlet

mkdir -p "${OUT_ROOT}"
cat > "${OUT_ROOT}/d11_case.json" <<EOF
{
 "schema_version": "ambench.d11-case/1",
 "N": ${N},
 "T_cut_C": "${TCUT_C}",
 "T_cut_K": "${TCUT_K}",
 "arm": "${D11_ARM}",
 "scan_convention": "${CONV}",
 "dt_s": ${DT},
 "computational_layers": ${COMP_LAYERS},
 "layer_thickness_m": ${LAYER_THICK},
 "steps_per_layer": ${STEPS_PER_LAYER},
 "scan_steps_per_line": ${SCAN_STEPS},
 "dwell_steps": ${DWELL_STEPS},
 "scan_phase_s": ${T_SCAN},
 "mechanics_every": ${MECH_EVERY},
 "mechanics_solves_per_layer": ${MECH_PER_LAYER},
 "laser_power_W": ${LASER_POWER},
 "scan_speed_m_s": ${SCAN_SPEED},
 "build_steps": $(( COMP_LAYERS * STEPS_PER_LAYER )),
 "mesh": "${MESH}",
 "config": "${CONFIG}"
}
EOF

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
XLA_PLATFORM="${XLA_PLATFORM:-gpu}"
export JAX_PLATFORM_NAME="${XLA_PLATFORM}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m jax_fem_am.simulation.runner \
  --config "${CONFIG}" \
  --inp "${MESH}" \
  --mesh-length-scale 1.0e-3 \
  --output-dir "${OUT_ROOT}" \
  --profile-json "${OUT_ROOT}/profile.json" \
  --profile-label "d11-${TAG}-${RUN_ID}" \
  --xla-platform "${XLA_PLATFORM}" \
  --xla-preallocate off \
  --xla-linear-solver pardiso \
  --build-axis z \
  --base-side min \
  --support-thickness 12.7e-3 \
  --layers "${COMP_LAYERS}" \
  --layer-thickness "${LAYER_THICK}" \
  --layer-activation-mode layer_on_scan \
  --layer-activation-geometry centroid \
  --future-layer-mode void \
  --powder-mode powder \
  --reset-activation-temperature \
  --activation-reset-temperature "${PLATE_K}" \
  --powder-elset POWDER \
  --powder-solid-E 10e9 \
  --powder-solid-yield 1e6 \
  --powder-solid-hardening 1e7 \
  --solidus-temperature 1552.0 \
  --liquidus-temperature 1552.0 \
  --latent-heat 0.0 \
  --phase-history-model legacy_reset \
  "${RELAX_ARGS[@]}" \
  --laser-power "${LASER_POWER}" \
  --source-model legacy \
  --beam-radius 2.0e-3 \
  --source-depth 1.0e-3 \
  --scan-start 0.0 \
  --scan-end "${SCAN_LEN}" \
  --hatch-start -0.0025 \
  --hatch-end 0.0025 \
  --hatch-lines-per-layer "${HATCH_LINES}" \
  --serpentine \
  --scan-speed "${SCAN_SPEED}" \
  --scan-steps-per-layer "${SCAN_STEPS}" \
  --dt "${DT}" \
  --dwell-steps-between-layers "${DWELL_STEPS}" \
  --preheat-temperature "${PLATE_K}" \
  --bottom-temperature "${PLATE_K}" \
  --bottom-thermal-bc fixed \
  --ambient "${PLATE_K}" \
  --surface-selection exterior \
  --boundary-tol 1.0e-6 \
  --quadrature-order 2 \
  --mechanics-every "${MECH_EVERY}" \
  --mechanics-rel-tol 5e-5 \
  --mechanics-acceptance abaqus \
  --mechanics-max-iter 50 \
  --mechanics-line-search \
  --mechanics-max-cuts 3 \
  --bottom-mechanics-bc fixed \
  --cooling-steps 90 \
  --cooling-dt 40.0 \
  "$@" 2>&1 | tee "${OUT_ROOT}/run.log"
