#!/usr/bin/env bash
# ============================================================================
# V1-E bare-plate single track -- PROTOCOL DRAFT (WP2, review P0-1).
#
# Code-to-experiment leg: NIST AMB2018-02 bare IN625 plate, room
# temperature, no powder-model contribution (deviations D-V1-07 promise).
#
# FAIL-CLOSED: this script refuses to run until the frozen protocol
# supplies the two inputs that have NO defensible default:
#   ABSORPTIVITY   bare-plate absorptivity (parity 0.62 is powder DRS -
#                  invalid here; supply the sourced value or sweep arm)
#   SOURCE_DEPTH   heat-source depth scale in metres for dense metal
#                  (the 100 um OPD is a powder proxy - invalid here)
# Optional protocol knobs (defaults are DRAFT choices, registered in
# V1PE-RERUN-PLAN.md and subject to the domain/BC sweep):
#   LZ             domain depth, default 0.60e-3 (NIST plate is 3.2 mm; the
#                  depth ladder 0.3/0.6/1.2 mm must show insensitivity)
#   AMBIENT_K      default 293.15 (D-V1-11 room-temperature reading; the
#                  actual preheat is NOT stated by Lane -> interval arm)
#   EXTRA_ARGS     e.g. "--source-depth-cutoff 2.0e-5" for band variants
#
# Example (values illustrative only - the plan must source them):
#   ABSORPTIVITY=0.30 SOURCE_DEPTH=1.0e-5 bash run_v1e_bareplate_draft.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MODEL_DIR}/../../../../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v python >/dev/null 2>&1 && [ -f /home/user/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
fi

: "${ABSORPTIVITY:?V1-E refuses to run without a sourced bare-plate ABSORPTIVITY (see header)}"
: "${SOURCE_DEPTH:?V1-E refuses to run without a sourced dense-metal SOURCE_DEPTH (see header)}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
POWER="${POWER:-195}"
SPEED="${SPEED:-0.800}"
CASE_TAG="${CASE_TAG:-v1e_bareB}"
DT="${DT:-1.0e-5}"
LZ="${LZ:-0.60e-3}"
CELL="${CELL:-10.0e-6}"
AMBIENT_K="${AMBIENT_K:-293.15}"
THERMAL_OUT_EVERY="${THERMAL_OUT_EVERY:-2}"
OUT_ROOT="${OUT_ROOT:-${WORK_ROOT}/output/v1e_${CASE_TAG}_P${POWER}_${RUN_ID}}"

mkdir -p "$(dirname "${OUT_ROOT}")"
if ! mkdir "${OUT_ROOT}" 2>/dev/null; then
  echo "v1e: refusing existing OUT_ROOT: ${OUT_ROOT}" >&2
  exit 2
fi

MESH="${OUT_ROOT}/v1e_bareplate_c3d8.inp"
CONFIG="${OUT_ROOT}/v1e_material_config.json"
PATH_FILE="${OUT_ROOT}/v1e_path_${CASE_TAG}.csv"

"${PYTHON_BIN}" "${MODEL_DIR}/make_v1_mesh.py" \
  --cell-size "${CELL}" --lz "${LZ}" --output "${MESH}"
"${PYTHON_BIN}" "${MODEL_DIR}/make_v1e_material_config.py" \
  --absorptivity "${ABSORPTIVITY}" --output "${CONFIG}" \
  ${TABLE_ROOT:+--table-root "${TABLE_ROOT}"}
"${PYTHON_BIN}" "${MODEL_DIR}/make_v1_path.py" \
  --power "${POWER}" --speed "${SPEED}" --z "${LZ}" --output "${PATH_FILE}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORM_NAME="${XLA_PLATFORM:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1

# The mesh keeps a 20 um "layer" band at the top purely so the runner's
# activation machinery has a part region; its material is solid (config
# collapses powder -> solid), so physically the domain is one bare plate.
# Bottom BC: convection to ambient (draft; NOT the parity fixed-preheat
# Dirichlet - the NIST plate has no temperature-controlled underside at
# this depth). The depth/BC sensitivity arm is part of the plan.
SUPPORT_THICKNESS="$(${PYTHON_BIN} -c "print(f'{${LZ} - 2.0e-5:.6e}')")"

cd "${WORK_ROOT}"
SOLVER_CMD=(
  "${PYTHON_BIN}" -m jax_fem_am.simulation.runner
  --config "${CONFIG}"
  --inp "${MESH}"
  --output-dir "${OUT_ROOT}"
  --profile-json "${OUT_ROOT}/profile.json"
  --profile-label "v1e-${CASE_TAG}-P${POWER}"
  --xla-platform "${XLA_PLATFORM:-cpu}"
  --xla-preallocate off
  --xla-linear-solver "${LINEAR_SOLVER:-pardiso}"
  --build-axis z
  --base-side min
  --layer-thickness 2.0e-5
  --layers 1
  --support-thickness "${SUPPORT_THICKNESS}"
  --path-file "${PATH_FILE}"
  --path-length-scale 1.0
  --source-model legacy
  --beam-radius 5.0e-5
  --source-depth "${SOURCE_DEPTH}"
  --laser-power "${POWER}"
  --dt "${DT}"
  --layer-activation-mode layer_on_scan
  --layer-activation-geometry intersection
  --future-layer-mode void
  --active-window-below-layers 0
  --inactive-mass-factor 1.0
  --powder-mode powder
  --surface-selection exterior
  --boundary-tol 1.0e-6
  --quadrature-order 2
  --ambient "${AMBIENT_K}"
  --preheat-temperature "${AMBIENT_K}"
  --bottom-thermal-bc convection
  --front-surface-loss-radiation
  --cooling-steps 30
  --cooling-dt "${DT}"
  --mechanics-every 0
  --no-xla-thermal-only-mechanics-surrogate
  --thermal-mass-lumping
  --thermal-output-every "${THERMAL_OUT_EVERY}"
  --summary-every 25
  --phase-history-model paper_irreversible
)
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  SOLVER_CMD+=(${EXTRA_ARGS})
fi

printf '%q ' "${SOLVER_CMD[@]}" > "${OUT_ROOT}/solver_command.txt"
printf '\n' >> "${OUT_ROOT}/solver_command.txt"
{
  echo "protocol_status: DRAFT (freeze V1PE-RERUN-PLAN.md before record runs)"
  echo "absorptivity: ${ABSORPTIVITY}"
  echo "source_depth_m: ${SOURCE_DEPTH}"
  echo "lz_m: ${LZ}  cell_m: ${CELL}  ambient_k: ${AMBIENT_K}"
} > "${OUT_ROOT}/v1e_protocol_inputs.txt"

"${SOLVER_CMD[@]}"

"${PYTHON_BIN}" -m jax_fem_am.verification.run_audit "${OUT_ROOT}" \
  --output "${OUT_ROOT}/v1e_run_audit.json" \
  --ambient "${AMBIENT_K}" \
  --quality-threshold 0.05 || echo "v1e WARNING: run_audit failed" >&2

LAYER_TOP="${LZ}"
SUBSTRATE_TOP="$(${PYTHON_BIN} -c "print(f'{${LZ} - 2.0e-5:.6e}')")"
"${PYTHON_BIN}" "${MODEL_DIR}/analyze_v1.py" "${OUT_ROOT}" \
  --layer-top "${LAYER_TOP}" --substrate-top "${SUBSTRATE_TOP}" \
  --out "${OUT_ROOT}/v1_meltpool_metrics.json"

echo "v1e ${CASE_TAG} complete: ${OUT_ROOT}"
