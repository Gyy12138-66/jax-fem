#!/usr/bin/env bash
# ============================================================================
# D-11 deliverable B: the N-convergence triple on the SHARED 0.1 mm mesh.
#
#   bash d11_run_b.sh [--dry-run] [N ...]        default: 50 25 10
#
# B decouples the z DISCRETISATION from the layer LUMPING: all three members
# run on one 0.1 mm mesh (amb_d11_N5.inp, 90 720 cells) and N controls only
# how many mesh rows are activated per deposition event (10 / 5 / 2). Any
# M1/M2 difference is then attributable to aggregation alone -- which is what
# D.7 means by "the matrix IS the D.5 lump-ratio study".
#
# Everything the campaign must not get wrong is encoded here rather than left
# to the operator:
#   * shared mesh + config are built if missing;
#   * members run SERIALLY (the smoke measured 13 GB RAM + 11.8 GB GPU for one
#     member at 294k mechanics DOF; two do not fit in a 31 GB / 16.3 GB box);
#   * the run is RESUMABLE -- a member with d11_metrics.json is skipped, so an
#     interruption in a ~21 h campaign costs one member, not the campaign;
#   * output root is separate from stage 1, because B's absolute values are
#     NOT comparable with the stage-1 per-N-mesh values (registered convention
#     change, 2026-08-04);
#   * the alignment assertion in run_d11_case.sh refuses any N whose
#     computational layer is not an integer number of mesh rows.
#
# Pre-registered verdict rule (locked before this runs): deciding metric is the
# load-bearing thick leg L1, M1 AND M2 both < 5 %. If both pairs still fail,
# that is true aggregation non-convergence -> D.5-style interval publication.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${PYTHON_BIN:-/home/user/miniconda3/envs/jax-fem-gpu/bin/python}"

DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; shift; fi
NS=("$@"); [[ ${#NS[@]} -gt 0 ]] || NS=(50 25 10)

SHARED="${CASE_DIR}/derived/meshes/amb_d11_N5.inp"
CONFIG="${CASE_DIR}/derived/d11/amb_material_config_D11_mean.json"
OUT_ROOT="${D11_B_OUT_ROOT:-/home/user/work/output/d11_B}"
TCUT="${D11_B_TCUT:-1000}"

echo "=== D-11 deliverable B: shared-mesh N convergence ==="
echo "    members : ${NS[*]} at T_cut=${TCUT}"
echo "    mesh    : ${SHARED}"
echo "    out     : ${OUT_ROOT}"

if [[ ! -f "${SHARED}" ]]; then
  echo "--- building shared 0.1 mm mesh"
  "${PY}" "${SCRIPT_DIR}/make_d11_mesh.py" 5 >/dev/null || exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "--- building material config"
  "${PY}" "${SCRIPT_DIR}/make_d11_config.py" --arm mean >/dev/null || exit 2
fi

ELEMS=$("${PY}" -c "import json;print(json.load(open('${CASE_DIR}/derived/meshes/amb_d11_N5_summary.json'))['elements'])")
echo "    shared mesh elements: ${ELEMS}"
[[ "${ELEMS}" == "90720" ]] || echo "    WARNING: expected 90720 elements"

echo "--- alignment preflight (expect 10 / 5 / 2 rows per computational layer)"
for N in "${NS[@]}"; do
  tmp="/tmp/d11_b_align_N${N}"; rm -rf "${tmp}"
  D11_MESH="${SHARED}" OUT_ROOT="${tmp}" \
    bash "${SCRIPT_DIR}/run_d11_case.sh" "${N}" "${TCUT}" --xla-dry-run 2>&1 \
    | grep -E 'mesh row|MISALIGNED' || { echo "    N=${N}: PREFLIGHT FAILED"; exit 3; }
done

if [[ "${DRY}" == "1" ]]; then
  echo "--- dry run: stopping before production"
  exit 0
fi

specs=(); for N in "${NS[@]}"; do specs+=("${N}:${TCUT}"); done
export D11_MESH="${SHARED}"
export D11_OUT_ROOT="${OUT_ROOT}"
export D11_SERIAL=1
export D11_RESUME="${D11_RESUME:-1}"
export D11_TOTAL_THREADS="${D11_TOTAL_THREADS:-26}"

echo "--- running ${#specs[@]} members serially (resume=${D11_RESUME})"
bash "${SCRIPT_DIR}/d11_run_stage.sh" "${OUT_ROOT}/B.done" "${specs[@]}"
rc=$?

echo "--- matrix report"
"${PY}" "${SCRIPT_DIR}/d11_matrix_report.py" "${OUT_ROOT}" \
  --out "${CASE_DIR}/derived/d11"
echo "B_DONE rc=${rc}"
exit "${rc}"
