#!/usr/bin/env bash
# ============================================================================
# D-11 T_cut band: the four remaining T_cut members on the shared 0.1 mm mesh.
#
#   bash d11_run_tcut.sh [--dry-run] [specs...]
#     default specs: 10:800 10:900 10:1100 10:none
#
# Completes the 5-point band together with the existing N10_T1000 member from
# the B campaign. Same mesh, same harness, same output root -- so the five are
# directly comparable, which is the whole point.
#
# Order is 800 -> 900 -> 1100 -> none. `none` runs LAST deliberately: it is the
# member most likely to diverge (no high-temperature memory reset at all), and
# putting it last means a divergence cannot cost the other three.
#
# `none` failure handling is NOT a rescue path. It is classified by
# d11_none_guard.py into two pre-registered tiers:
#   wiring-suspect       died before real activation (ledger <= 12 steps) ->
#                        bug suspicion, must NOT be registered as infeasible
#   infeasible-candidate diverged after build progress -> may be registered,
#                        with evidence on disk
# No parameter is ever tuned to make `none` converge.
#
# Env: D11_TOTAL_THREADS (default 20 -- leaves ~10 CPU threads for the
#      concurrent side campaign), D11_RESUME (default 1), D11_ARM (default
#      mean; the m_low/m_high arm campaign is then a thin wrapper over this
#      driver), D11_B_OUT_ROOT.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${PYTHON_BIN:-/home/user/miniconda3/envs/jax-fem-gpu/bin/python}"

DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; shift; fi
SPECS=("$@")
[[ ${#SPECS[@]} -gt 0 ]] || SPECS=(10:800 10:900 10:1100 10:none)

SHARED="${CASE_DIR}/derived/meshes/amb_d11_N5.inp"
CONFIG_ARM="${D11_ARM:-mean}"
CONFIG="${CASE_DIR}/derived/d11/amb_material_config_D11_${CONFIG_ARM}.json"
OUT_ROOT="${D11_B_OUT_ROOT:-/home/user/work/output/d11_B}"
MARKER="${OUT_ROOT}/B_tcut.done"

echo "=== D-11 T_cut band on the shared mesh ==="
echo "    specs : ${SPECS[*]}"
echo "    arm   : ${CONFIG_ARM}"
echo "    mesh  : ${SHARED}"
echo "    out   : ${OUT_ROOT}"

[[ -f "${SHARED}" ]] || { "${PY}" "${SCRIPT_DIR}/make_d11_mesh.py" 5 >/dev/null || exit 2; }
[[ -f "${CONFIG}" ]] || { "${PY}" "${SCRIPT_DIR}/make_d11_config.py" --arm "${CONFIG_ARM}" >/dev/null || exit 2; }

ELEMS=$("${PY}" -c "import json;print(json.load(open('${CASE_DIR}/derived/meshes/amb_d11_N5_summary.json'))['elements'])")
echo "    shared mesh elements: ${ELEMS}"
if [[ "${ELEMS}" != "90720" ]]; then
  echo "PREFLIGHT FAILED: expected 90720 elements, got ${ELEMS}" >&2; exit 3
fi

# ---- preflight: alignment + tag collision -----------------------------------
echo "--- preflight: alignment (N=10 must be 2 rows) and tag collisions"
collide=0
for spec in "${SPECS[@]}"; do
  IFS=: read -r N T ARM <<< "${spec}"
  ARM="${ARM:-${CONFIG_ARM}}"
  tag="N${N}_T${T}"
  [[ "${ARM}" == "mean" ]] || tag="${tag}_${ARM}"

  tmp="/tmp/d11_tcut_align_${tag}"; rm -rf "${tmp}"
  line=$(D11_MESH="${SHARED}" D11_ARM="${ARM}" OUT_ROOT="${tmp}" \
    bash "${SCRIPT_DIR}/run_d11_case.sh" "${N}" "${T}" --xla-dry-run 2>&1 \
    | grep -E 'mesh row|MISALIGNED')
  echo "    ${tag}: ${line:-NO ALIGNMENT LINE}"
  case "${line}" in
    *MISALIGNED*|'') echo "PREFLIGHT FAILED for ${tag}" >&2; exit 3 ;;
  esac
  rows=$(sed -n 's/.*, \([0-9]*\) rows per.*/\1/p' <<< "${line}")
  if [[ "${N}" == "10" && "${rows}" != "2" ]]; then
    echo "PREFLIGHT FAILED: N=10 expected 2 rows, got ${rows}" >&2; exit 3
  fi

  if [[ -e "${OUT_ROOT}/${tag}" ]]; then
    if [[ -f "${OUT_ROOT}/${tag}/d11_metrics.json" ]]; then
      echo "      note: ${tag} already complete (resume will skip it)"
    else
      echo "      WARNING: ${tag} exists without metrics; it will be cleared"
      collide=1
    fi
  fi
done
# The band members must not clash with B's completed N-sweep members. Compare
# FULL tags including the arm suffix: N10_T1000_mlow is a different member from
# N10_T1000, so the arm campaign (D11_ARM=mlow, spec 10:1000) is legitimate and
# must not be blocked here.
for keep in N50_T1000 N25_T1000 N10_T1000; do
  for spec in "${SPECS[@]}"; do
    IFS=: read -r N T ARM <<< "${spec}"
    ARM="${ARM:-${CONFIG_ARM}}"
    tag="N${N}_T${T}"
    [[ "${ARM}" == "mean" ]] || tag="${tag}_${ARM}"
    if [[ "${tag}" == "${keep}" ]]; then
      echo "PREFLIGHT FAILED: spec ${spec} (arm ${ARM}) resolves to ${tag}," >&2
      echo "  which would overwrite completed B member ${keep}." >&2
      exit 3
    fi
  done
done
echo "    tag collision check: OK (B's N50/N25/N10_T1000 untouched)"
[[ "${collide}" == "0" ]] || echo "    (partial dirs above will be cleared by the stage runner)"

if [[ "${DRY}" == "1" ]]; then
  echo "--- dry run: stopping before production"
  exit 0
fi

# ---- production -------------------------------------------------------------
export D11_MESH="${SHARED}"
export D11_OUT_ROOT="${OUT_ROOT}"
export D11_SERIAL=1
export D11_RESUME="${D11_RESUME:-1}"
export D11_TOTAL_THREADS="${D11_TOTAL_THREADS:-20}"
export D11_ARM="${CONFIG_ARM}"

echo "--- running ${#SPECS[@]} members serially (threads=${D11_TOTAL_THREADS}, resume=${D11_RESUME})"
bash "${SCRIPT_DIR}/d11_run_stage.sh" "${MARKER}" "${SPECS[@]}"
rc=$?

# ---- none-member classification (never a rescue) ----------------------------
for spec in "${SPECS[@]}"; do
  IFS=: read -r N T ARM <<< "${spec}"
  [[ "${T}" == "none" ]] || continue
  ARM="${ARM:-${CONFIG_ARM}}"
  tag="N${N}_T${T}"
  [[ "${ARM}" == "mean" ]] || tag="${tag}_${ARM}"
  member_rc=$(cat "${OUT_ROOT}/${tag}.rc" 2>/dev/null || echo 1)
  echo "--- none guardrail for ${tag} (rc=${member_rc})"
  "${PY}" "${SCRIPT_DIR}/d11_none_guard.py" "${OUT_ROOT}/${tag}" \
    --rc "${member_rc}" | tee -a "${OUT_ROOT}/${tag}.stdout"
done

echo "--- matrix report (T_cut band on N=10)"
"${PY}" "${SCRIPT_DIR}/d11_matrix_report.py" "${OUT_ROOT}" \
  --out "${CASE_DIR}/derived/d11"
echo "TCUT_DONE rc=${rc}"
exit "${rc}"
