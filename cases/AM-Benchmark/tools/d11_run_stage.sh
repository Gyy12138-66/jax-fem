#!/usr/bin/env bash
# ============================================================================
# D-11 stage runner: run a set of matrix members concurrently, then extract
# metrics for each.
#
#   bash d11_run_stage.sh <marker-file> <N>:<T_cut>[:<arm>] ...
#   e.g. bash d11_run_stage.sh /tmp/stage1.done 50:1000 25:1000
#        bash d11_run_stage.sh /tmp/stage3.done 50:1000:mlow 50:1000:mhigh
#
# Members are independent, so they run in parallel. MKL_NUM_THREADS is split
# across them (pardiso is the bottleneck and it is CPU, not GPU) -- set
# D11_TOTAL_THREADS to the core budget you want to spend.
#
# Output layout is the one the campaign agreed: OUT_ROOT=<root>/N{N}_T{T},
# with _<arm> appended for the yield-uncertainty arms.
#
# Writes <marker-file> only after every member AND its metrics extraction has
# finished, so a caller can block on it. Per-member exit codes are recorded in
# the marker.
# ============================================================================
set -uo pipefail

MARKER="${1:?usage: d11_run_stage.sh <marker-file> <N>:<T_cut>[:<arm>] ...}"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/user/miniconda3/envs/jax-fem-gpu/bin/python}"
ROOT="${D11_OUT_ROOT:-/home/user/work/output/d11}"
TOTAL_THREADS="${D11_TOTAL_THREADS:-28}"

rm -f "${MARKER}"
mkdir -p "${ROOT}"
n_members=$#
per=$(( TOTAL_THREADS / n_members ))
(( per >= 1 )) || per=1

pids=(); tags=()
for spec in "$@"; do
  IFS=: read -r N T ARM <<< "${spec}"
  ARM="${ARM:-mean}"
  tag="N${N}_T${T}"
  [[ "${ARM}" == "mean" ]] || tag="${tag}_${ARM}"
  out="${ROOT}/${tag}"
  rm -rf "${out}" "${ROOT}/${tag}.rc" "${ROOT}/${tag}.metrics.rc"
  (
    export MKL_NUM_THREADS="${per}" OMP_NUM_THREADS="${per}"
    D11_ARM="${ARM}" OUT_ROOT="${out}" \
      bash "${SCRIPT_DIR}/run_d11_case.sh" "${N}" "${T}" \
      > "${ROOT}/${tag}.stdout" 2>&1
    rc=$?
    echo "${rc}" > "${ROOT}/${tag}.rc"
    if [[ "${rc}" == "0" ]]; then
      "${PY}" "${SCRIPT_DIR}/d11_metrics.py" "${out}" \
        >> "${ROOT}/${tag}.stdout" 2>&1
      echo "$?" > "${ROOT}/${tag}.metrics.rc"
    fi
  ) &
  pids+=($!); tags+=("${tag}")
  echo "launched ${tag} (arm ${ARM}, ${per} threads) -> ${out}"
done

fail=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || fail=1
done

{
  echo "stage finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for tag in "${tags[@]}"; do
    echo "  ${tag} run_rc=$(cat "${ROOT}/${tag}.rc" 2>/dev/null || echo '?')" \
         "metrics_rc=$(cat "${ROOT}/${tag}.metrics.rc" 2>/dev/null || echo '-')"
  done
} > "${MARKER}"
cat "${MARKER}"
echo "STAGE_DONE"
exit "${fail}"
