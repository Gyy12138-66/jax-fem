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

# D11_SERIAL=1 runs members ONE AT A TIME with the full thread budget each.
# Required on the 0.1 mm shared mesh: the B smoke measured 13 GB RAM + 11.8 GB
# GPU for a single member at 294k mechanics DOF, so two concurrent members do
# not fit in this box (31 GB / 16.3 GB) and would OOM mid-campaign.
# D11_RESUME=1 skips any member that already has d11_metrics.json -- a 20 h
# serial campaign should not restart from zero after one interruption.
SERIAL="${D11_SERIAL:-0}"
RESUME="${D11_RESUME:-0}"
if [[ "${SERIAL}" == "1" ]]; then
  per="${TOTAL_THREADS}"
else
  per=$(( TOTAL_THREADS / n_members ))
fi
(( per >= 1 )) || per=1

run_member () {   # <N> <T> <ARM> <tag> <out>
  local N="$1" T="$2" ARM="$3" tag="$4" out="$5"
  export MKL_NUM_THREADS="${per}" OMP_NUM_THREADS="${per}"
  D11_ARM="${ARM}" OUT_ROOT="${out}" \
    bash "${SCRIPT_DIR}/run_d11_case.sh" "${N}" "${T}" \
    > "${ROOT}/${tag}.stdout" 2>&1
  local rc=$?
  echo "${rc}" > "${ROOT}/${tag}.rc"
  if [[ "${rc}" == "0" ]]; then
    "${PY}" "${SCRIPT_DIR}/d11_metrics.py" "${out}" \
      >> "${ROOT}/${tag}.stdout" 2>&1
    echo "$?" > "${ROOT}/${tag}.metrics.rc"
  fi
  return "${rc}"
}

pids=(); tags=(); fail=0
for spec in "$@"; do
  IFS=: read -r N T ARM <<< "${spec}"
  ARM="${ARM:-mean}"
  tag="N${N}_T${T}"
  [[ "${ARM}" == "mean" ]] || tag="${tag}_${ARM}"
  out="${ROOT}/${tag}"
  tags+=("${tag}")

  if [[ "${RESUME}" == "1" && -f "${out}/d11_metrics.json" ]]; then
    echo "skip ${tag}: already has d11_metrics.json (D11_RESUME=1)"
    echo 0 > "${ROOT}/${tag}.rc"; echo 0 > "${ROOT}/${tag}.metrics.rc"
    continue
  fi
  rm -rf "${out}" "${ROOT}/${tag}.rc" "${ROOT}/${tag}.metrics.rc"

  if [[ "${SERIAL}" == "1" ]]; then
    echo "running ${tag} (arm ${ARM}, ${per} threads, serial) -> ${out}"
    run_member "${N}" "${T}" "${ARM}" "${tag}" "${out}" || fail=1
    echo "  ${tag} rc=$(cat "${ROOT}/${tag}.rc")"
  else
    ( run_member "${N}" "${T}" "${ARM}" "${tag}" "${out}" ) &
    pids+=($!)
    echo "launched ${tag} (arm ${ARM}, ${per} threads) -> ${out}"
  fi
done

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
