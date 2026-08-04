#!/usr/bin/env bash
# ============================================================================
# D-11 pre-flight: does the model actually reach T_cut?
#
#   bash d11_thermal_probe.sh [N ...]                  # default convention
#   D11_PROBE_CONVENTIONS="energy power l0_density fixed_power" \
#     bash d11_thermal_probe.sh 50 25 10               # full comparison table
#
# T_cut has NO effect unless material crosses it on cooling (the relaxation
# mask is `T_quad >= T_cut`), so if peak temperature stays below the sweep
# ladder {1073, 1173, 1273, 1373} K the entire T_cut axis collapses to one
# answer and the 15-run matrix measures nothing on it. That is a multi-day
# mistake to discover afterwards; this probe costs minutes.
#
# Thermal only (--mechanics-every 0), dwell cut to 2 steps -- peak T is
# reached during a scan phase, so a short dwell does not change the answer.
#
# The probe builds the FULL height. It must: peak T occurs late in the build,
# when the part is tall and the substrate heat sink is far away. An earlier
# revision of this script probed only the first two layers and under-reported
# peak T by 300-950 K (e.g. fixed_power N=10: 762 K shallow vs 1707 K full),
# which flips the "is the axis alive" verdict. Do not reintroduce
# --max-print-layers here.
#
# Reference: the validated L0 run peaks at 2630 K.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS=("$@")
[[ ${#NS[@]} -gt 0 ]] || NS=(50 25 10)
CONVS=(${D11_PROBE_CONVENTIONS:-fixed_power})
THREADS="${D11_TOTAL_THREADS:-8}"

printf '%-12s %-4s %8s %8s %12s   %s\n' \
       convention N steps power_W peak_T_K 'T_cut reached (C)'
for conv in "${CONVS[@]}"; do
  for N in "${NS[@]}"; do
    out="/tmp/d11_probe_${conv}_N${N}"
    rm -rf "${out}"
    MKL_NUM_THREADS="${THREADS}" OMP_NUM_THREADS="${THREADS}" \
    D11_SCAN_CONVENTION="${conv}" OUT_ROOT="${out}" \
      bash "${SCRIPT_DIR}/run_d11_case.sh" "${N}" 1000 \
        --mechanics-every 0 \
        --dwell-steps-between-layers 2 --cooling-steps 0 \
        --summary-every 1 >"${out}.log" 2>&1
    peak=$(grep -o 'T_max=[0-9.]*' "${out}.log" | cut -d= -f2 | sort -g | tail -1)
    pw=$(grep -o '"laser_power_W": [0-9.]*' "${out}/d11_case.json" 2>/dev/null | cut -d' ' -f2)
    st=$(grep -o '"steps_per_layer": [0-9]*' "${out}/d11_case.json" 2>/dev/null | cut -d' ' -f2)
    reached=""
    for pair in 800:1073 900:1173 1000:1273 1100:1373; do
      c="${pair%%:*}"; k="${pair##*:}"
      if awk -v p="${peak:-0}" -v k="${k}" 'BEGIN{exit !(p>=k)}'; then
        reached="${reached}${reached:+,}${c}"
      fi
    done
    printf '%-12s %-4s %8s %8s %12s   %s\n' \
           "${conv}" "${N}" "${st:-?}" "${pw:-?}" "${peak:-FAIL}" \
           "${reached:-NONE -- axis degenerate}"
  done
done
echo PROBE_DONE
