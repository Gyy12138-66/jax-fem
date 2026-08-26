#!/usr/bin/env bash
# ============================================================================
# V1-P source-implementation ambiguity arms (WP2, review P0-2/P0-3).
#
# Four thermal-only arms on the shared CBM-B operating point, isolating one
# implementation reading each:
#   p_legacy       as-published harness (half-space exponential source,
#                  fixture frozen-solid) - the historical baseline, re-run
#                  in this harness so every arm shares commit + environment.
#   p_band_renorm  deposition confined to the 20 um powder layer AND
#                  renormalized to the full absorbed power (the "absorption
#                  happens in the powder layer" paper reading).
#   p_band_trunc   deposition confined to the 20 um layer WITHOUT
#                  renormalization (tail energy treated as never absorbed;
#                  lower-bound bracket arm).
#   p_fixphase     half-space source, but substrate/support thermal
#                  properties follow temperature through mushy/liquid
#                  (isolates review P0-3 alone).
#
# Zero-calibration: no arm is tuned toward any measurement. The adopted-arm
# rule lives in ../../V1PE-RERUN-PLAN.md and must be frozen BEFORE reading
# the cross-arm comparison.
#
# Usage (box-159):
#   bash run_v1p_srcband_arms.sh              # all four arms + comparison
#   ARMS="p_band_renorm" bash ...             # subset
#   SKIP_LEGACY=1 BASELINE_DIR=/path/to/old   # reuse an existing baseline
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${WORK_ROOT:-$(cd "${MODEL_DIR}/../../../../.." && pwd)}/output/v1p_srcband_${RUN_ID}}"
ARMS="${ARMS:-p_legacy p_band_renorm p_band_trunc p_fixphase}"
CUTOFF="${CUTOFF:-2.0e-5}"   # = powder-layer thickness (D-V1 geometry)

declare -A ARM_ARGS=(
  [p_legacy]=""
  [p_band_renorm]="--source-depth-cutoff ${CUTOFF} --source-cutoff-renormalize"
  [p_band_trunc]="--source-depth-cutoff ${CUTOFF}"
  [p_fixphase]="--fixture-thermal-phase follow-temperature"
)

mkdir -p "${CAMPAIGN_ROOT}"
echo "campaign root: ${CAMPAIGN_ROOT}"
METRIC_DIRS=()

for arm in ${ARMS}; do
  [[ -v "ARM_ARGS[$arm]" ]] || { echo "unknown arm: $arm" >&2; exit 2; }
  out="${CAMPAIGN_ROOT}/${arm}"
  if [[ "$arm" == "p_legacy" && "${SKIP_LEGACY:-0}" == "1" ]]; then
    [[ -d "${BASELINE_DIR:-}" ]] || { echo "SKIP_LEGACY=1 needs BASELINE_DIR" >&2; exit 2; }
    echo "arm p_legacy: reusing ${BASELINE_DIR}"
    out="${BASELINE_DIR}"
  else
    echo "=== arm ${arm}: ${ARM_ARGS[$arm]:-<baseline>} ==="
    CASE_TAG="cbmB_${arm}" OUT_ROOT="${out}" RUN_ID="${RUN_ID}" \
      EXTRA_ARGS="${ARM_ARGS[$arm]}" \
      bash "${MODEL_DIR}/run_v1_cbmB.sh"
  fi
  "${PYTHON_BIN}" "${MODEL_DIR}/analyze_v1.py" "${out}" \
    --out "${out}/v1_meltpool_metrics.json"
  METRIC_DIRS+=("${arm}=${out}")
done

"${PYTHON_BIN}" "${MODEL_DIR}/analyze_v1p_arms.py" \
  --output "${CAMPAIGN_ROOT}/v1p_srcband_comparison" \
  "${METRIC_DIRS[@]}"

echo "v1p source-band arms complete: ${CAMPAIGN_ROOT}"
