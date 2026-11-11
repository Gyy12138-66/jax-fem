#!/usr/bin/env bash
# ============================================================================
# V1-P mesh-convergence ladder (WP2, review P0-5): 20 / 10 / 5 um uniform
# hexes on the CBM-B operating point. Scan path (12.5 um row spacing) and
# dt are held fixed so the ladder isolates the spatial-discretization
# factor; the path-step ladder is a separate preregistered study.
#
# Cost note: 5 um means 200 x 96 x 60 = 1,152,000 elements (8x baseline);
# schedule it as the long leg.
#
# ARM_EXTRA selects which protocol arm the ladder runs on (default: the
# as-published baseline). The frozen plan must pin this BEFORE production:
#   ARM_EXTRA="" bash run_v1p_mesh3.sh
#   ARM_EXTRA="--source-depth-cutoff 2.0e-5 --source-cutoff-renormalize" ...
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${WORK_ROOT:-$(cd "${MODEL_DIR}/../../../../.." && pwd)}/output/v1p_mesh3_${RUN_ID}}"
CELLS="${CELLS:-20 10 5}"   # um, coarse-to-fine so partial campaigns still order
ARM_EXTRA="${ARM_EXTRA:-}"

mkdir -p "${CAMPAIGN_ROOT}"
METRIC_DIRS=()

for cell_um in ${CELLS}; do
  tag="mesh${cell_um}um"
  out="${CAMPAIGN_ROOT}/${tag}"
  mesh="${CAMPAIGN_ROOT}/v1_single_track_c3d8_${cell_um}um.inp"
  echo "=== ${tag}: generating mesh ==="
  "${PYTHON_BIN}" "${MODEL_DIR}/make_v1_mesh.py" \
    --cell-size "${cell_um}e-6" --output "${mesh}"
  echo "=== ${tag}: solving ==="
  CASE_TAG="cbmB_${tag}" OUT_ROOT="${out}" RUN_ID="${RUN_ID}" \
    MESH="${mesh}" EXTRA_ARGS="${ARM_EXTRA}" \
    bash "${MODEL_DIR}/run_v1_cbmB.sh"
  "${PYTHON_BIN}" "${MODEL_DIR}/analyze_v1.py" "${out}" \
    --out "${out}/v1_meltpool_metrics.json"
  METRIC_DIRS+=("${tag}=${out}")
done

"${PYTHON_BIN}" "${MODEL_DIR}/analyze_v1p_arms.py" \
  --output "${CAMPAIGN_ROOT}/v1p_mesh3_comparison" \
  "${METRIC_DIRS[@]}"

echo "v1p mesh ladder complete: ${CAMPAIGN_ROOT}"
