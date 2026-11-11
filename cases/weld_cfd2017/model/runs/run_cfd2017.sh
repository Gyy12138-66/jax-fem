#!/usr/bin/env bash
# 2017 hybrid laser-MIG CFD case (A7N01 bead-on-plate, 20 x 44 x 6 mm full plate): our conduction solver
# runs heating + cooling with a single equivalent Gaussian source; the CFD frames only calibrate the pool.
# MODE=thermal : thermal-only, CFD-like boundary (h = 100 everywhere) for the pool comparison
# MODE=mech    : thermo-mechanical, realistic boundary (h = 8.2, eps 0.6), Lu 2020 A7N01 tables
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
CASE=$REPO/cases/weld_cfd2017
PY=${PY:-/home/user/miniconda3/envs/jax-fem-gpu/bin/python}
MODE=${MODE:-thermal}
OUT=${OUT:-$HOME/work/159/output/cfd2017_${MODE}}
INP=${INP:-$CASE/inputs/cfd2017_plate_05mm.inp}
PATHCSV=${PATHCSV:-$CASE/inputs/cfd2017_single_source_path.csv}
RB=${RB:-5.0e-3}        # legacy source exp(-2 r^2/rb^2): arc sigma_j = 2.5 mm -> rb = 2 sigma_j
DEPTH=${DEPTH:-2.0e-3}  # exponential depth of the equivalent source [m]
OUT_EVERY=${OUT_EVERY:-10}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-true}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
cd "$REPO"
if [ "$MODE" = "thermal" ]; then
  CFG=$CASE/inputs/material/a7n01_chen2020_thermal_only.json
  EXTRA=(--mechanics-every 0 --convection-h 100 --emissivity 0.0)
else
  CFG=$CASE/inputs/material/a7n01_material_config.json
  EXTRA=(--mechanics-every 1 --no-release-after-cooling --bottom-mechanics-bc edge_minimal --edge-minimal-axis x
         --mechanics-rel-tol 5e-5 --mechanics-line-search --mechanics-max-cuts 3
         --reset-plastic-on-solidify --elastic-melt --mechanics-output-every "$OUT_EVERY")
fi
echo "[$(date -u +%FT%TZ)] cfd2017 MODE=$MODE INP=$INP PATH=$PATHCSV OUT=$OUT rb=$RB depth=$DEPTH branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)@$(git -C "$REPO" rev-parse --short HEAD)"
PYTHONPATH=$REPO JAX_PLATFORMS=${JAX_PLATFORMS:-cuda} $PY -m jax_fem_am.simulation.runner \
  --config "$CFG" \
  --inp "$INP" --mesh-length-scale 1.0 \
  --build-axis z --base-side min \
  --path-file "$PATHCSV" --dt 0.005 --layers 1 --recoat-time 0 --cooling-steps 0 \
  --layer-activation-mode along_path --born-phase solid \
  --source-model legacy --beam-radius "$RB" --source-depth "$DEPTH" --absorptivity 1.0 \
  --preheat-temperature 298.0 --ambient 298.0 \
  --surface-selection exterior --surface-side-faces --bottom-thermal-bc convection \
  --quadrature-order 2 \
  --xla-linear-solver "${LINSOLVER:-pardiso}" --xla-pardiso-mode "${PARDISO_MODE:-phase23}" \
  --xla-cell-target-batch-size 131072 \
  --thermal-output-every "$OUT_EVERY" \
  "${EXTRA[@]}" \
  --output-dir "$OUT" "$@"
