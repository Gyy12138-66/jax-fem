#!/usr/bin/env bash
# 2017 hybrid laser-MIG CFD case, mechanics only: nodal temperature history prescribed from the CFD frames
# (Desktop/熔池温度场result/output/v01 mapped by make_prescribed_temperature.py), thermal solve skipped.
# CPU multi-core: PARDISO (MKL threads) + XLA CPU. One runner step per temperature frame, mechanics every step.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
CASE=$REPO/cases/weld_cfd2017
PY=${PY:-/home/user/miniconda3/envs/jax-fem-env/bin/python}
TAG=${TAG:-v01a}
OUT=${OUT:-$HOME/work/159/output/cfd2017_prescribed_${TAG}}
INP=${INP:-$CASE/inputs/cfd2017_plate_05mm.inp}
PATHCSV=${PATHCSV:-$CASE/inputs/cfd2017_prescribed_${TAG}_path.csv}
TNPZ=${TNPZ:-$CASE/inputs/cfd2017_prescribed_${TAG}.npz}
THREADS=${THREADS:-16}
DT0=$(awk -F, 'NR==2{print $1}' "$PATHCSV")
export JAX_PLATFORMS=cpu MKL_NUM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS PYTHONUNBUFFERED=1
mkdir -p "$OUT"
cd "$REPO"
echo "[$(date -u +%FT%TZ)] cfd2017 prescribed-T mechanics TAG=$TAG INP=$INP PATH=$PATHCSV T=$TNPZ OUT=$OUT threads=$THREADS branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)@$(git -C "$REPO" rev-parse --short HEAD)" | tee "$OUT/run.log"
PYTHONPATH=$REPO $PY -m jax_fem_am.simulation.runner \
  --config "$CASE/inputs/material/a7n01_material_config.json" \
  --inp "$INP" --mesh-length-scale 1.0 \
  --build-axis z --base-side min \
  --path-file "$PATHCSV" --dt "$DT0" --layers 1 --recoat-time 0 --cooling-steps 0 \
  --layer-activation-mode front --born-phase solid \
  --prescribed-temperature-file "$TNPZ" \
  --preheat-temperature 298.0 --ambient 298.0 \
  --surface-selection exterior --bottom-thermal-bc convection \
  --quadrature-order 2 \
  --mechanics-every 1 --no-release-after-cooling \
  --bottom-mechanics-bc "${MECHBC:-symmetry_plane}" --symmetry-plane-axis "${SYMAXIS:-y}" --symmetry-plane-side min \
  --mechanics-rel-tol 5e-5 --mechanics-line-search --mechanics-max-cuts 3 \
  --reset-plastic-on-solidify --elastic-melt \
  --xla-linear-solver "${LINSOLVER:-pardiso}" --xla-pardiso-mode "${PARDISO_MODE:-phase23}" \
  --thermal-output-every 1 --mechanics-output-every 1 \
  --output-dir "$OUT" "$@" 2>&1 | tee -a "$OUT/run.log"
