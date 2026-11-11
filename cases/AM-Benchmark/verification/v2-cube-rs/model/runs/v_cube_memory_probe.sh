#!/usr/bin/env bash
# ============================================================================
# Memory / cost probe for the PRODUCTION cube mesh (D-V2-07 graded substrate).
# Activates ALL 50 slabs at once (a path row carrying layer=50 activates every
# layer <= 50 under layer_on_scan) so the very first mechanics solve is the
# worst case: substrate + full 10x10x10 mm part, ~1.0 M mechanics DOF.
# Measures max RSS (/usr/bin/time -v), peak GPU memory (nvidia-smi sampler),
# and the wall time of one full-size mechanics solve and one release solve.
#   bash v_cube_memory_probe.sh            (GPU env, PARDISO, same contract flags as smoke2)
# ============================================================================
set -u
REPO="${REPO:-/home/user/work/d11_B_tree}"
PYBIN="${PYBIN:-/home/user/miniconda3/envs/jax-fem-gpu/bin/python}"
PLATFORM="${PLATFORM:-gpu}"
OUT="${OUT:-/home/user/work/159/output/v2_cube_prod_memprobe}"
BATCH="${BATCH:-}"   # --xla-cell-target-batch-size override (GPU assembly chunk); empty = wrapper default 262144
# XLA_PYTHON_CLIENT_ALLOCATOR=platform makes XLA free device memory between phases instead of
# keeping it in its BFC cache (the release solve builds a second problem instance on top of the
# build one; with the cache the 3rd probe OOMed at 12.4 GB resident + release, 2026-08-27)
[ -n "${ALLOCATOR:-}" ] && export XLA_PYTHON_CLIENT_ALLOCATOR="$ALLOCATOR"
PRE_SMOKE="${PRE_SMOKE:-/home/user/work/159/output/v2_cube_smoke_smoke2/preflight}"
M="$REPO/cases/AM-Benchmark/verification/v2-cube-rs/model"
export PYTHONPATH="$REPO" JAX_PLATFORM_NAME="$PLATFORM" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
       MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}" OMP_NUM_THREADS="${MKL_NUM_THREADS:-16}"
mkdir -p "$OUT"; cd "$REPO"
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$OUT/probe.log"; }

SUB_Z=6.0e-3; PART_XY=10.0e-3; SUB_XY="${SUB_XY:-30.0e-3}"   # SUB_XY=20.0e-3 probes the reduced-footprint variant
SUB_TAG=$(python3 -c "print(int(round(float('$SUB_XY')*1e3)))")
MESH="$OUT/v2_cube_prod_graded_sub${SUB_TAG}_c3d8.inp"
if [ ! -f "$MESH" ]; then
  say "generating production mesh (10x10x10 part @200um, ${SUB_TAG}x${SUB_TAG}x6 substrate graded 1.4)"
  "$PYBIN" "$M/make_v2_mesh_cube.py" --res 2.0e-4 --part-xy 10.0e-3 --part-z 10.0e-3 \
     --sub-xy "$SUB_XY" --sub-z 6.0e-3 --sub-grading 1.4 --output "$MESH" 2>&1 | tee -a "$OUT/probe.log"
fi
# three rows at the part centre, all carrying layer=50 (slab top z = sub_z + 50*200um)
ZTOP=$(python3 -c "print($SUB_Z + 50*2.0e-4)")
C=$(python3 -c "print(0.5*$SUB_XY)")
cat > "$OUT/path_probe.csv" <<CSV
time,x,y,z,power,laser_on,layer,hatch,mode,front_coord
0.1,$C,$C,$ZTOP,140,1,50,1,scan,$ZTOP
0.2,$C,$C,$ZTOP,140,1,50,1,scan,$ZTOP
0.3,$C,$C,$ZTOP,0,0,50,0,recoat,$ZTOP
CSV
mapfile -t ARGV < <(python3 -c "import json;print('\n'.join(json.load(open('$PRE_SMOKE/runner_contract.json'))['argv']))")
# GPU memory sampler
( while true; do echo "$(date -u +%H:%M:%S) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"; sleep 2; done ) > "$OUT/gpu_mem.log" 2>&1 &
SAMPLER=$!
rm -rf "$OUT/run"   # the ledger recorder refuses to overwrite a previous attempt's artifacts
say "probe start: platform=$PLATFORM python=$PYBIN batch=${BATCH:-default} allocator=${ALLOCATOR:-default}"
/usr/bin/time -v "$PYBIN" -m jax_fem_am.simulation.runner "${ARGV[@]}" \
  --inp "$MESH" --path-file "$OUT/path_probe.csv" --layers 50 --support-thickness "$SUB_Z" \
  --dt 0.1 --mechanics-every 1 --cooling-steps 1 --cooling-dt 10.0 \
  --release-after-cooling --release-anchor-mode rigid_body \
  --release-cut-box 0 "$SUB_XY" 0 "$SUB_XY" 0 "$SUB_Z" \
  --thermal-output-every 100000 --mechanics-output-every 100000 --summary-every 1 \
  --xla-platform "$PLATFORM" ${BATCH:+--xla-cell-target-batch-size "$BATCH"} \
  --output-dir "$OUT/run" --profile-json "$OUT/run/profile.json" --profile-label "cube-prod-memprobe" \
  > "$OUT/run.log" 2> "$OUT/time.log"
RC=$?
kill $SAMPLER 2>/dev/null
say "probe end rc=$RC"
grep -E "Maximum resident|Elapsed|Exit status" "$OUT/time.log" | tee -a "$OUT/probe.log"
say "peak GPU MiB: $(awk '{print $2}' "$OUT/gpu_mem.log" | sort -n | tail -1)"
grep -E "^global_step=|release_u_max|deactivating" "$OUT/run.log" | sed -E 's/ laser_center.*//' | cut -c1-200 | tee -a "$OUT/probe.log"
"$PYBIN" - "$OUT/run/profile.json" <<'PY' 2>/dev/null | tee -a "$OUT/probe.log"
import json,sys; p=json.load(open(sys.argv[1])); ss=p['stage_seconds']; sc=p['stage_calls']
print('wall',round(p['wall_seconds'],1),'assembly',round(ss.get('assembly',0),1),'solver',round(ss.get('solver',0),1),'newton',round(p['meta'].get('newton_wall_seconds',0),1),'nonlinear calls',sc.get('nonlinear_solve'),'solver calls',sc.get('solver'))
PY
say "MEMPROBE_DONE $OUT"
