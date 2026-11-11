#!/bin/bash
# Lane gates for merging into test (LANES.md). No single environment runs everything:
#   jax-fem-gpu  (~/miniconda3)  : pyamg + CUDA; no date-time format checker  -> unit, solver, meltpool, gpu, integration
#   jax-fem-env  (~/miniforge3)  : the pipeline's CPU-side env (preflight, check_159), rfc3339-validator, no pyamg
#                                   -> contract, with JAX_PLATFORMS=cpu (Kaess G0-v2 declares the CPU platform)
# Tests marked cpu_platform are skipped (with the reason) outside a CPU backend; see tests/conftest.py.
# usage: bash tests/run_gates.sh [out_dir]      (run from the worktree root; REPO=... to point elsewhere)
cd "${REPO:-$(dirname "$0")/..}" || exit 1
D=${1:-/home/user/work/159/output/gates_$(git rev-parse --abbrev-ref HEAD)_$(date +%Y%m%d_%H%M)}; mkdir -p "$D"
GPU=/home/user/miniconda3/envs/jax-fem-gpu/bin/python
CPU=/home/user/miniforge3/envs/jax-fem-env/bin/python
S=$D/summary.txt
echo "== $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD) dirty=$(git status --short | wc -l)" > "$S"
run() { name=$1; shift; echo "== $name start $(date +%T)" >> "$S"; "$@" > "$D/$name.log" 2>&1; rc=$?
        echo "   rc=$rc $(grep -E '^(FAILED|ERROR)|[0-9]+ (passed|failed|skipped)|^Ran |^OK' "$D/$name.log" | tail -4 | tr '\n' ' ')" >> "$S"; }
run unit     env JAX_PLATFORMS=cpu PYTHONPATH=. $GPU -m pytest tests/unit -v -p no:cacheprovider -rfs
run solver   env JAX_PLATFORMS=cpu PYTHONPATH=. $GPU -m pytest tests -m solver -v -p no:cacheprovider -rfs
run meltpool env JAX_PLATFORMS=cpu PYTHONPATH=. $GPU -m pytest tests -m meltpool -v -p no:cacheprovider -rfs
run gpu      env PYTHONPATH=. $GPU -m pytest tests -m gpu -v -p no:cacheprovider -rfs
run contract env JAX_PLATFORMS=cpu PYTHONPATH=. $CPU -m pytest tests/contract -v -p no:cacheprovider -rfs
run integ    env JAX_PLATFORMS=cpu PYTHONPATH=. $GPU -m pytest tests/integration -v -p no:cacheprovider -rfs
echo "== GATES_DONE $(date +%T)" >> "$S"
cat "$S"
