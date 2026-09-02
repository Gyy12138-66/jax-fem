#!/usr/bin/env bash
# Copy the SMALL result files of a finished comparison into results/ (next to
# this script) so the repo carries the evidence without the VTUs.
#   bash collect_results.sh [OUT=<output dir>]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-/home/user/work/159/output/v2_cube_smoke_smoke2/solver_compare}"
RES="$HERE/results"
mkdir -p "$RES"
cp -f "$OUT/solver_compare.json" "$RES/" 2>/dev/null
cp -f "$OUT/solver_compare.log" "$RES/" 2>/dev/null
for d in "$OUT"/*/; do
  arm=$(basename "$d")
  [ -f "$d/profile.json" ] || continue
  mkdir -p "$RES/$arm"
  cp -f "$d/profile.json" "$d/used_config.json" "$d/thermal_energy_ledger_summary.json" "$RES/$arm/" 2>/dev/null
  { head -40 "$d/run.log"; echo "..."; tail -30 "$d/run.log"; } > "$RES/$arm/run_log_head_tail.txt" 2>/dev/null
  grep -E "^global_step=" "$d/run.log" | sed -E 's/ laser_center.*//' > "$RES/$arm/summary_lines.txt" 2>/dev/null
done
echo "collected into $RES:"; find "$RES" -type f | sort
