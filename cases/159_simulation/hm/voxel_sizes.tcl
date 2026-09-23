# HyperMesh 2025 batch: voxel HEX8 meshes of 0119 at several sizes (E8/E9 scaling, 2026-09-23).
# Same recipe as the 2026-08-28 0.5 mm production mesh: STEP in mm, *voxel_lattice_hex_mesh mode 3
# (keep every voxel touching the solid; mesh/convert_hm_voxel.py then keeps the centre-inside ones).
#   D:\Altair2025\hwdesktop\hm\bin\win64\hmbatch.exe -tcl voxel_sizes.tcl
# Edit STEP / OUTDIR / SIZES below; one export per size: <OUTDIR>/0119_hm_voxel_<size>mm.inp
set STEP   "C:/Users/user/Desktop/paper/fastscan_solver_paper/stl/0119.stp"
set OUTDIR "C:/Users/user/Desktop/159/schema"
set SIZES  {1.0 0.4}
set TPL    "D:/Altair2025/hwdesktop/templates/feoutput/abaqus/standard.3d"

set LOG [open "$OUTDIR/0119_hm_voxel_sizes.log" w]
proc log {m} { global LOG; puts $LOG "[clock format [clock seconds] -format %H:%M:%S] $m"; flush $LOG }
proc try {label script} { if {[catch {uplevel 1 $script} err]} { log "FAIL $label: $err"; return 0 } else { log "ok   $label"; return 1 } }
# HM puts the voxels into the solid's own component, so count, export and clear by element config (HEX8 = 208)
proc count_hex {} { *createmark elems 2 "by config" 208; return [hm_marklength elems 2] }

*templatefileset $TPL
*elementtype 208 1
*geomimport "step_ct" $STEP "CleanupTol=-0.01" "CreationType=Parts" "DoNotMergeEdges=off" "ImportBlanked=off" "ImportFreeCurves=off" "ImportFreePoints=off" "ScaleFactor=1.0" "SkipCreationOfSolid=off" "SplitComponents=Body" "StitchingAcrossBodies=on" "TargetUnits=MPA (mm t N s)"
*createmark solids 1 "all"
log "solids [llength [hm_getmark solids 1]] bbox [hm_getboundingbox solids 1 0 0 0]"

foreach s $SIZES {
  set tag [string map {. p} $s]
  try "voxel $s mm mode 3" {
    *voxel_lattice_hex_mesh_init $s
    *createmark solids 1 "all"
    *voxel_lattice_hex_mesh_add_entities solids 1 3
    *voxel_lattice_hex_mesh_create
    *voxel_lattice_hex_mesh_end
  }
  set n [count_hex]
  log "size $s mm hex8: $n"
  if {$n > 0} {
    try "export $s" {
      *createstringarray 6 "APPLY_QUOTE_RULES" "EXPORTIDS_SKIP" "HMENGINEERING_XML" "HMBOMCOMMENTS_XML" "HMSUBSYSTEMCOMMENTS_XML" "HMMATCOMMENTS_XML"
      *feoutputwithdata $TPL "$OUTDIR/0119_hm_voxel_${tag}mm.inp" 0 0 2 1 6
    }
    *createmark elems 2 "all"
    *deletemark elems 2
    log "cleared, hex8 left: [count_hex]"
  }
}
log "done"
close $LOG
*quit 1
