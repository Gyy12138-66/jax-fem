set LOG [open "C:/Users/user/AppData/Local/Temp/claude/C--Users-user/e16fef2d-3058-4e17-8273-e5048865166c/scratchpad/hm/probe2.log" w]
proc log {m} { global LOG; puts $LOG $m; flush $LOG }
proc try {label script} { if {[catch {uplevel 1 $script} err]} { log "FAIL $label: $err"; return 0 } else { log "ok   $label"; return 1 } }
*templatefileset "D:/Altair2025/hwdesktop/templates/feoutput/abaqus/standard.3d"
try import {
  *geomimport "step_ct" "C:/Users/user/Desktop/159/schema/0119.stp" "CleanupTol=-0.01" "CreationType=Parts" "DoNotMergeEdges=off" "ImportBlanked=off" "ImportFreeCurves=off" "ImportFreePoints=off" "ScaleFactor=1.0" "SkipCreationOfSolid=off" "SplitComponents=Body" "StitchingAcrossBodies=on" "TargetUnits=MPA (mm t N s)"
}
*createmark solids 1 "all"
set sol [hm_getmark solids 1]
log "solids: $sol"
try bbox_solids { log "bbox solids: [hm_getboundingbox solids 1 0 0 0]" }
*createmark surfs 1 "all"
set surfs [hm_getmark surfs 1]
log "surfaces: [llength $surfs]"
*createmark lines 1 "all"
log "lines: [llength [hm_getmark lines 1]]"
foreach s $surfs {
  *createmark surfs 2 $s
  set bb "?"; catch { set bb [hm_getboundingbox surfs 2 0 0 0] }
  set ar "?"; catch { set ar [hm_getvalue surfs id=$s dataname=area] }
  set ty "?"; catch { set ty [hm_getvalue surfs id=$s dataname=surfacetype] }
  log "surf $s area $ar type $ty bbox $bb"
}
try count_before { log "solid volume: [hm_getvalue solids id=[lindex $sol 0] dataname=volume]" }
try remove_holes {
  *createmark surfs 1 "all"
  *remove_solid_holes surfaces 1 3.40282347e+38 3.40282347e+38 0 1 0
}
*createmark surfs 1 "all"
log "surfaces after hole removal: [llength [hm_getmark surfs 1]]"
try volume_after { log "solid volume after: [hm_getvalue solids id=[lindex $sol 0] dataname=volume]" }
try savehm { *writefile "C:/Users/user/AppData/Local/Temp/claude/C--Users-user/e16fef2d-3058-4e17-8273-e5048865166c/scratchpad/hm/0119_noholes.hm" 1 }
close $LOG
*quit 1
