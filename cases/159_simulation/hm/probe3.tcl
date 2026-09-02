set LOG [open "C:/Users/user/AppData/Local/Temp/claude/C--Users-user/e16fef2d-3058-4e17-8273-e5048865166c/scratchpad/hm/probe3.log" w]
proc log {m} { global LOG; puts $LOG $m; flush $LOG }
proc try {label script} { if {[catch {uplevel 1 $script} err]} { log "FAIL $label: $err"; return 0 } else { log "ok   $label"; return 1 } }
*templatefileset "D:/Altair2025/hwdesktop/templates/feoutput/abaqus/standard.3d"
*geomimport "step_ct" "C:/Users/user/Desktop/159/schema/0119.stp" "CleanupTol=-0.01" "CreationType=Parts" "DoNotMergeEdges=off" "ImportBlanked=off" "ImportFreeCurves=off" "ImportFreePoints=off" "ScaleFactor=1.0" "SkipCreationOfSolid=off" "SplitComponents=Body" "StitchingAcrossBodies=on" "TargetUnits=MPA (mm t N s)"
set x0 3753.9972470982
foreach dx {2.4 7.4 63.0 66.0 90.8} {
  set x [expr {$x0 + $dx}]
  try "cut at $dx" {
    *createmark solids 1 "all"
    *createplane 1 1 0 0 $x 0 0
    *body_splitmerge_with_plane solids 1 1
  }
}
*createmark solids 1 "all"
set sol [hm_getmark solids 1]
log "solids after cuts: [llength $sol]"
foreach s $sol {
  *createmark solids 2 $s
  set bb "?"; catch { set bb [hm_getboundingbox solids 2 0 0 0] }
  set vol "?"; catch { set vol [hm_getvalue solids id=$s dataname=volume] }
  set m1 "?"; catch { set m1 [hm_getvalue solids id=$s dataname=mappable] }
  set m2 "?"; catch { set m2 [hm_getvalue solids id=$s dataname=mappablestate] }
  set m3 "?"; catch { set m3 [hm_getvalue solids id=$s dataname=mappingtype] }
  *createmark surfs 3 "by solid" $s
  set ns "?"; catch { set ns [llength [hm_getmark surfs 3]] }
  set xr "?"; if {$bb != "?"} { set xr [format "%.2f..%.2f" [expr {[lindex $bb 0]-$x0}] [expr {[lindex $bb 3]-$x0}]] }
  log "solid $s x=$xr vol=$vol surfs=$ns mappable=$m1 state=$m2 type=$m3"
}
try mapcount { log "mappable count: [*solidmap_evaluate_mappable_count]" }
try savehm { *writefile "C:/Users/user/AppData/Local/Temp/claude/C--Users-user/e16fef2d-3058-4e17-8273-e5048865166c/scratchpad/hm/0119_cut.hm" 1 }
close $LOG
*quit 1
