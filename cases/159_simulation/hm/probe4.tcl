set LOG [open "C:/Users/user/AppData/Local/Temp/claude/C--Users-user/e16fef2d-3058-4e17-8273-e5048865166c/scratchpad/hm/probe4.log" w]
proc log {m} { global LOG; puts $LOG $m; flush $LOG }
proc try {label script} { if {[catch {uplevel 1 $script} err]} { log "FAIL $label: $err"; return 0 } else { log "ok   $label"; return 1 } }
proc surfs_of_solid {sid} { *createmark surfs 2 "by solids" $sid; return [hm_getmark surfs 2] }
proc planar_x_face {sid xtarget} {
  # surface of solid sid whose bbox is flat in x at xtarget (largest area)
  set best ""; set besta -1
  foreach s [surfs_of_solid $sid] {
    *createmark surfs 2 $s
    if {[catch {set bb [hm_getboundingbox surfs 2 0 0 0]}]} continue
    set dx [expr {[lindex $bb 3]-[lindex $bb 0]}]
    if {$dx < 0.01 && abs([lindex $bb 0]-$xtarget) < 0.05} {
      set a [hm_getvalue surfs id=$s dataname=area]
      if {$a > $besta} { set besta $a; set best $s }
    }
  }
  return [list $best $besta]
}
*templatefileset "D:/Altair2025/hwdesktop/templates/feoutput/abaqus/standard.3d"
*geomimport "step_ct" "C:/Users/user/Desktop/159/schema/0119.stp" "CleanupTol=-0.01" "CreationType=Parts" "DoNotMergeEdges=off" "ImportBlanked=off" "ImportFreeCurves=off" "ImportFreePoints=off" "ScaleFactor=1.0" "SkipCreationOfSolid=off" "SplitComponents=Body" "StitchingAcrossBodies=on" "TargetUnits=MPA (mm t N s)"
set x0 3753.9972470982
foreach dx {2.4 7.4 63.0 66.0 90.8} {
  set x [expr {$x0 + $dx}]
  *createmark solids 1 "all"
  *createplane 1 1 0 0 $x 0 0
  *body_splitmerge_with_plane solids 1 1
}
*createmark solids 1 "all"
set seg4 ""; set sliver ""
foreach s [hm_getmark solids 1] {
  *createmark solids 2 $s
  set bb [hm_getboundingbox solids 2 0 0 0]
  set xa [expr {[lindex $bb 0]-$x0}]; set xb [expr {[lindex $bb 3]-$x0}]
  log "solid $s x=[format %.2f $xa]..[format %.2f $xb] surfs=[llength [surfs_of_solid $s]]"
  if {$xa > 7.0 && $xa < 7.6 && $xb > 62.5} { set seg4 $s }
  if {$xa > 90.7} { set sliver $s }
}
if {$sliver != ""} { try "delete sliver" { *createmark solids 1 $sliver; *deletemark solids 1 } }
log "segment 4 solid: $seg4"
lassign [planar_x_face $seg4 [expr {$x0+7.4}]] src srca
lassign [planar_x_face $seg4 [expr {$x0+63.0}]] dst dsta
log "source face $src area $srca ; dest face $dst area $dsta"
try "create comp" { *createentity comps name=PART }
*currentcollector comps "PART"
try "automesh source (quads, autodecide)" {
  *setedgedensitylink 0
  *elementorder 1
  *createmark surfaces 1 $src
  *interactiveremeshsurf 1 2.0 1 1 2 1 1
  *set_meshfaceparams 0 1 1 0 0 1 0.5 1 1
  *automesh 0 1 1
  *storemeshtodatabase 1
  *ameshclearsurface
}
*createmark elems 1 "all"
set n2d [hm_marklength elems 1]
log "2D elements after automesh: $n2d"
set nq "?"; catch { *createmark elems 2 "by config" 104; set nq [hm_marklength elems 2] }
set nt "?"; catch { *createmark elems 2 "by config" 103; set nt [hm_marklength elems 2] }
log "quads $nq trias $nt"
try "solidmap seg4 (explicit source/dest/along, 278 layers)" {
  *solidmap_begin 0
  *solidmap_prepare_usrdataptr "SOURCE" 4
  *createmark surfs 1 $src
  *solid_prepare_entitylst surfs 0
  *solidmap_prepare_usrdataptr "DEST" 4
  *createmark surfs 1 $dst
  *solid_prepare_entitylst surfs 0
  *solidmap_prepare_usrdataptr "ALONG" 32
  *createmark solids 1 $seg4
  *solid_prepare_entitylst solids 0
  *solidmap_end 8194 278 0 0
}
*createmark elems 1 "all"
log "elements after solidmap attempt 1: [hm_marklength elems 1]"
if {[hm_marklength elems 1] == $n2d} {
  try "solidmap seg4 (solid only, 278 layers)" {
    *solidmap_begin 0
    *solidmap_prepare_usrdataptr "SOURCE" 4
    *solidmap_prepare_usrdataptr "DEST" 4
    *solidmap_prepare_usrdataptr "ALONG" 32
    *createmark solids 1 $seg4
    *solid_prepare_entitylst solids 0
    *solidmap_end 8194 278 0 0
  }
  *createmark elems 1 "all"
  log "elements after solidmap attempt 2: [hm_marklength elems 1]"
}
set nh "?"; catch { *createmark elems 2 "by config" 208; set nh [hm_marklength elems 2] }
log "hex8 elements: $nh"
try savehm { *writefile "C:/Users/user/AppData/Local/Temp/claude/C--Users-user/e16fef2d-3058-4e17-8273-e5048865166c/scratchpad/hm/0119_seg4_test.hm" 1 }
close $LOG
*quit 1
