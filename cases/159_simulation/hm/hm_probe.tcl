set f [open "C:/Users/user/AppData/Local/Temp/claude/C--Users-user/e16fef2d-3058-4e17-8273-e5048865166c/scratchpad/hm/hm_probe.log" w]
puts $f "tcl [info patchlevel]"
catch {puts $f "hm_version [hm_info -appinfo HM_VERSION]"} e1
puts $f "e1 $e1"
catch {puts $f "altair_home [hm_info -appinfo ALTAIR_HOME]"} e2
puts $f "e2 $e2"
catch {
  *createentity comps name=probe
  *createmark comps 1 "all"
  puts $f "comps [llength [hm_getmark comps 1]]"
} e3
puts $f "e3 $e3"
catch {puts $f "solidblock [catch {*solidblock 0 0 0 10 0 0 0 10 0 0 0 10} rc] rc=$rc"} e4
puts $f "e4 $e4"
catch {*createmark solids 1 "all"; puts $f "solids [llength [hm_getmark solids 1]]"} e5
puts $f "e5 $e5"
close $f
*quit 1
