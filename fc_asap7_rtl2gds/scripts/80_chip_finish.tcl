# =============================================================================
# 80_chip_finish.tcl - filler insertion and final PG connectivity.
# =============================================================================
if {[catch {
  set fillers [get_lib_cells -quiet */FILLER*]
  if {[llength $fillers] > 0} {
    create_stdcell_fillers -lib_cells $fillers
  } else {
    puts "WARN: no FILLER cells found in reference libraries"
  }
} msg]} {
  puts "WARN: filler insertion skipped: $msg"
}

catch { connect_pg_net -automatic }

catch { check_design -checks {all} }

save_block
save_lib -all
puts "INFO: chip finish complete"
