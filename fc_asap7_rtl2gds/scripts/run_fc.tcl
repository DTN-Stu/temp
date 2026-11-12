# =============================================================================
# run_fc.tcl - master driver.
#
#   fc_shell -f scripts/run_fc.tcl
#   fc_shell -f scripts/run_fc.tcl -from 40_floorplan -to 90_write_outputs
#
# Stage names:
#   00_setup 10_create_lib 20_read_design 30_scenarios 40_floorplan
#   50_compile_fusion 60_clock_opt 70_route 80_chip_finish 90_write_outputs
# =============================================================================
set SCRIPT_DIR [file dirname [file normalize [info script]]]
source [file join $SCRIPT_DIR .. config.tcl]

if {![info exists argv]} { set argv {} }

set STAGES {
  00_setup
  10_create_lib
  20_read_design
  30_scenarios
  40_floorplan
  50_compile_fusion
  60_clock_opt
  70_route
  80_chip_finish
  90_write_outputs
}

set from_stage [lindex $STAGES 0]
set to_stage   [lindex $STAGES end]

for {set i 0} {$i < [llength $argv]} {incr i} {
  set a [lindex $argv $i]
  switch -- $a {
    -from { set from_stage [lindex $argv [expr {$i + 1}]]; incr i }
    -to   { set to_stage   [lindex $argv [expr {$i + 1}]]; incr i }
    default { puts "WARN: ignoring argument '$a'" }
  }
}

set start_idx [lsearch -exact $STAGES $from_stage]
if {$start_idx < 0} {
  puts "WARN: unknown -from '$from_stage'; starting at 00_setup"
  set start_idx 0
}
set stop_idx [lsearch -exact $STAGES $to_stage]
if {$stop_idx < 0} {
  puts "WARN: unknown -to '$to_stage'; running to the end"
  set stop_idx [expr {[llength $STAGES] - 1}]
}
if {$stop_idx < $start_idx} {
  error "invalid range: $from_stage .. $to_stage"
}

# Resuming after library creation: reopen the saved library and block.
if {$start_idx > 1} {
  set _lib_path $::WORK_DIR/$::LIB_NAME.nlib
  if {[file exists $_lib_path]} {
    puts "INFO: resuming from $_lib_path"
    catch { open_lib $_lib_path }
    catch { open_block $_lib_path:$::DESIGN }
  } else {
    puts "WARN: $_lib_path not found; include 10_create_lib in the run"
  }
}

set n [llength $STAGES]
for {set i 0} {$i < $n} {incr i} {
  if {$i < $start_idx || $i > $stop_idx} { continue }
  set stage [lindex $STAGES $i]
  puts "\n============================================================"
  puts "\[[expr {$i + 1}]/$n\] $stage"
  puts "============================================================"
  source [file join $SCRIPT_DIR ${stage}.tcl]
}

puts "\nINFO: flow finished ($from_stage .. $to_stage)"
exit
