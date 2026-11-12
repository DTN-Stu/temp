# =============================================================================
# 10_lib2db.tcl - compile ASAP7 .lib (NLDM) into Synopsys .db with Library
# Compiler.  Run with:  lc_shell -f pdk/10_lib2db.tcl
# =============================================================================
set script_dir [file dirname [file normalize [info script]]]
set root [expr {[info exists ::env(PROJ_ROOT)] && $::env(PROJ_ROOT) ne "" \
                ? $::env(PROJ_ROOT) : [file normalize [file join $script_dir ..]]}]
set pdk  [expr {[info exists ::env(PDK_DIR)] && $::env(PDK_DIR) ne "" \
                ? $::env(PDK_DIR) : [file join $root pdk_data]}]

set libdir $pdk/lib
set dbdir  $pdk/db
file mkdir $dbdir

set libs [glob -nocomplain -directory $libdir *.lib]
if {[llength $libs] == 0} {
  error "No .lib files in $libdir - run pdk/fetch_pdk.sh first"
}

set ok 0
set fail 0
foreach lib $libs {
  set base [file rootname [file tail $lib]]
  set out  [file join $dbdir ${base}.db]
  if {[file exists $out] && [file mtime $out] >= [file mtime $lib]} {
    puts "SKIP  $base"
    continue
  }
  puts "===== $base"
  if {[catch {
    read_lib $lib
    write_lib $base -format db -output $out
    catch { remove_lib $base }
    incr ok
  } msg]} {
    puts "ERROR: $base : $msg"
    incr fail
  }
}

puts "============================================================"
puts "DB conversion complete: $ok ok, $fail failed"
puts "Output directory: $dbdir"
if {$fail > 0} { exit 1 }
exit
