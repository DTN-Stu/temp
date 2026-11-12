# =============================================================================
# 20_read_design.tcl - read the RTL filelist + wrapper, elaborate and link.
# =============================================================================
set lib_path $::WORK_DIR/$::LIB_NAME.nlib

proc _abs_if_rel {base p} {
  if {[file pathtype $p] eq "absolute"} { return $p }
  return [file normalize [file join $base $p]]
}

set rtl_files {}
set incdirs   $::RTL_INCDIRS
set defines   $::VERILOG_DEFINES

if {![file exists $::RTL_FILELIST]} {
  error "RTL filelist not found: $::RTL_FILELIST  (run scripts/00_extract_rtl.sh)"
}

set fh [open $::RTL_FILELIST r]
while {[gets $fh line] >= 0} {
  set line [string trim $line]
  if {$line eq ""} { continue }
  if {[string match "#*" $line] || [string match "//*" $line]} { continue }
  if {[string match "+incdir+*" $line]} {
    lappend incdirs [_abs_if_rel $::RTL_DIR [string range $line 8 end]]
    continue
  }
  if {[string match "+define+*" $line]} {
    foreach d [split [string range $line 8 end] "+"] {
      if {$d ne ""} { lappend defines $d }
    }
    continue
  }
  if {[string match "-*" $line]} { continue }
  lappend rtl_files [_abs_if_rel $::RTL_DIR $line]
}
close $fh

foreach f $::EXTRA_RTL {
  if {[file exists $f]} {
    lappend rtl_files $f
  } else {
    error "Wrapper RTL not found: $f"
  }
}
set incdirs [lsort -unique $incdirs]
set defines [lsort -unique $defines]

# Let read_verilog resolve `include "pe_defines.svh"
if {[info exists ::search_path]} {
  set ::search_path [concat $::search_path $incdirs]
} else {
  set ::search_path $incdirs
}

set rv_opts [list -library [file tail $lib_path] -design $::DESIGN -top $::DESIGN]
lappend rv_opts -sv
if {[llength $defines]} { lappend rv_opts -define $defines }

puts "INFO: read_verilog options: $rv_opts"
puts "INFO: [llength $rtl_files] RTL files, incdirs: $incdirs"

read_verilog {*}$rv_opts {*}$rtl_files
link_block

catch { check_design -checks {unresolved} }
puts "INFO: design linked as $::DESIGN"
