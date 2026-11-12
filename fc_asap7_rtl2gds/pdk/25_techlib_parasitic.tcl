# =============================================================================
# 25_techlib_parasitic.tcl - optional technology-only NDM.
# Run with:  icc2_lm_shell -f pdk/25_techlib_parasitic.tcl
#
# Produces pdk_data/ndm/asap7_tech.ndm.  If a TLU+ and layer map exist they are
# attached here, and config.tcl can then use:
#     set TECH_SOURCE $PDK_TECH_NDM
# =============================================================================
set script_dir [file dirname [file normalize [info script]]]
set root [expr {[info exists ::env(PROJ_ROOT)] && $::env(PROJ_ROOT) ne "" \
                ? $::env(PROJ_ROOT) : [file normalize [file join $script_dir ..]]}]
set pdk  [expr {[info exists ::env(PDK_DIR)] && $::env(PDK_DIR) ne "" \
                ? $::env(PDK_DIR) : [file join $root pdk_data]}]

set tf   $pdk/asap7_4x.tf
set out  $pdk/ndm/asap7_tech.ndm
set tlup $pdk/tluplus/typical.tluplus
set map  $pdk/tluplus/typical.map

if {![file exists $tf]} { error "Missing $tf - run pdk/lef2tf.sh first" }
file mkdir [file dirname $out]

create_workspace tech_ws -technology $tf

if {[file exists $tlup] && [file exists $map]} {
  create_parasitic_tech -tlup $tlup -layermap $map -name typical
  puts "INFO: attached TLU+ 'typical'"
} else {
  puts "WARN: $tlup or $map missing - tech NDM will have no parasitic tech"
}

check_workspace -allow_missing
commit_workspace -force -output $out
remove_workspace
puts "DONE: $out"
exit
